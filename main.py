import os
import json
import pandas as pd
import click
import threading
import hashlib
import queue
import logging
from typing import Any, cast
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rich.console import Console
from dotenv import load_dotenv
from llm import LLM, AuthenticationError
from database import Database
from schemas import EVALUATION_SCHEMA
from logger import (
    setup_logger,
    log_exception,
    log_api_call,
    log_cost_summary,
    log_run_summary,
)

# Load environment variables from .env file
load_dotenv()

console = Console()
db = Database()
logger = logging.getLogger("benchmarking")


class ThreadSafeCounter:
    def __init__(self, initial_value=0.0):
        self._value = initial_value
        self._lock = threading.Lock()

    def add(self, amount):
        with self._lock:
            self._value += amount

    def get(self):
        with self._lock:
            return self._value


class ThreadSafeProgressBar:
    def __init__(self, total, desc, position=0):
        self._pbar = tqdm(total=total, desc=desc, position=position)
        self._lock = threading.Lock()

    def update(self, n=1):
        with self._lock:
            self._pbar.update(n)

    def set_postfix(self, **kwargs):
        with self._lock:
            self._pbar.set_postfix(**kwargs)

    def refresh(self):
        with self._lock:
            self._pbar.refresh()

    def close(self):
        with self._lock:
            self._pbar.close()


def get_question_id(question_text):
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()


def get_all_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def clean_json_response(response):
    """Remove code fences from JSON response and parse it."""
    # Strip leading/trailing whitespace
    response = response.strip()

    # Check for code fences and extract content
    if response.startswith("```"):
        # Find the end of the first line (after ```json or ```)
        lines = response.split("\n")
        if len(lines) >= 2:
            # Check if it's a json code block
            if lines[0].strip().lower() in ["```json", "```"]:
                # Remove the opening fence and any trailing closing fence
                content = "\n".join(lines[1:])
                # Remove closing fence if present
                if content.strip().endswith("```"):
                    content = content.rsplit("```", 1)[0].rstrip()
                return content
    return response


def generate_answer(question_data, student_model, student_prompt_template):
    """Generate student answer (producer function)"""
    question = question_data.get("question")
    question_id = question_data.get("question_id")
    usage_records = []

    # Get student answer
    try:
        logger.debug(f"Generating answer for question {question_id[:8]}...")
        student_answer, student_usage = student_model.generate_response(
            [
                {"role": "system", "content": student_prompt_template},
                {"role": "user", "content": question},
            ],
            # response_format=STUDENT_ANSWER_SCHEMA,
        )

        # try:
        #     student_answer = json.loads(clean_json_response(student_answer))
        # except Exception:
        #     pass

        if student_usage:
            student_usage["model"] = student_model.model_name
            student_usage["role"] = "student"
            usage_records.append(student_usage)

        log_api_call(
            logger,
            student_model.model_name,
            f"Generate answer for Q:{question_id[:8]}",
            success=True,
            details=f"QuestionID: {question_id}, Cost: ${student_usage.get('cost', 0):.4f}",
        )

        return {
            "question_data": question_data,
            "student_answer": student_answer,
            "usage_records": usage_records,
            "error": None,
        }
    except AuthenticationError as e:
        error_msg = f"Authentication Error: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        console.print(
            "[yellow]Hint: This usually means your API key is invalid or you have run out of credits on OpenRouter.[/yellow]"
        )
        logger.error(
            f"Authentication error for question {question_id[:8]}: {error_msg}"
        )
        return {"question_data": question_data, "error": error_msg, "usage_records": []}
    except Exception as e:
        error_msg = f"Student model error: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        log_exception(logger, e, f"Generating answer for question {question_id[:8]}")
        return {"question_data": question_data, "error": error_msg, "usage_records": []}


def evaluate_answer(
    answer_data,
    evaluator_models,
    prompt_template,
    evaluator_executor,
    run_id,
):
    """Evaluate a generated answer (consumer function)"""
    if answer_data.get("error"):
        return {
            "question_id": answer_data["question_data"]["question_id"],
            "error": answer_data["error"],
        }, answer_data["usage_records"]

    question_data = answer_data["question_data"]
    student_answer = answer_data["student_answer"]
    usage_records = answer_data["usage_records"].copy()

    question = question_data.get("question")
    original_answer = question_data.get("solution")

    # Ensure student_answer is a dict for formatting
    # if isinstance(student_answer, str):
    #     try:
    #         student_answer = json.loads(student_answer)
    #     except json.JSONDecodeError:
    #         student_answer = {
    #             "step_by_step_reasoning": "N/A",
    #             "final_answer": student_answer,
    #         }

    # Prepare evaluation prompt
    try:
        prompt = prompt_template.format(
            question=question, answer=student_answer, original_answer=original_answer
        )
    except KeyError as e:
        error_msg = f"Prompt formatting error: Missing key {e} in prompt template. Check for unescaped curly braces in evaluation_prompt.md"
        console.print(f"[red]{error_msg}[/red]")
        logger.error(error_msg)
        return {"error": error_msg}, usage_records

    # Run evaluators in parallel
    evaluations = {}

    # Check for existing evaluations in the database to avoid redundant API calls
    # This is useful when resuming a run where some evaluators finished but others didn't
    from database import Evaluation

    session = db.get_session()
    try:
        # Check for existing evaluations for THIS run to avoid redundant API calls
        existing_evals = (
            session.query(Evaluation)
            .filter_by(
                run_id=run_id,
                question_id=question_data.get("question_id"),
                student_answer=student_answer,
            )
            .all()
        )
        for e in existing_evals:
            raw_eval_str = e.raw_evaluation
            evaluations[e.evaluator_model] = {
                "result": (
                    json.loads(str(raw_eval_str)) if raw_eval_str is not None else {}
                ),
                "cost": 0.0,  # Cost already accounted for in DB
                "already_in_db": True,
            }
    finally:
        session.close()

    def run_evaluator(eval_model):
        # Skip if already evaluated by this model
        if eval_model.model_name in evaluations:
            logger.debug(
                f"Skipping evaluator {eval_model.model_name} for question {question_data.get('question_id', '')[:8]} (already in DB)"
            )
            return evaluations[eval_model.model_name]["result"], {
                "cost": 0.0,
                "already_in_db": True,
            }

        try:
            logger.debug(
                f"Running evaluator {eval_model.model_name} for question {question_data.get('question_id', '')[:8]}..."
            )
            response, eval_usage = eval_model.generate_response(
                [
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": prompt},
                ],
                response_format=EVALUATION_SCHEMA,
            )
            if eval_usage:
                eval_usage["model"] = eval_model.model_name
                eval_usage["role"] = "evaluator"

            # Clean response by removing code fences if present, then parse JSON
            cleaned_response = clean_json_response(response)
            try:
                parsed_response = json.loads(cleaned_response)
                log_api_call(
                    logger,
                    eval_model.model_name,
                    f"Evaluate Q:{question_data.get('question_id', '')[:8]}",
                    success=True,
                    details=f"QuestionID: {question_data.get('question_id', '')}, Cost: ${eval_usage.get('cost', 0):.4f}",
                )
                return parsed_response, eval_usage
            except json.JSONDecodeError as je:
                logger.warning(
                    f"Failed to parse JSON from evaluator {eval_model.model_name}: {str(je)}"
                )
                return response, eval_usage
        except Exception as e:
            log_exception(logger, e, f"Running evaluator {eval_model.model_name}")
            return f"Error: {str(e)}", {}

    future_to_model = {
        evaluator_executor.submit(run_evaluator, m): m.model_name
        for m in evaluator_models
    }
    for future in as_completed(future_to_model):
        model_name = future_to_model[future]
        res, usage = future.result()
        evaluations[model_name] = {
            "result": res,
            "cost": usage.get("cost", 0.0),
            "already_in_db": usage.get("already_in_db", False),
        }
        if usage:
            usage_records.append(usage)

    return {
        "question": question,
        "student_answer": student_answer,
        "original_answer": original_answer,
        "evaluations": evaluations,
    }, usage_records


@click.command()
@click.option("--models-csv", default="models.csv", help="Path to models CSV")
@click.option(
    "--evaluators-csv", default="evalulator.csv", help="Path to evaluators CSV"
)
@click.option(
    "--data-dir", default="benchmarking_data", help="Path to benchmarking data"
)
@click.option(
    "--prompt-file",
    default="evaluation_prompt.md",
    help="Path to evaluation prompt template",
)
@click.option(
    "--student-prompt-file",
    default="student_prompt.md",
    help="Path to student prompt template",
)
@click.option(
    "--class-level",
    default=None,
    help="Class/Level to evaluate (e.g., 'Class 10')",
)
def main(
    models_csv,
    evaluators_csv,
    data_dir,
    prompt_file,
    student_prompt_file,
    class_level,
):
    global logger
    # Setup logging first
    logger = setup_logger("benchmarking", "logs")
    logger.info("Starting benchmarking run")

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        error_msg = "OPENROUTER_API_KEY not found in environment or .env file."
        console.print(f"[bold red]Error: {error_msg}[/bold red]")
        logger.error(error_msg)
        return

    # Load models
    try:
        models_df = pd.read_csv(models_csv)
        evaluators_df = pd.read_csv(evaluators_csv)
        logger.info(
            f"Loaded {len(models_df)} student models and {len(evaluators_df)} evaluator models"
        )
    except Exception as e:
        log_exception(
            logger, e, f"Loading model CSVs from {models_csv} and {evaluators_csv}"
        )
        console.print(f"[bold red]Error loading models: {e}[/bold red]")
        return

    # Select Student Model
    console.print("[bold blue]Select Student Model:[/bold blue]")
    for i, model in enumerate(models_df["model"]):
        console.print(f"{i + 1}. {model}")

    model_idx = click.prompt("Enter model number", type=int) - 1
    student_model_name = models_df.iloc[model_idx]["model"]
    logger.info(f"Selected student model: {student_model_name}")
    try:
        student_model = LLM(student_model_name)
        logger.info(f"Successfully initialized student model: {student_model_name}")
    except AuthenticationError as e:
        error_msg = f"Authentication Error: {e}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        console.print(
            "[yellow]Please check if your OPENROUTER_API_KEY is valid and has sufficient credits.[/yellow]"
        )
        logger.error(error_msg)
        return
    except Exception as e:
        error_msg = f"Error initializing student model: {e}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        log_exception(logger, e, f"Initializing student model {student_model_name}")
        return

    # Select Evaluators
    num_evaluators = click.prompt("How many evaluators do you want to use?", type=int)

    console.print("[bold blue]Select Evaluators:[/bold blue]")
    for i, model in enumerate(evaluators_df["model"]):
        console.print(f"{i + 1}. {model}")

    evaluator_indices = click.prompt(
        f"Enter {num_evaluators} evaluator numbers separated by space", type=str
    ).split()

    if len(evaluator_indices) != num_evaluators:
        error_msg = f"Error: You must select exactly {num_evaluators} evaluators."
        console.print(f"[red]{error_msg}[/red]")
        logger.error(error_msg)
        return

    try:
        selected_evaluator_names = [
            evaluators_df.iloc[int(idx) - 1]["model"] for idx in evaluator_indices
        ]
        logger.info(f"Selected evaluators: {', '.join(selected_evaluator_names)}")
    except (ValueError, IndexError) as e:
        error_msg = "Error: Invalid evaluator selection."
        console.print(f"[red]{error_msg}[/red]")
        log_exception(logger, e, "Selecting evaluators")
        return

    try:
        evaluator_models = [LLM(name) for name in selected_evaluator_names]
        logger.info(
            f"Successfully initialized {len(evaluator_models)} evaluator models"
        )
    except AuthenticationError as e:
        error_msg = f"Authentication Error during evaluator initialization: {e}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        logger.error(error_msg)
        return
    except Exception as e:
        error_msg = f"Error initializing evaluator models: {e}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        log_exception(logger, e, "Initializing evaluator models")
        return

    # Select Class/Level
    available_classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    if not class_level:
        console.print("[bold blue]Select Class/Level:[/bold blue]")
        console.print("0. All Classes")
        for i, cls in enumerate(available_classes):
            console.print(f"{i + 1}. {cls}")
        cls_idx = click.prompt("Enter class number", type=int, default=0) - 1
        if cls_idx >= 0:
            selected_data_dir = os.path.join(data_dir, available_classes[cls_idx])
            logger.info(f"Selected class: {available_classes[cls_idx]}")
        else:
            selected_data_dir = data_dir
            logger.info("Selected: All classes")
    else:
        # Try to match class_level with available_classes
        matches = [
            cls for cls in available_classes if class_level.lower() in cls.lower()
        ]
        if matches:
            if len(matches) > 1:
                console.print(
                    f"[yellow]Multiple matches for '{class_level}': {matches}. Using the first one: {matches[0]}[/yellow]"
                )
                logger.warning(
                    f"Multiple matches for '{class_level}': {matches}. Using: {matches[0]}"
                )
            selected_data_dir = os.path.join(data_dir, matches[0])
            logger.info(f"Selected class from CLI: {matches[0]}")
        else:
            error_msg = f"No match found for class/level '{class_level}'."
            console.print(f"[red]Error: {error_msg}[/red]")
            logger.error(error_msg)
            return

    # Parallel Configuration
    student_workers = click.prompt(
        "How many parallel student instances do you want to run?", type=int, default=1
    )
    evaluator_workers = click.prompt(
        "How many parallel evaluator threads do you want to run?", type=int, default=5
    )
    logger.info(
        f"Parallel configuration - Student workers: {student_workers}, Evaluator workers: {evaluator_workers}"
    )

    # Load prompt templates
    try:
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
        logger.info(f"Loaded evaluation prompt from {prompt_file}")

        with open(student_prompt_file, "r") as f:
            student_prompt_template = f.read()
        logger.info(f"Loaded student prompt from {student_prompt_file}")
    except Exception as e:
        log_exception(logger, e, "Loading prompt templates")
        console.print(f"[red]Error loading prompts: {e}[/red]")
        return

    # Get all questions and populate database
    json_files = get_all_json_files(selected_data_dir)
    logger.info(f"Found {len(json_files)} JSON files in {selected_data_dir}")
    all_questions_map = {}
    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        q_text = item.get("question", "")
                        solution = item.get("solution")

                        # Handle new schema with options and answer
                        if "options" in item and "answer" in item:
                            options = item["options"]
                            if isinstance(options, dict):
                                # Sort options by key to maintain consistent order (a, b, c, d)
                                sorted_options = sorted(options.items())
                                options_text = "\n".join(
                                    [f"  {k.upper()}) {v}" for k, v in sorted_options]
                                )
                                q_text = f"{q_text}\n\n**Options:**\n{options_text}"

                            # Use 'answer' as solution if 'solution' is missing
                            if not solution:
                                answer_text = item["answer"]
                                # Find which option matches the answer
                                option_letter = None
                                if isinstance(options, dict):
                                    for opt_key, opt_value in options.items():
                                        if opt_value == answer_text:
                                            option_letter = opt_key.upper()
                                            break

                                # Format solution with both option letter and answer text
                                if option_letter:
                                    solution = (
                                        f"**Answer: {option_letter})**\n\n{answer_text}"
                                    )
                                else:
                                    solution = answer_text

                        q_id = get_question_id(q_text)

                        # Skip if already processed (deduplication)
                        if q_id in all_questions_map:
                            continue

                        item["source_file"] = file_path
                        item["question_id"] = q_id

                        # Update item with formatted question and solution
                        item["question"] = q_text
                        item["solution"] = solution

                        # Add to database if not exists
                        db.add_question(
                            q_id,
                            q_text,
                            solution,
                            file_path,
                            metadata={
                                "subject": item.get("subject"),
                                "topic": item.get("topic"),
                                "options": item.get("options"),
                            },
                        )
                        all_questions_map[q_id] = item
            except Exception as e:
                warning_msg = f"Warning: Could not read {file_path}: {e}"
                console.print(f"[yellow]{warning_msg}[/yellow]")
                logger.warning(warning_msg)

    all_questions = list(all_questions_map.values())
    console.print(
        f"Found {len(all_questions)} unique questions in {len(json_files)} files."
    )
    logger.info(
        f"Loaded {len(all_questions)} unique questions from {len(json_files)} files"
    )

    # Get or create a run in the database (resume if incomplete run exists)
    run_id, is_resuming = db.get_or_create_run(student_model_name)
    if is_resuming:
        console.print(
            f"[bold yellow]Resuming incomplete run ID: {run_id}[/bold yellow]"
        )
        logger.info(f"Resuming incomplete run ID: {run_id}")
    else:
        console.print(
            f"[bold green]Created new benchmarking run ID: {run_id}[/bold green]"
        )
        logger.info(f"Created new benchmarking run ID: {run_id}")

    # Create output directory inside benchmarking folder
    benchmarking_dir = "benchmarking"
    os.makedirs(benchmarking_dir, exist_ok=True)
    timestamp_dir = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(
        benchmarking_dir, f"{student_model_name.replace('/', '_')}_{timestamp_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Type 1: Cross-run cache - Fully completed evaluations from previous runs
    # We also include the current run_id to skip questions already fully evaluated in this run
    cross_run_cache = db.get_cached_results(
        student_model_name, selected_evaluator_names, include_run_id=run_id
    )
    initial_gen_cost = 0.0
    initial_eval_cost = 0.0
    if cross_run_cache:
        console.print(
            f"[bold green]Found {len(cross_run_cache)} fully evaluated results (cached).[/bold green]"
        )
        logger.info(f"Found {len(cross_run_cache)} cached results")
        for item in cross_run_cache.values():
            initial_gen_cost += item.get("gen_cost", 0.0)
            initial_eval_cost += item.get("eval_cost", 0.0)
    else:
        console.print(
            f"[dim]No cached results found for student model '{student_model_name}'.[/dim]"
        )
        logger.info(f"No cached results found for student model '{student_model_name}'")

    # Type 2: Same-run resume - Student answers generated in THIS run but not yet evaluated
    unevaluated = db.get_unevaluated_answers(run_id)
    unevaluated_map = {a["question_id"]: a for a in unevaluated}
    if unevaluated:
        console.print(
            f"[bold yellow]Found {len(unevaluated)} generated answers pending evaluation in current run (resuming).[/bold yellow]"
        )
        logger.info(
            f"Found {len(unevaluated)} unevaluated answers pending in current run"
        )
        # Add generation costs from unevaluated answers to initial cost
        for answer in unevaluated:
            initial_gen_cost += answer.get("cost", 0.0)

    # Run parallel generation and evaluation
    results = []
    # Use a single filename for the whole run as requested: yyyymmddhhmmss.json
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = timestamp_str + ".json"
    output_path = os.path.join(output_dir, output_filename)

    lock = threading.Lock()
    answer_queue = queue.Queue(maxsize=student_workers * 2)  # Limit queue size
    evaluator_executor = ThreadPoolExecutor(max_workers=evaluator_workers)
    total_gen_cost = ThreadSafeCounter(initial_gen_cost)
    total_eval_cost = ThreadSafeCounter(initial_eval_cost)

    # Calculate total work including unevaluated answers that need evaluation
    questions_in_dataset = set(q["question_id"] for q in all_questions)
    unevaluated_not_in_dataset = set(unevaluated_map.keys()) - questions_in_dataset
    total_work_items = len(all_questions) + len(unevaluated_not_in_dataset)

    # Two progress bars: one for generation, one for evaluation
    gen_pbar = ThreadSafeProgressBar(
        total_work_items, desc="Generating answers", position=0
    )
    eval_pbar = ThreadSafeProgressBar(
        total_work_items, desc="Evaluating answers", position=1
    )

    generation_complete = threading.Event()

    # Producer: Generate answers
    def answer_producer():
        try:
            processed_q_ids = set()

            # Type 1: Cross-run cache (including already evaluated in current run)
            # These skip both generation AND evaluation
            for question_data in all_questions:
                q_id = question_data["question_id"]
                if q_id in cross_run_cache:
                    answer_queue.put(
                        {
                            "question_data": question_data,
                            "cross_run_cached": True,  # Flag: skip both generation and evaluation
                            "cached_result": cross_run_cache[q_id],
                        }
                    )
                    processed_q_ids.add(q_id)
                    gen_pbar.update(1)
                    gen_pbar.refresh()

            # Type 2: Resume - Add unevaluated answers from current run to queue
            # These need evaluation but generation is already done
            for q_id, answer_info in unevaluated_map.items():
                if q_id in processed_q_ids:
                    continue

                # Find the question_data from all_questions
                question_data = next(
                    (q for q in all_questions if q["question_id"] == q_id), None
                )

                # Create usage record for the already-generated answer (cost already in DB)
                gen_cost = answer_info.get("cost", 0.0)
                usage_records = (
                    [
                        {
                            "model": student_model_name,
                            "role": "student",
                            "cost": gen_cost,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "already_in_db": True,
                        }
                    ]
                    if gen_cost > 0
                    else []
                )

                if question_data:
                    # Question found in current dataset
                    answer_queue.put(
                        {
                            "question_data": question_data,
                            "student_answer": answer_info["student_answer"],
                            "usage_records": usage_records,
                            "error": None,
                            "resume": True,
                        }
                    )
                    processed_q_ids.add(q_id)
                    gen_pbar.update(1)
                    gen_pbar.refresh()
                else:
                    # Question not in current dataset but has unevaluated answer in DB
                    session = db.get_session()
                    try:
                        from database import Question

                        db_question = session.query(Question).filter_by(id=q_id).first()
                        if db_question:
                            question_data = {
                                "question_id": q_id,
                                "question": db_question.question_text,
                                "solution": db_question.solution,
                                "source_file": db_question.source_file,
                            }
                            answer_queue.put(
                                {
                                    "question_data": question_data,
                                    "student_answer": answer_info["student_answer"],
                                    "usage_records": usage_records,
                                    "error": None,
                                    "resume": True,
                                }
                            )
                            processed_q_ids.add(q_id)
                            gen_pbar.update(1)
                            gen_pbar.refresh()
                        else:
                            console.print(
                                f"[yellow]Warning: Question {q_id} not found in database, skipping unevaluated answer[/yellow]"
                            )
                    finally:
                        session.close()

            # New questions: Generate fresh answers
            with ThreadPoolExecutor(max_workers=student_workers) as gen_executor:
                futures = []
                for question_data in all_questions:
                    q_id = question_data["question_id"]
                    if q_id not in processed_q_ids:
                        # Submit for generation
                        future = gen_executor.submit(
                            generate_answer,
                            question_data,
                            student_model,
                            student_prompt_template,
                        )
                        futures.append(future)

                # Collect results and put in queue
                for future in as_completed(futures):
                    answer_data = future.result()

                    # Save to DB immediately for resume capability
                    if not answer_data.get("error"):
                        q_id = answer_data["question_data"]["question_id"]
                        student_answer = answer_data["student_answer"]
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cost = sum(
                            u.get("cost", 0.0) for u in answer_data["usage_records"]
                        )

                        # Protect database writes with lock for thread-safety
                        with lock:
                            db.save_student_answer(
                                run_id, q_id, student_answer, cost, timestamp
                            )
                            total_gen_cost.add(cost)
                            logger.debug(
                                f"Saved answer to DB - RunID: {run_id}, QuestionID: {q_id}"
                            )

                    answer_queue.put(answer_data)
                    gen_pbar.update(1)
                    gen_pbar.set_postfix(gen_cost=f"${total_gen_cost.get():.4f}")
                    gen_pbar.refresh()
        except Exception as e:
            logger.error(f"Error in answer_producer: {e}", exc_info=True)
            console.print(f"[bold red]Producer Error: {e}[/bold red]")
        finally:
            generation_complete.set()
            console.print("[green]Answer generation complete![/green]")

    def process_from_queue():
        while not (generation_complete.is_set() and answer_queue.empty()):
            try:
                answer_data = answer_queue.get(timeout=1)
            except queue.Empty:
                continue

            question_data = answer_data["question_data"]
            q_id = question_data["question_id"]

            # Type 1: Cross-run cached - Already fully evaluated in previous run
            if answer_data.get("cross_run_cached"):
                result = answer_data["cached_result"]
                result["question"] = question_data.get("question")
                result["original_answer"] = question_data.get("solution")
                result["source_file"] = question_data.get("source_file")
                usage_records = []
                logger.debug(
                    f"Using cached result - RunID: {run_id}, QuestionID: {q_id}"
                )

                # Ensure student answer is in student_answers table for consistency
                with lock:
                    db.save_student_answer(
                        run_id,
                        q_id,
                        result.get("student_answer"),
                        result.get("gen_cost", 0.0),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    db.mark_answer_evaluated(run_id, q_id)

                # No evaluation needed - just use cached result
            else:
                # Type 2: Resume OR New question - Need evaluation
                logger.debug(
                    f"Evaluating question - RunID: {run_id}, QuestionID: {q_id}"
                )
                result, usage_records = evaluate_answer(
                    answer_data,
                    evaluator_models,
                    prompt_template,
                    evaluator_executor,
                    run_id,
                )
                result["question_id"] = q_id
                result["source_file"] = question_data.get("source_file")

            # CRITICAL SECTION: All shared state modifications in ONE lock for 100% thread-safety
            with lock:
                # 1. Add result to shared list
                results.append(result)

                # 2. Track costs and save to DB
                for usage in usage_records:
                    # Skip adding to counter and saving to DB if already accounted for (resumed answers)
                    if usage.get("already_in_db"):
                        continue

                    role = usage.get("role")
                    cost = usage.get("cost", 0.0)
                    if role == "student":
                        total_gen_cost.add(cost)
                    elif role == "evaluator":
                        total_eval_cost.add(cost)

                    usage["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    usage["question_source"] = result.get("source_file", "")

                    logger.debug(
                        f"Adding cost to DB - RunID: {run_id}, QuestionID: {q_id}, Model: {usage.get('model')}, Role: {usage.get('role')}"
                    )
                    db.add_cost(
                        run_id,
                        q_id,
                        usage.get("model"),
                        usage.get("role"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("total_tokens"),
                        usage.get("cost"),
                        usage.get("timestamp"),
                    )

                # 3. Extract scores and save evaluations to DB
                num_successful_evals = 0
                if "evaluations" in result and isinstance(result["evaluations"], dict):
                    evals_dict = cast(dict[str, Any], result["evaluations"])
                    for eval_model_name, eval_info in evals_dict.items():
                        if isinstance(eval_info, dict):
                            # Skip saving if it was already in DB
                            if eval_info.get("already_in_db"):
                                num_successful_evals += 1
                                continue

                            eval_data = eval_info.get("result")
                            eval_cost = eval_info.get("cost", 0.0)
                            if isinstance(eval_data, dict):
                                db.add_evaluation(
                                    run_id,
                                    q_id,
                                    eval_model_name,
                                    result.get("student_answer"),
                                    eval_data.get("Correctness"),
                                    eval_data.get("Completeness"),
                                    eval_data.get("Clarity"),
                                    eval_data.get("Overall_Score"),
                                    eval_cost,
                                    eval_data,
                                )
                                num_successful_evals += 1

                # 4. Mark answer as evaluated only if all evaluators succeeded
                if not answer_data.get("cross_run_cached"):
                    if num_successful_evals >= len(evaluator_models):
                        db.mark_answer_evaluated(run_id, q_id)
                    else:
                        logger.warning(
                            f"Question {q_id[:8]} not marked as evaluated: only {num_successful_evals}/{len(evaluator_models)} evaluators succeeded"
                        )

                # 5. Update progress bar
                eval_pbar.update(1)
                eval_pbar.set_postfix(eval_cost=f"${total_eval_cost.get():.4f}")

                # Create a copy for saving outside the lock to avoid blocking other threads
                results_snapshot = results.copy()

            # 6. Save intermediate results to JSON (OUTSIDE LOCK)
            with open(output_path, "w") as f:
                json.dump(results_snapshot, f, indent=4)

            answer_queue.task_done()

    # Start producer thread
    producer_thread = threading.Thread(target=answer_producer, daemon=True)
    producer_thread.start()

    # Start consumer threads
    consumer_threads = []
    for _ in range(min(evaluator_workers, len(all_questions))):
        t = threading.Thread(target=process_from_queue, daemon=True)
        t.start()
        consumer_threads.append(t)

    try:
        # Wait for completion
        producer_thread.join()
        for t in consumer_threads:
            t.join()
    except KeyboardInterrupt:
        console.print(
            "[bold yellow]Interrupted by user. Shutting down...[/bold yellow]"
        )
        logger.warning("Run interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}", exc_info=True)
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
    finally:
        gen_pbar.close()
        eval_pbar.close()
        evaluator_executor.shutdown(wait=True)
        db.close_session()

    # Mark run as complete if all questions were processed
    if len(results) >= total_work_items:
        db.mark_run_complete(run_id)
        console.print(f"[bold green]Run {run_id} marked as complete.[/bold green]")
    else:
        console.print(
            f"[bold yellow]Run {run_id} finished with {len(results)}/{total_work_items} items. Run remains incomplete for resuming.[/bold yellow]"
        )

    console.print(
        f"[bold green]Benchmarking complete! Results saved to {output_dir}[/bold green]"
    )
    console.print(f"JSON: {output_path}")
    logger.info(f"Benchmarking complete! Results saved to: {output_path}")
    logger.info(f"Processed {len(results)} total questions")

    # Display Summary
    console.print("\n[bold blue]Run Summary:[/bold blue]")
    summary = db.get_run_summary(run_id)
    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Evaluator",
            "Total Questions",
            "Avg Correctness",
            "Avg Completeness",
            "Avg Clarity",
            "Avg Overall Score",
            "Total Evaluator Cost",
        ],
    )
    console.print(summary_df)
    log_run_summary(
        logger, run_id, student_model_name, selected_evaluator_names, summary_df
    )

    console.print("\n[bold blue]Cost Summary:[/bold blue]")
    costs = db.get_total_costs(run_id)
    costs_df = pd.DataFrame(
        costs,
        columns=["Model", "Role", "Prompt Tokens", "Completion Tokens", "Total Cost"],
    )
    console.print(costs_df)
    log_cost_summary(
        logger, total_gen_cost.get(), total_eval_cost.get(), len(all_questions)
    )
    logger.info("Run completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
