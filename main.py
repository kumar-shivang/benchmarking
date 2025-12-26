import os
import json
import pandas as pd
import click
import threading
import hashlib
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rich.console import Console
from llm import LLM, AuthenticationError
from database import Database

console = Console()
db = Database()


def get_question_id(question_text):
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()


def get_all_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def generate_answer(question_data, student_model, student_prompt_template):
    """Generate student answer (producer function)"""
    question = question_data.get("question")
    usage_records = []

    # Get student answer
    try:
        student_answer, student_usage = student_model.generate_response(
            [
                {"role": "system", "content": student_prompt_template},
                {"role": "user", "content": question},
            ]
        )
        if student_usage:
            student_usage["model"] = student_model.model_name
            student_usage["role"] = "student"
            usage_records.append(student_usage)

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
        return {"question_data": question_data, "error": error_msg, "usage_records": []}
    except Exception as e:
        error_msg = f"Student model error: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        return {"question_data": question_data, "error": error_msg, "usage_records": []}


def evaluate_answer(
    answer_data,
    evaluator_models,
    prompt_template,
    evaluator_executor,
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

    # Prepare evaluation prompt
    try:
        prompt = prompt_template.format(
            question=question, answer=student_answer, original_answer=original_answer
        )
    except KeyError as e:
        error_msg = f"Prompt formatting error: Missing key {e} in prompt template. Check for unescaped curly braces in evaluation_prompt.md"
        console.print(f"[red]{error_msg}[/red]")
        return {"error": error_msg}, usage_records

    # Run evaluators in parallel
    evaluations = {}

    def run_evaluator(eval_model):
        try:
            response, eval_usage = eval_model.generate_response(
                [
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": prompt},
                ]
            )
            if eval_usage:
                eval_usage["model"] = eval_model.model_name
                eval_usage["role"] = "evaluator"

            # Try to parse JSON if possible, otherwise keep as string
            try:
                parsed_response = json.loads(response)
                return parsed_response, eval_usage
            except json.JSONDecodeError:
                return response, eval_usage
        except Exception as e:
            return f"Error: {str(e)}", {}

    future_to_model = {
        evaluator_executor.submit(run_evaluator, m): m.model_name
        for m in evaluator_models
    }
    for future in as_completed(future_to_model):
        model_name = future_to_model[future]
        res, usage = future.result()
        evaluations[model_name] = {"result": res, "cost": usage.get("cost", 0.0)}
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
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print(
            "[bold red]Error: OPENROUTER_API_KEY not found in environment or .env file.[/bold red]"
        )
        return

    # Load models
    models_df = pd.read_csv(models_csv)
    evaluators_df = pd.read_csv(evaluators_csv)

    # Select Student Model
    console.print("[bold blue]Select Student Model:[/bold blue]")
    for i, model in enumerate(models_df["model"]):
        console.print(f"{i + 1}. {model}")

    model_idx = click.prompt("Enter model number", type=int) - 1
    student_model_name = models_df.iloc[model_idx]["model"]
    try:
        student_model = LLM(student_model_name)
    except AuthenticationError as e:
        console.print(f"[bold red]Authentication Error:[/bold red] {e}")
        console.print(
            "[yellow]Please check if your OPENROUTER_API_KEY is valid and has sufficient credits.[/yellow]"
        )
        return
    except Exception as e:
        console.print(f"[bold red]Error initializing student model:[/bold red] {e}")
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
        console.print(
            f"[red]Error: You must select exactly {num_evaluators} evaluators.[/red]"
        )
        return

    try:
        selected_evaluator_names = [
            evaluators_df.iloc[int(idx) - 1]["model"] for idx in evaluator_indices
        ]
    except (ValueError, IndexError):
        console.print("[red]Error: Invalid evaluator selection.[/red]")
        return

    try:
        evaluator_models = [LLM(name) for name in selected_evaluator_names]
    except AuthenticationError as e:
        console.print(
            f"[bold red]Authentication Error during evaluator initialization:[/bold red] {e}"
        )
        return
    except Exception as e:
        console.print(f"[bold red]Error initializing evaluator models:[/bold red] {e}")
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
        else:
            selected_data_dir = data_dir
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
            selected_data_dir = os.path.join(data_dir, matches[0])
        else:
            console.print(
                f"[red]Error: No match found for class/level '{class_level}'.[/red]"
            )
            return

    # Parallel Configuration
    student_workers = click.prompt(
        "How many parallel student instances do you want to run?", type=int, default=1
    )
    evaluator_workers = click.prompt(
        "How many parallel evaluator threads do you want to run?", type=int, default=5
    )

    # Load prompt templates
    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    with open(student_prompt_file, "r") as f:
        student_prompt_template = f.read()

    # Get all questions and populate database
    json_files = get_all_json_files(selected_data_dir)
    all_questions = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        q_text = item.get("question", "")
                        q_id = get_question_id(q_text)
                        item["source_file"] = file_path
                        item["question_id"] = q_id

                        # Add to database if not exists
                        db.add_question(
                            q_id,
                            q_text,
                            item.get("solution"),
                            file_path,
                            metadata={
                                "subject": item.get("subject"),
                                "topic": item.get("topic"),
                            },
                        )
                        all_questions.append(item)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]"
                )

    console.print(f"Found {len(all_questions)} questions in {len(json_files)} files.")

    # Create a new run in the database
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = db.create_run(student_model_name, run_timestamp)
    console.print(f"[bold green]Created benchmarking run ID: {run_id}[/bold green]")

    # Create output directory inside benchmarking folder
    benchmarking_dir = "benchmarking"
    os.makedirs(benchmarking_dir, exist_ok=True)
    timestamp_dir = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(
        benchmarking_dir, f"{student_model_name.replace('/', '_')}_{timestamp_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Type 1: Cross-run cache - Fully completed evaluations from previous runs
    cross_run_cache = db.get_cached_results(student_model_name)
    initial_gen_cost = 0.0
    initial_eval_cost = 0.0
    if cross_run_cache:
        console.print(
            f"[bold green]Found {len(cross_run_cache)} fully evaluated results from previous runs (cross-run cache).[/bold green]"
        )
        for item in cross_run_cache.values():
            initial_gen_cost += item.get("gen_cost", 0.0)
            initial_eval_cost += item.get("eval_cost", 0.0)
    else:
        console.print(
            f"[dim]No cross-run cache found for student model '{student_model_name}'.[/dim]"
        )

    # Type 2: Same-run resume - Student answers generated in THIS run but not yet evaluated
    unevaluated = db.get_unevaluated_answers(run_id)
    unevaluated_map = {a["question_id"]: a for a in unevaluated}
    if unevaluated:
        console.print(
            f"[bold yellow]Found {len(unevaluated)} generated answers pending evaluation in current run (resuming).[/bold yellow]"
        )

    # Run parallel generation and evaluation
    results = []
    # Use a single filename for the whole run as requested: yyyymmddhhmmss.json
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = timestamp_str + ".json"
    output_path = os.path.join(output_dir, output_filename)

    lock = threading.Lock()
    answer_queue = queue.Queue(maxsize=student_workers * 2)  # Limit queue size
    evaluator_executor = ThreadPoolExecutor(max_workers=evaluator_workers)
    total_gen_cost = initial_gen_cost
    total_eval_cost = initial_eval_cost

    # Two progress bars: one for generation, one for evaluation
    gen_pbar = tqdm(total=len(all_questions), desc="Generating answers", position=0)
    eval_pbar = tqdm(total=len(all_questions), desc="Evaluating answers", position=1)

    generation_complete = threading.Event()

    # Producer: Generate answers
    def answer_producer():
        # Type 2: Resume - Add unevaluated answers from current run to queue
        # These need evaluation but generation is already done
        for q_id, answer_info in unevaluated_map.items():
            # Find the question_data
            question_data = next(
                (q for q in all_questions if q["question_id"] == q_id), None
            )
            if question_data:
                answer_queue.put(
                    {
                        "question_data": question_data,
                        "student_answer": answer_info["student_answer"],
                        "usage_records": [],  # Already saved in DB
                        "error": None,
                        "resume": True,  # Flag: skip generation, needs evaluation
                    }
                )
                with lock:
                    gen_pbar.update(1)  # Generation was already done

        # Type 1: Cross-run cache - Add fully evaluated results from previous runs
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
                with lock:
                    gen_pbar.update(1)  # Generation was already done in previous run
                    eval_pbar.update(1)  # Evaluation was already done in previous run

        # New questions: Generate fresh answers
        with ThreadPoolExecutor(max_workers=student_workers) as gen_executor:
            futures = []
            for question_data in all_questions:
                q_id = question_data["question_id"]
                # Only generate if not in cross-run cache and not already generated in current run
                if q_id not in cross_run_cache and q_id not in unevaluated_map:
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
                    cost = sum(u.get("cost", 0.0) for u in answer_data["usage_records"])
                    
                    # Protect database writes with lock for thread-safety
                    with lock:
                        db.save_student_answer(
                            run_id, q_id, student_answer, cost, timestamp
                        )
                        total_gen_cost += cost

                answer_queue.put(answer_data)
                with lock:
                    gen_pbar.update(1)
                    gen_pbar.set_postfix(gen_cost=f"${total_gen_cost:.4f}")

        generation_complete.set()
        console.print("[green]Answer generation complete![/green]")

    # Consumer: Evaluate answers
    def process_from_queue():
        nonlocal total_gen_cost, total_eval_cost

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
                # No evaluation needed - just use cached result
            else:
                # Type 2: Resume OR New question - Need evaluation
                result, usage_records = evaluate_answer(
                    answer_data,
                    evaluator_models,
                    prompt_template,
                    evaluator_executor,
                )
                result["question_id"] = q_id
                result["source_file"] = question_data.get("source_file")

            # CRITICAL SECTION: All shared state modifications in ONE lock for 100% thread-safety
            with lock:
                # 1. Add result to shared list
                results.append(result)

                # 2. Track costs and save to DB
                for usage in usage_records:
                    role = usage.get("role")
                    cost = usage.get("cost", 0.0)
                    if role == "student":
                        total_gen_cost += cost
                    elif role == "evaluator":
                        total_eval_cost += cost

                    usage["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    usage["question_source"] = result.get("source_file", "")

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
                if "evaluations" in result and isinstance(result["evaluations"], dict):
                    for eval_model_name, eval_info in result["evaluations"].items():
                        if isinstance(eval_info, dict):
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

                # 4. Mark answer as evaluated
                if not answer_data.get("cross_run_cached"):
                    db.mark_answer_evaluated(run_id, q_id)

                # 5. Update progress bar
                eval_pbar.update(1)
                eval_pbar.set_postfix(eval_cost=f"${total_eval_cost:.4f}")

                # 6. Save intermediate results to JSON
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)

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

    # Wait for completion
    producer_thread.join()
    for t in consumer_threads:
        t.join()

    gen_pbar.close()
    eval_pbar.close()
    evaluator_executor.shutdown(wait=True)

    console.print(
        f"[bold green]Benchmarking complete! Results saved to {output_dir}[/bold green]"
    )
    console.print(f"JSON: {output_path}")

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

    console.print("\n[bold blue]Cost Summary:[/bold blue]")
    costs = db.get_total_costs(run_id)
    costs_df = pd.DataFrame(
        costs,
        columns=["Model", "Role", "Prompt Tokens", "Completion Tokens", "Total Cost"],
    )
    console.print(costs_df)


if __name__ == "__main__":
    main()
