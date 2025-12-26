import os
import json
import pandas as pd
import click
import threading
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rich.console import Console
from llm import LLM, AuthenticationError

console = Console()


def get_question_id(question_text):
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()


def get_all_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def evaluate_question(
    question_data,
    student_model,
    evaluator_models,
    prompt_template,
    evaluator_executor,
    student_prompt_template,
):
    question = question_data.get("question")
    original_answer = question_data.get("solution")
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
    except AuthenticationError as e:
        error_msg = f"Authentication Error: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        console.print(
            "[yellow]Hint: This usually means your API key is invalid or you have run out of credits on OpenRouter.[/yellow]"
        )
        return {"error": error_msg}, []
    except Exception as e:
        error_msg = f"Student model error: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        return {"error": error_msg}, []

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
        evaluations[model_name] = res
        if usage:
            usage_records.append(usage)

    return {
        "question": question,
        "student_answer": student_answer,
        "original_answer": original_answer,
        "evaluations": evaluations,
    }, usage_records


def load_existing_results(output_dir):
    cache = {}
    if not os.path.exists(output_dir):
        return cache

    for file in os.listdir(output_dir):
        if file.endswith(".json"):
            file_path = os.path.join(output_dir, file)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            q_id = item.get("question_id")
                            if q_id:
                                cache[q_id] = item
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load cache from {file_path}: {e}[/yellow]"
                )
    return cache


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
def main(models_csv, evaluators_csv, data_dir, prompt_file, student_prompt_file):
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

    # Get all questions
    json_files = get_all_json_files(data_dir)
    all_questions = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        item["source_file"] = file_path
                        item["question_id"] = get_question_id(item.get("question", ""))
                        all_questions.append(item)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]"
                )

    console.print(f"Found {len(all_questions)} questions in {len(json_files)} files.")

    # Create output directory inside benchmarking folder
    benchmarking_dir = "benchmarking"
    os.makedirs(benchmarking_dir, exist_ok=True)
    timestamp_dir = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(
        benchmarking_dir, f"{student_model_name.replace('/', '_')}_{timestamp_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load existing results for resume
    cache = load_existing_results(output_dir)
    if cache:
        console.print(
            f"[bold green]Found {len(cache)} existing results. Resuming...[/bold green]"
        )

    # Run evaluations
    results = []
    score_records = []
    cost_records = []
    # Use a single filename for the whole run as requested: yyyymmddhhmmss.json
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = timestamp_str + ".json"
    csv_filename = timestamp_str + ".csv"
    cost_filename = timestamp_str + "_costs.csv"
    output_path = os.path.join(output_dir, output_filename)
    csv_path = os.path.join(output_dir, csv_filename)
    cost_path = os.path.join(output_dir, cost_filename)

    lock = threading.Lock()
    evaluator_executor = ThreadPoolExecutor(max_workers=evaluator_workers)

    def process_question(question_data):
        q_id = question_data["question_id"]

        if q_id in cache:
            result = cache[q_id]
            usage_records = []
        else:
            result, usage_records = evaluate_question(
                question_data,
                student_model,
                evaluator_models,
                prompt_template,
                evaluator_executor,
                student_prompt_template,
            )
            result["question_id"] = q_id
            result["source_file"] = question_data.get("source_file")

        with lock:
            results.append(result)

            # Track costs
            for usage in usage_records:
                usage["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                usage["question_source"] = result.get("source_file", "")
                cost_records.append(usage)

            # Extract scores for CSV
            if "evaluations" in result and isinstance(result["evaluations"], dict):
                for eval_model_name, eval_data in result["evaluations"].items():
                    if isinstance(eval_data, dict):
                        record = {
                            "question": result.get("question", "")[:100] + "...",
                            "source_file": result.get("source_file", ""),
                            "evaluator": eval_model_name,
                            "correctness": eval_data.get("Correctness"),
                            "completeness": eval_data.get("Completeness"),
                            "clarity": eval_data.get("Clarity"),
                            "overall_score": eval_data.get("Overall_Score"),
                        }
                        score_records.append(record)

            # Save intermediate results to JSON
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

            # Save intermediate results to CSV
            if score_records:
                pd.DataFrame(score_records).to_csv(csv_path, index=False)

            # Save intermediate costs to CSV
            if cost_records:
                pd.DataFrame(cost_records).to_csv(cost_path, index=False)

    with ThreadPoolExecutor(max_workers=student_workers) as student_executor:
        list(
            tqdm(
                student_executor.map(process_question, all_questions),
                total=len(all_questions),
                desc="Evaluating questions",
            )
        )

    evaluator_executor.shutdown(wait=True)

    console.print(
        f"[bold green]Benchmarking complete! Results saved to {output_dir}[/bold green]"
    )
    console.print(f"JSON: {output_path}")
    console.print(f"Scores CSV: {csv_path}")
    console.print(f"Costs CSV: {cost_path}")


if __name__ == "__main__":
    main()
