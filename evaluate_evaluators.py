import os
import json
import random
import pandas as pd
from rich.console import Console
from rich.table import Table
from llm import LLM
from database import Database, Question
from schemas import EVALUATION_SCHEMA
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

console = Console()


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


def get_random_questions(db, n=5):
    session = db.get_session()
    try:
        # Get all question IDs first to pick random ones efficiently
        # or just fetch all if the dataset is small enough.
        # Assuming dataset might be large, let's count first.
        count = session.query(Question).count()
        if count < n:
            console.print(
                f"[yellow]Warning: Only {count} questions in DB, requested {n}.[/yellow]"
            )
            questions = session.query(Question).all()
        else:
            # Fetch all and sample (if not too huge) or use random offset
            # For simplicity, let's fetch all IDs and sample
            all_ids = [q.id for q in session.query(Question.id).all()]
            selected_ids = random.sample(all_ids, n)
            questions = (
                session.query(Question).filter(Question.id.in_(selected_ids)).all()
            )
        return questions
    finally:
        session.close()


def generate_and_save_answer(args):
    (
        question,
        model_name,
        prompt_template,
        is_good,
        output_dir,
    ) = args

    answer_type = "good" if is_good else "bad"
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    output_filename = os.path.join(
        output_dir, f"q{question.id}_{answer_type}_{safe_model_name}.json"
    )

    if os.path.exists(output_filename):
        # console.print(f"Skipping existing answer: {os.path.basename(output_filename)}")
        return

    try:
        llm = LLM(model_name)
        system_prompt = prompt_template
        user_prompt = question.question_text

        console.print(
            f"Generating {answer_type} answer for Q{question.id} using {model_name}..."
        )

        response, _ = llm.generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # response_format=STUDENT_ANSWER_SCHEMA,
        )

        # try:
        #     parsed_response = json.loads(clean_json_response(response))
        # except Exception:
        #     parsed_response = response
        parsed_response = response

        answer_data = {
            "question_id": question.id,
            "question_text": question.question_text,
            "solution": question.solution,
            "student_answer": parsed_response,
            "model": model_name,
            "type": answer_type,
        }

        with open(output_filename, "w") as f:
            json.dump(answer_data, f, indent=2)

    except Exception as e:
        console.print(
            f"[red]Error generating answer with {model_name} for Q{question.id}: {e}[/red]"
        )


def generate_all_answers(questions, student_models, good_prompt, bad_prompt):
    output_dir = "temp/generated_answers"
    os.makedirs(output_dir, exist_ok=True)

    good_student_models, bad_student_models = student_models

    tasks = []
    for q in questions:
        # Assign good models
        if good_student_models:
            model_name = random.choice(good_student_models)
            tasks.append((q, model_name, good_prompt, True, output_dir))
        # Assign bad models
        if bad_student_models:
            model_name = random.choice(bad_student_models)
            tasks.append((q, model_name, bad_prompt, False, output_dir))

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(generate_and_save_answer, tasks))


def evaluate_and_save_evaluation(args):
    answer_file, eval_model_name, prompt_template, output_dir = args

    base_name = os.path.basename(answer_file).replace(".json", "")
    safe_eval_model_name = eval_model_name.replace("/", "_")
    output_filename = os.path.join(
        output_dir, f"eval_{base_name}_by_{safe_eval_model_name}.json"
    )

    if os.path.exists(output_filename):
        # console.print(f"Skipping existing evaluation: {os.path.basename(output_filename)}")
        return

    try:
        with open(answer_file, "r") as f:
            ans = json.load(f)

        eval_llm = LLM(eval_model_name)

        # Ensure student_answer is a dict for formatting
        student_answer = ans["student_answer"]
        # if isinstance(student_answer, str):
        #     try:
        #         student_answer = json.loads(student_answer)
        #     except json.JSONDecodeError:
        #         # Fallback if it's just a plain string answer
        #         student_answer = {
        #             "step_by_step_reasoning": "N/A",
        #             "final_answer": student_answer,
        #         }

        # Sanitize student_answer to ensure no booleans
        if isinstance(student_answer, dict):
            for k, v in student_answer.items():
                if isinstance(v, bool):
                    student_answer[k] = str(v)

        prompt = prompt_template.format(
            question=ans["question_text"],
            answer=student_answer,
            original_answer=ans["solution"],
        )

        console.print(
            f"Evaluating {os.path.basename(answer_file)} with {eval_model_name}..."
        )
        response, _ = eval_llm.generate_response(
            [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ],
            response_format=EVALUATION_SCHEMA,
        )

        cleaned_response = clean_json_response(response)
        eval_json = json.loads(cleaned_response)

        eval_data = {
            "evaluator_model": eval_model_name,
            "original_answer_type": ans["type"],
            "evaluation": eval_json,
        }

        with open(output_filename, "w") as f:
            json.dump(eval_data, f, indent=2)

    except json.JSONDecodeError:
        console.print(f"[yellow]Failed to parse JSON from {eval_model_name}[/yellow]")
        error_filename = output_filename.replace(".json", "_error.txt")
        with open(error_filename, "w") as f:
            f.write(response)
        console.print(
            f"[yellow]Saved raw response to {os.path.basename(error_filename)}[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error evaluating with {eval_model_name}: {e}[/red]")


def evaluate_all_answers(evaluator_models, prompt_template):
    answers_dir = "temp/generated_answers"
    output_dir = "temp/evaluations"
    os.makedirs(output_dir, exist_ok=True)

    answer_files = [
        os.path.join(answers_dir, f)
        for f in os.listdir(answers_dir)
        if f.endswith(".json")
    ]

    tasks = []
    for answer_file in answer_files:
        for eval_model in evaluator_models:
            tasks.append((answer_file, eval_model, prompt_template, output_dir))

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(evaluate_and_save_evaluation, tasks))


def present_results():
    evaluations_dir = "temp/evaluations"
    if not os.path.exists(evaluations_dir):
        console.print("[red]No evaluations found. Run with --evaluate first.[/red]")
        return

    results = {}  # evaluator_model -> {good_scores: [], bad_scores: []}

    eval_files = [
        os.path.join(evaluations_dir, f)
        for f in os.listdir(evaluations_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(evaluations_dir, f))
    ]

    for f in eval_files:
        try:
            with open(f, "r") as file:
                data = json.load(file)

            if "evaluator_model" not in data:
                console.print(
                    f"[yellow]Skipping {f}: Missing 'evaluator_model' key[/yellow]"
                )
                continue

            model = data["evaluator_model"]
            if model not in results:
                results[model] = {
                    "good_scores": [],
                    "bad_scores": [],
                    "raw_evaluations": [],
                }

            # Handle case sensitivity for keys
            evaluation_data = {k.lower(): v for k, v in data["evaluation"].items()}
            score = evaluation_data.get("overall_score", 0)
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = 0

            if data["original_answer_type"] == "good":
                results[model]["good_scores"].append(score)
            else:
                results[model]["bad_scores"].append(score)

            results[model]["raw_evaluations"].append(data)
        except Exception as e:
            console.print(f"[yellow]Could not process file {f}: {e}[/yellow]")

    table = Table(title="Evaluator Performance Report")
    table.add_column("Evaluator Model", style="cyan")
    table.add_column("Avg Good Score", style="green")
    table.add_column("Avg Bad Score", style="red")
    table.add_column("Gap (Good - Bad)", style="bold yellow")
    table.add_column("Good Count", style="green")
    table.add_column("Bad Count", style="red")

    report_data = []
    for model, data in results.items():
        good_scores = data["good_scores"]
        bad_scores = data["bad_scores"]

        if not good_scores and not bad_scores:
            continue

        avg_good = sum(good_scores) / len(good_scores) if good_scores else 0
        avg_bad = sum(bad_scores) / len(bad_scores) if bad_scores else 0
        gap = avg_good - avg_bad

        report_data.append(
            {
                "model": model,
                "avg_good": avg_good,
                "avg_bad": avg_bad,
                "gap": gap,
                "good_count": len(good_scores),
                "bad_count": len(bad_scores),
            }
        )

    report_data.sort(key=lambda x: x["gap"], reverse=True)

    for row in report_data:
        table.add_row(
            row["model"],
            f"{row['avg_good']:.2f}",
            f"{row['avg_bad']:.2f}",
            f"{row['gap']:.2f}",
            str(row["good_count"]),
            str(row["bad_count"]),
        )

    console.print(table)

    report_filename = (
        f"evaluator_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Detailed report saved to {report_filename}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate student answers for benchmarking evaluator models."
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate student answers."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate student answers.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Parse evaluations and present results.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to use for generation.",
    )
    args = parser.parse_args()

    if not args.generate and not args.evaluate and not args.report:
        console.print(
            "[yellow]Please specify a stage: --generate, --evaluate, --report, or a combination.[/yellow]"
        )
        return

    # Common setup
    db = Database()
    try:
        models_df = pd.read_csv("models.csv")
        student_models_pool = models_df["model"].tolist()
    except Exception as e:
        console.print(f"[red]Error reading models.csv: {e}[/red]")
        return

    if args.generate:
        console.print("[bold cyan]--- GENERATION STAGE ---[/bold cyan]")
        questions = get_random_questions(db, args.num_questions)
        if not questions:
            console.print("[red]No questions found in database.[/red]")
            return

        console.print(f"[green]Selected {len(questions)} questions.[/green]")

        random.shuffle(student_models_pool)
        # Use random models for good answers
        good_student_models = student_models_pool

        # Use specific models for bad answers as requested
        bad_student_models = ["deepseek/deepseek-v3.2-speciale"]

        with open("student_prompt.md", "r") as f:
            good_prompt = f.read()

        bad_prompt = (
            "You are a student attempting to solve this math problem, but you are struggling and prone to making common errors. "
            "Please generate a solution that contains mistakes typical of a student at this level. "
            "Do NOT simply write nonsense; the errors should be plausible.\n\n"
            "Include one or more of the following types of errors:\n"
            "- **Conceptual Errors:** Misunderstanding a core concept, using a formula incorrectly, or applying a rule where it doesn't belong.\n"
            "- **Calculation Errors:** Making simple arithmetic mistakes (e.g., adding instead of multiplying, sign errors, fraction mistakes).\n"
            "- **Misinterpretation:** Misreading the question or solving for the wrong variable.\n"
            "- **Logic Gaps:** Skipping necessary steps or making unjustified leaps in reasoning.\n\n"
            "Guidelines:\n"
            "- **CRITICAL:** Your answer must contain the ACTUAL DERIVATION/WORK of the student. "
            "Do NOT explain *what* the student is doing (e.g., do NOT write 'The student incorrectly adds...'). "
            "Instead, WRITE THE MATH AND TEXT EXACTLY AS THE STUDENT WOULD WRITE IT ON AN EXAM.\n"
            "- The final answer should be incorrect due to these errors.\n"
            "- Do NOT explicitly state 'I am making a mistake' or 'This is wrong'. Act completely confident in your incorrect answer."
        )

        generate_all_answers(
            questions,
            (good_student_models, bad_student_models),
            good_prompt,
            bad_prompt,
        )
        console.print("[bold green]Generation complete.[/bold green]")

    if args.evaluate:
        console.print("[bold cyan]--- EVALUATION STAGE ---[/bold cyan]")
        try:
            eval_df = pd.read_csv("evalulator.csv")
            evaluator_models = eval_df["model"].tolist()
        except Exception as e:
            console.print(f"[red]Error reading evalulator.csv: {e}[/red]")
            return

        with open("evaluation_prompt.md", "r") as f:
            eval_prompt_template = f.read()

        console.print(
            f"[bold]Evaluating with {len(evaluator_models)} evaluators...[/bold]"
        )
        evaluate_all_answers(evaluator_models, eval_prompt_template)
        console.print("[bold green]Evaluation complete.[/bold green]")

    if args.report:
        console.print("[bold cyan]--- REPORTING STAGE ---[/bold cyan]")
        present_results()


if __name__ == "__main__":
    main()
