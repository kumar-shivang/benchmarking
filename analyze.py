import pandas as pd
import click
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from database import Database
from dotenv import load_dotenv

load_dotenv(override=True)

console = Console()
db = Database()


@click.command()
@click.option("--export", is_flag=True, help="Export analysis to CSV files")
def analyze_performance(export):
    engine = db.engine

    # 1. Score Analysis
    console.print("\n[bold blue]Model Performance Analysis (Scores)[/bold blue]")

    query_scores = """
    SELECT 
        r.student_model,
        COUNT(DISTINCT e.question_id) as questions_evaluated,
        AVG(e.correctness) as avg_correctness,
        AVG(e.completeness) as avg_completeness,
        AVG(e.clarity) as avg_clarity,
        AVG(e.overall_score) as avg_overall_score
    FROM evaluations e
    JOIN runs r ON e.run_id = r.id
    GROUP BY r.student_model
    ORDER BY avg_overall_score DESC
    """

    df_scores = None
    try:
        df_scores = pd.read_sql(query_scores, engine)

        if df_scores.empty:
            console.print("[yellow]No evaluation data found.[/yellow]")
        else:
            table_scores = Table(title="Average Scores by Model")
            table_scores.add_column("Model", style="cyan")
            table_scores.add_column("Questions", justify="right")
            table_scores.add_column("Correctness", justify="right")
            table_scores.add_column("Completeness", justify="right")
            table_scores.add_column("Clarity", justify="right")
            table_scores.add_column("Overall Score", justify="right", style="green")

            for _, row in df_scores.iterrows():
                table_scores.add_row(
                    row["student_model"],
                    str(row["questions_evaluated"]),
                    f"{row['avg_correctness']:.2f}",
                    f"{row['avg_completeness']:.2f}",
                    f"{row['avg_clarity']:.2f}",
                    f"{row['avg_overall_score']:.2f}",
                )

            console.print(table_scores)
    except Exception as e:
        console.print(f"[red]Error fetching scores: {e}[/red]")

    console.print()

    # 2. Cost Analysis
    console.print("[bold blue]Cost Analysis[/bold blue]")

    query_costs = """
    SELECT 
        r.student_model,
        SUM(CASE WHEN c.role = 'student' THEN c.cost ELSE 0 END) as student_cost,
        SUM(CASE WHEN c.role = 'evaluator' THEN c.cost ELSE 0 END) as evaluator_cost,
        SUM(c.cost) as total_cost,
        SUM(c.total_tokens) as total_tokens,
        COUNT(DISTINCT c.question_id) as questions_processed
    FROM costs c
    JOIN runs r ON c.run_id = r.id
    GROUP BY r.student_model
    ORDER BY total_cost ASC
    """

    df_costs = None
    try:
        df_costs = pd.read_sql(query_costs, engine)

        if df_costs.empty:
            console.print("[yellow]No cost data found.[/yellow]")
        else:
            # Calculate avg cost per question
            df_costs["avg_cost_per_question"] = df_costs.apply(
                lambda row: (
                    row["total_cost"] / row["questions_processed"]
                    if row["questions_processed"] > 0
                    else 0
                ),
                axis=1,
            )

            table_costs = Table(title="Cost & Token Usage by Model")
            table_costs.add_column("Model", style="cyan")
            table_costs.add_column("Student Cost", justify="right")
            table_costs.add_column("Eval Cost", justify="right")
            table_costs.add_column("Total Cost", justify="right", style="green")
            table_costs.add_column("Avg Cost/Q", justify="right")

            for _, row in df_costs.iterrows():
                table_costs.add_row(
                    row["student_model"],
                    f"${row['student_cost']:.4f}",
                    f"${row['evaluator_cost']:.4f}",
                    f"${row['total_cost']:.4f}",
                    f"${row['avg_cost_per_question']:.4f}",
                )

            console.print(table_costs)
    except Exception as e:
        console.print(f"[red]Error fetching costs: {e}[/red]")

    # Export to CSV if requested
    if export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = "analysis_exports"
        os.makedirs(export_dir, exist_ok=True)

        if df_scores is not None and not df_scores.empty:
            scores_path = os.path.join(
                export_dir, f"performance_analysis_{timestamp}.csv"
            )
            df_scores.to_csv(scores_path, index=False)
            console.print(
                f"\n[green]Performance analysis exported to: {scores_path}[/green]"
            )

        if df_costs is not None and not df_costs.empty:
            costs_path = os.path.join(export_dir, f"cost_analysis_{timestamp}.csv")
            df_costs.to_csv(costs_path, index=False)
            console.print(f"[green]Cost analysis exported to: {costs_path}[/green]")


if __name__ == "__main__":
    analyze_performance()
