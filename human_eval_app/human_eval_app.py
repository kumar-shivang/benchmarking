"""
Flask Human Evaluation App

This app allows human evaluators
to evaluate student answers. All data is stored in a local SQLite database.
"""

import os
import csv
import sqlite3
import logging
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.exceptions import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Use local SQLite for human evaluations
# Set absolute path to the database file in the same directory as this script
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "human_eval.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# Models for Human Evaluation Database
class HumanQuestion(db.Model):
    id = db.Column(db.String, primary_key=True)
    question_text = db.Column(db.Text, nullable=False)
    solution = db.Column(db.Text)
    source_file = db.Column(db.String)
    metadata_json = db.Column(db.Text)


class HumanRun(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_model = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.String, nullable=False)


class HumanStudentAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.Integer, db.ForeignKey("human_run.id"))
    question_id = db.Column(db.String, db.ForeignKey("human_question.id"))
    student_answer = db.Column(db.Text)
    cost = db.Column(db.Float)
    timestamp = db.Column(db.String)
    evaluated = db.Column(db.Integer, default=0)


class HumanEvaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_answer_id = db.Column(db.Integer, db.ForeignKey("human_student_answer.id"))
    run_id = db.Column(db.Integer, db.ForeignKey("human_run.id"))
    question_id = db.Column(db.String, db.ForeignKey("human_question.id"))
    evaluator_name = db.Column(db.String)
    student_answer = db.Column(db.Text)

    # Criteria matching LLM schema
    correctness = db.Column(db.Integer)
    correctness_explanation = db.Column(db.Text)

    completeness = db.Column(db.Integer)
    completeness_explanation = db.Column(db.Text)

    clarity = db.Column(db.Integer)
    clarity_explanation = db.Column(db.Text)

    overall_score = db.Column(db.Float)
    overall_explanation = db.Column(db.Text)

    reasoning = db.Column(db.Text)  # General reasoning/remarks

    created_at = db.Column(db.String, default=datetime.now().isoformat())


@app.route("/")
def index():
    """Home page with navigation."""
    return render_template("index.html")


@app.route("/set_evaluator", methods=["POST"])
def set_evaluator():
    """Set the evaluator name in the session."""
    evaluator_name = request.form.get("evaluator_name")
    if evaluator_name:
        session["evaluator_name"] = evaluator_name
        flash(f"Evaluator name set to: {evaluator_name}", "success")
    return redirect(url_for("index"))


@app.route("/evaluate")
def evaluate_list():
    """List all student answers with evaluation status and filters."""
    # Get filter parameters
    source_file = request.args.get("source_file")
    model_name = request.args.get("model_name")
    status = request.args.get("status")  # 'all', 'pending', 'evaluated'

    # Base query
    query = (
        db.session.query(HumanStudentAnswer, HumanQuestion, HumanRun)
        .join(HumanQuestion, HumanStudentAnswer.question_id == HumanQuestion.id)
        .join(HumanRun, HumanStudentAnswer.run_id == HumanRun.id)
    )

    # Apply filters
    if source_file and source_file != "all":
        query = query.filter(HumanQuestion.source_file == source_file)
    if model_name and model_name != "all":
        query = query.filter(HumanRun.student_model == model_name)

    # For status filtering, we check if a HumanEvaluation record exists
    if status == "pending":
        query = query.filter(
            ~db.session.query(HumanEvaluation)
            .filter(HumanEvaluation.student_answer_id == HumanStudentAnswer.id)
            .exists()
        )
    elif status == "evaluated":
        query = query.filter(
            db.session.query(HumanEvaluation)
            .filter(HumanEvaluation.student_answer_id == HumanStudentAnswer.id)
            .exists()
        )

    results = query.all()

    # Get unique values for filters
    all_sources = db.session.query(HumanQuestion.source_file).distinct().all()
    all_models = db.session.query(HumanRun.student_model).distinct().all()

    sources = [s[0] for s in all_sources if s[0]]
    models = [m[0] for m in all_models if m[0]]

    # Group by question for the UI
    questions = {}
    for answer, question, run in results:
        # Check if this specific answer has been evaluated by a human
        is_evaluated = (
            db.session.query(HumanEvaluation)
            .filter_by(student_answer_id=answer.id)
            .first()
            is not None
        )

        if answer.question_id not in questions:
            questions[answer.question_id] = {
                "text": question.question_text,
                "source": question.source_file,
                "answers": [],
            }
        questions[answer.question_id]["answers"].append(
            {
                "id": answer.id,
                "run_id": answer.run_id,
                "model": run.student_model,
                "timestamp": answer.timestamp,
                "evaluated": is_evaluated,
            }
        )

    return render_template(
        "evaluate_list.html",
        questions=questions,
        sources=sources,
        models=models,
        selected_source=source_file,
        selected_model=model_name,
        selected_status=status,
    )


@app.route("/evaluate/<int:answer_id>", methods=["GET", "POST"])
def evaluate_answer(answer_id):
    """Evaluate a specific student answer."""
    answer = HumanStudentAnswer.query.get_or_404(answer_id)
    question = HumanQuestion.query.get(answer.question_id)
    run = HumanRun.query.get(answer.run_id)

    if request.method == "POST":
        try:
            evaluator_name = request.form.get("evaluator_name") or session.get(
                "evaluator_name", "Anonymous"
            )

            # Also update session if a new name was provided in the form
            if request.form.get("evaluator_name"):
                session["evaluator_name"] = request.form.get("evaluator_name")

            eval_record = HumanEvaluation(
                student_answer_id=answer.id,
                run_id=answer.run_id,
                question_id=answer.question_id,
                evaluator_name=evaluator_name,
                student_answer=answer.student_answer,
                correctness=int(request.form["correctness"]),
                correctness_explanation=request.form.get("correctness_explanation", ""),
                completeness=int(request.form["completeness"]),
                completeness_explanation=request.form.get(
                    "completeness_explanation", ""
                ),
                clarity=int(request.form["clarity"]),
                clarity_explanation=request.form.get("clarity_explanation", ""),
                overall_score=float(request.form["overall_score"]),
                overall_explanation=request.form.get("overall_explanation", ""),
                reasoning=request.form.get("reasoning", ""),
            )

            # Mark answer as evaluated
            answer.evaluated = 1

            db.session.add(eval_record)
            db.session.commit()

            flash("Evaluation saved successfully!", "success")
            return redirect(url_for("evaluate_list"))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving evaluation: {e}")
            flash(
                "An error occurred while saving the evaluation. Please try again.",
                "danger",
            )
            return redirect(url_for("evaluate_answer", answer_id=answer_id))

    return render_template("evaluate.html", answer=answer, question=question, run=run)


@app.route("/browse")
def browse():
    """Browse all questions with filters."""
    source_file = request.args.get("source_file")

    query = HumanQuestion.query
    if source_file and source_file != "all":
        query = query.filter_by(source_file=source_file)

    questions = query.order_by(HumanQuestion.source_file).all()

    # Get unique sources for filter
    all_sources = db.session.query(HumanQuestion.source_file).distinct().all()
    sources = [s[0] for s in all_sources if s[0]]

    return render_template(
        "browse.html", questions=questions, sources=sources, selected_source=source_file
    )


@app.route("/question/<question_id>")
def view_question(question_id):
    """View a specific question with all student answers and evaluations."""
    question = HumanQuestion.query.get_or_404(question_id)
    answers = HumanStudentAnswer.query.filter_by(question_id=question_id).all()

    results = []
    for answer in answers:
        evaluations = HumanEvaluation.query.filter_by(
            run_id=answer.run_id, question_id=question_id
        ).all()
        results.append(
            {
                "answer": answer,
                "run": HumanRun.query.get(answer.run_id),
                "evaluations": evaluations,
            }
        )

    return render_template("question.html", question=question, results=results)


@app.route("/export")
def export_evaluations():
    """Export human evaluations to CSV with only answer ID and evaluation details."""
    evaluations = HumanEvaluation.query.all()

    import io

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "student_answer_id",
            "evaluator_name",
            "correctness",
            "correctness_explanation",
            "completeness",
            "completeness_explanation",
            "clarity",
            "clarity_explanation",
            "overall_score",
            "overall_explanation",
            "reasoning",
            "created_at",
        ]
    )

    for eval in evaluations:
        writer.writerow(
            [
                eval.student_answer_id,
                eval.evaluator_name,
                eval.correctness,
                eval.correctness_explanation,
                eval.completeness,
                eval.completeness_explanation,
                eval.clarity,
                eval.clarity_explanation,
                eval.overall_score,
                eval.overall_explanation,
                eval.reasoning,
                eval.created_at,
            ]
        )

    from flask import Response

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=human_evaluations.csv"},
    )


@app.route("/stats")
def stats():
    """Show evaluation statistics."""
    total_evaluations = HumanEvaluation.query.count()
    total_answers = HumanStudentAnswer.query.count()

    # Count unique answers that have at least one human evaluation
    evaluated = db.session.query(HumanEvaluation.student_answer_id).distinct().count()

    # Average scores
    from sqlalchemy import func

    avg_scores = db.session.query(
        func.avg(HumanEvaluation.correctness),
        func.avg(HumanEvaluation.completeness),
        func.avg(HumanEvaluation.clarity),
        func.avg(HumanEvaluation.overall_score),
    ).first()

    return render_template(
        "stats.html",
        total_evaluations=total_evaluations,
        total_answers=total_answers,
        evaluated=evaluated,
        avg_correctness=avg_scores[0] or 0,
        avg_completeness=avg_scores[1] or 0,
        avg_clarity=avg_scores[2] or 0,
        avg_overall=avg_scores[3] or 0,
    )


def sync_from_source():
    """Sync data from human_eval_source.db to human_eval.db."""
    basedir = os.path.abspath(os.path.dirname(__file__))
    source_db_path = os.path.join(basedir, "human_eval_source.db")
    target_db_path = os.path.join(basedir, "human_eval.db")

    if not os.path.exists(source_db_path):
        logger.info(f"Source database {source_db_path} not found. Skipping sync.")
        return

    logger.info(f"Syncing data from {source_db_path} to {target_db_path}...")

    conn = None
    try:
        conn = sqlite3.connect(target_db_path)
        cursor = conn.cursor()

        # Start transaction
        cursor.execute("BEGIN TRANSACTION")

        cursor.execute(f"ATTACH DATABASE '{source_db_path}' AS source")

        # Sync Questions
        cursor.execute(
            "INSERT OR REPLACE INTO human_question SELECT * FROM source.human_question"
        )

        # Sync Runs
        cursor.execute(
            "INSERT OR REPLACE INTO human_run SELECT * FROM source.human_run"
        )

        # Sync Student Answers (preserving evaluated flag)
        # 1. Insert new rows
        cursor.execute(
            """
            INSERT OR IGNORE INTO human_student_answer (id, run_id, question_id, student_answer, cost, timestamp, evaluated)
            SELECT id, run_id, question_id, student_answer, cost, timestamp, evaluated FROM source.human_student_answer
            """
        )

        # 2. Update existing rows using UPDATE FROM syntax (SQLite 3.33.0+)
        cursor.execute(
            """
            UPDATE human_student_answer
            SET
                run_id = s.run_id,
                question_id = s.question_id,
                student_answer = s.student_answer,
                cost = s.cost,
                timestamp = s.timestamp
            FROM source.human_student_answer AS s
            WHERE human_student_answer.id = s.id
            """
        )

        conn.commit()
        logger.info("Sync completed successfully.")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error during sync: {e}")
    finally:
        if conn:
            conn.close()


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions."""
    if isinstance(e, HTTPException):
        return e
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return (
        jsonify(
            {"error": "An internal server error occurred. Please try again later."}
        ),
        500,
    )


# Initialize database and sync on startup
with app.app_context():
    try:
        db.create_all()
        sync_from_source()
    except Exception as e:
        logger.error(f"Initialization error: {e}")


if __name__ == "__main__":
    logger.info("📊 Human Evaluation App")
    logger.info("=" * 40)

    logger.info("🚀 Starting Flask server at http://localhost:5000")
    logger.info("Press Ctrl+C to stop")
    app.run(debug=True, port=5000)
