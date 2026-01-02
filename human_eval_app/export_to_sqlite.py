import sqlite3
import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path to import database.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database, Question, Run, StudentAnswer

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "export.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def validate_path(path):
    """Validate that the path is within the expected directory."""
    abs_path = os.path.abspath(path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not abs_path.startswith(base_dir):
        raise ValueError(f"Unsafe path detected: {path}")
    return abs_path


def export_to_human_eval_db(target_db_path="human_eval_app/human_eval_source.db"):
    """
    Copies data from the main PostgreSQL database to the human evaluation SQLite database.
    This database is intended to be read-only for the human evaluation app.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not found in environment.")
        return

    # Basic URL validation
    if not db_url.startswith(("postgresql://", "postgres://")):
        logger.error("Invalid DATABASE_URL format. Must be a PostgreSQL URL.")
        return

    try:
        target_db_path = validate_path(target_db_path)
    except ValueError as e:
        logger.error(str(e))
        return

    # Ensure the target directory exists
    target_dir = os.path.dirname(target_db_path)
    if target_dir:
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {target_dir}: {e}")
            return

    logger.info(f"Exporting data from PostgreSQL to {target_db_path}...")

    source_db = None
    session = None
    dest_conn = None

    try:
        # Initialize source database
        source_db = Database(db_url)
        session = source_db.Session()

        # Connect to target SQLite
        if os.path.exists(target_db_path):
            try:
                os.remove(target_db_path)
            except Exception as e:
                logger.error(
                    f"Failed to remove existing database file {target_db_path}: {e}"
                )
                return

        dest_conn = sqlite3.connect(target_db_path)
        dest_cursor = dest_conn.cursor()

        # Create tables in target (only data tables, no evaluation table)
        dest_cursor.execute(
            """
            CREATE TABLE human_question (
                id TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                solution TEXT,
                source_file TEXT,
                metadata_json TEXT
            )
        """
        )

        dest_cursor.execute(
            """
            CREATE TABLE human_run (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_model TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """
        )

        dest_cursor.execute(
            """
            CREATE TABLE human_student_answer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                question_id TEXT,
                student_answer TEXT,
                cost REAL,
                timestamp TEXT,
                evaluated INTEGER DEFAULT 0,
                FOREIGN KEY(run_id) REFERENCES human_run(id),
                FOREIGN KEY(question_id) REFERENCES human_question(id)
            )
        """
        )

        # Copy Questions
        logger.info("Copying questions...")
        questions = session.query(Question).all()
        dest_cursor.executemany(
            "INSERT INTO human_question VALUES (?, ?, ?, ?, ?)",
            [
                (q.id, q.question_text, q.solution, q.source_file, q.metadata_json)
                for q in questions
            ],
        )

        # Copy Runs
        logger.info("Copying runs...")
        runs = session.query(Run).all()
        dest_cursor.executemany(
            "INSERT INTO human_run VALUES (?, ?, ?)",
            [(r.id, r.student_model, r.timestamp) for r in runs],
        )

        # Copy Student Answers
        logger.info("Copying student answers...")
        answers = session.query(StudentAnswer).all()
        dest_cursor.executemany(
            "INSERT INTO human_student_answer VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    a.id,
                    a.run_id,
                    a.question_id,
                    a.student_answer,
                    a.cost,
                    a.timestamp,
                    0,  # Reset evaluated flag for human evaluators
                )
                for a in answers
            ],
        )

        dest_conn.commit()
        logger.info(f"Successfully exported data to {target_db_path}")

    except Exception as e:
        logger.error(f"An error occurred during export: {e}", exc_info=True)
        if dest_conn:
            dest_conn.rollback()
    finally:
        if dest_conn:
            dest_conn.close()
        if session:
            session.close()


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(script_dir, "human_eval_source.db")
    export_to_human_eval_db(target_path)
