from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    ForeignKey,
    UniqueConstraint,
    or_,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy import func, text, inspect
import json
import time
from datetime import datetime
from typing import Any
from logger import get_logger

Base = declarative_base()
logger = get_logger("benchmarking")


class Question(Base):
    __tablename__ = "questions"
    id: Any = Column(String, primary_key=True)
    question_text: Any = Column(Text, nullable=False)
    solution: Any = Column(Text)
    source_file: Any = Column(String)
    metadata_json: Any = Column(Text)  # Stored as JSON string

    evaluations = relationship("Evaluation", back_populates="question")
    costs = relationship("Cost", back_populates="question")


class Run(Base):
    __tablename__ = "runs"
    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    student_model: Any = Column(String, nullable=False)
    timestamp: Any = Column(String, nullable=False)
    status: Any = Column(
        String, default="incomplete"
    )  # "incomplete" or "complete_<run_id>"

    evaluations = relationship("Evaluation", back_populates="run")
    costs = relationship("Cost", back_populates="run")

    __table_args__ = (
        UniqueConstraint("student_model", "status", name="uq_student_model_status"),
    )


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    run_id: Any = Column(Integer, ForeignKey("runs.id"))
    question_id: Any = Column(String, ForeignKey("questions.id"))
    evaluator_model: Any = Column(String)
    student_answer: Any = Column(Text)
    correctness: Any = Column(Integer)
    completeness: Any = Column(Integer)
    clarity: Any = Column(Integer)
    overall_score: Any = Column(Float)
    cost: Any = Column(Float)
    raw_evaluation: Any = Column(Text)  # Stored as JSON string

    run = relationship("Run", back_populates="evaluations")
    question = relationship("Question", back_populates="evaluations")

    __table_args__ = (
        UniqueConstraint(
            "run_id", "question_id", "evaluator_model", name="uq_run_question_evaluator"
        ),
    )


class Cost(Base):
    __tablename__ = "costs"
    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    run_id: Any = Column(Integer, ForeignKey("runs.id"))
    question_id: Any = Column(String, ForeignKey("questions.id"))
    model: Any = Column(String)
    role: Any = Column(String)
    prompt_tokens: Any = Column(Integer)
    completion_tokens: Any = Column(Integer)
    total_tokens: Any = Column(Integer)
    cost: Any = Column(Float)
    timestamp: Any = Column(String)

    run = relationship("Run", back_populates="costs")
    question = relationship("Question", back_populates="costs")


class StudentAnswer(Base):
    """Store generated student answers before evaluation for resumability."""

    __tablename__ = "student_answers"
    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    run_id: Any = Column(Integer, ForeignKey("runs.id"))
    question_id: Any = Column(String, ForeignKey("questions.id"))
    student_answer: Any = Column(Text)
    cost: Any = Column(Float)
    timestamp: Any = Column(String)
    evaluated: Any = Column(Integer, default=0)  # 0=pending, 1=evaluated


class Database:
    def __init__(self, db_url=None):
        """
        Initialize database connection.

        Args:
            db_url: Database URL. If None, reads from DATABASE_URL env var.
                   Format: postgresql://user:password@localhost:5432/benchmarking
                   Falls back to SQLite if not provided.
        """
        import os

        if db_url is None:
            db_url = os.environ.get("DATABASE_URL")

        if db_url is None:
            # Fallback to SQLite for backward compatibility
            db_url = "sqlite:///benchmarking.db"
            logger.info("Using SQLite database: benchmarking.db")
            self.engine = create_engine(
                db_url,
                connect_args={
                    "timeout": 60,  # Increased timeout for SQLite locks
                    "check_same_thread": False,
                },
            )
        else:
            # PostgreSQL or other database
            logger.info(f"Connecting to database: {db_url.split('@')[0]}...")
            self.engine = create_engine(
                db_url, pool_size=10, max_overflow=20, pool_pre_ping=True
            )

        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}", exc_info=True)
            raise

        self._migrate_db()
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def _migrate_db(self):
        """Simple migration to add columns if they don't exist."""
        from sqlalchemy import text, inspect

        inspector = inspect(self.engine)

        # Check if cost column exists in evaluations table
        columns = [col["name"] for col in inspector.get_columns("evaluations")]
        if "cost" not in columns:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE evaluations ADD COLUMN cost FLOAT"))
                    conn.commit()
                    logger.info(
                        "Database migration: Added cost column to evaluations table"
                    )
            except Exception as e:
                logger.error(f"Database migration failed (cost): {e}", exc_info=True)

        # Check if status column exists in runs table
        run_columns = [col["name"] for col in inspector.get_columns("runs")]
        if "status" not in run_columns:
            try:
                with self.engine.connect() as conn:
                    conn.execute(
                        text(
                            "ALTER TABLE runs ADD COLUMN status VARCHAR DEFAULT 'incomplete'"
                        )
                    )
                    conn.commit()
                    logger.info("Database migration: Added status column to runs table")
            except Exception as e:
                logger.error(f"Database migration failed (status): {e}", exc_info=True)

        # Add unique constraint index if it doesn't exist
        try:
            indices = inspector.get_indexes("runs")
            index_names = [idx["name"] for idx in indices]
            if "uq_student_model_status" not in index_names:
                with self.engine.connect() as conn:
                    conn.execute(
                        text(
                            "CREATE UNIQUE INDEX uq_student_model_status ON runs (student_model, status)"
                        )
                    )
                    conn.commit()
                    logger.info(
                        "Database migration: Added unique index uq_student_model_status to runs table"
                    )
        except Exception as e:
            logger.error(
                f"Database migration failed (unique index runs): {e}", exc_info=True
            )

        # Add unique constraint index for evaluations if it doesn't exist
        try:
            indices = inspector.get_indexes("evaluations")
            index_names = [idx["name"] for idx in indices]
            if "uq_run_question_evaluator" not in index_names:
                with self.engine.connect() as conn:
                    conn.execute(
                        text(
                            "CREATE UNIQUE INDEX uq_run_question_evaluator ON evaluations (run_id, question_id, evaluator_model)"
                        )
                    )
                    conn.commit()
                    logger.info(
                        "Database migration: Added unique index uq_run_question_evaluator to evaluations table"
                    )
        except Exception as e:
            logger.error(
                f"Database migration failed (unique index evaluations): {e}",
                exc_info=True,
            )

    def get_session(self):
        return self.Session()

    def close_session(self):
        """Close the current session."""
        self.Session.remove()

    def _execute_with_retry(self, func, max_retries=3):
        """Execute a database operation with retry logic for locked database."""
        for attempt in range(max_retries):
            try:
                return func()
            except OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait_time = 0.1 * (2**attempt)
                    logger.warning(
                        f"Database locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)  # Exponential backoff
                    continue
                logger.error(
                    f"Database operation failed after {max_retries} retries: {str(e)}"
                )
                raise
        return None

    def add_question(self, q_id, text, solution, source_file, metadata=None):
        """Add a question to the database. Uses merge to handle duplicates gracefully."""

        def _add():
            session = self.get_session()
            try:
                new_q = Question(
                    id=q_id,
                    question_text=text,
                    solution=solution,
                    source_file=source_file,
                    metadata_json=json.dumps(metadata) if metadata else None,
                )
                session.merge(new_q)
                session.commit()
                logger.debug(f"Added/Updated question in database - QuestionID: {q_id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Error adding question {q_id[:8]}: {e}")
                raise
            finally:
                session.close()

        return self._execute_with_retry(_add)

    def create_run(self, student_model, timestamp):
        session = self.get_session()
        try:
            new_run = Run(
                student_model=student_model, timestamp=timestamp, status="incomplete"
            )
            session.add(new_run)
            session.commit()
            run_id = new_run.id
            logger.info(f"Created new run with ID: {run_id} for model: {student_model}")
            return run_id
        except Exception as e:
            logger.error(f"Failed to create run: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def get_or_create_run(self, student_model):
        """
        Get the most recent incomplete run for this model or create a new one.
        A run is defined by the student model name.
        Uses a more robust approach to handle concurrent creation.
        """
        session = self.get_session()
        try:
            # Try to find existing run first
            existing_run = (
                session.query(Run)
                .filter(Run.student_model == student_model, Run.status == "incomplete")
                .order_by(Run.id.desc())
                .first()
            )

            if existing_run:
                run_id = existing_run.id
                logger.info(
                    f"Using existing incomplete run - RunID: {run_id}, Model: {student_model}"
                )
                return run_id, True

            # No incomplete run exists, try to create a new one
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_run = Run(
                student_model=student_model, timestamp=timestamp, status="incomplete"
            )
            session.add(new_run)
            try:
                session.commit()
                run_id = new_run.id
                logger.info(
                    f"Created new run - RunID: {run_id}, Model: {student_model}, Timestamp: {timestamp}"
                )
                return run_id, False
            except IntegrityError:
                # Race condition: another process created the run just now
                session.rollback()
                # Re-query to get the one that was just created
                existing_run = (
                    session.query(Run)
                    .filter(
                        Run.student_model == student_model, Run.status == "incomplete"
                    )
                    .order_by(Run.id.desc())
                    .first()
                )
                if existing_run:
                    logger.info(
                        f"Using existing incomplete run (after race) - RunID: {existing_run.id}, Model: {student_model}"
                    )
                    return existing_run.id, True
                raise
        except Exception as e:
            session.rollback()
            logger.error(
                f"Error in get_or_create_run for {student_model}: {e}", exc_info=True
            )
            raise
        finally:
            session.close()

    def mark_run_complete(self, run_id):
        """Mark a run as complete."""
        session = self.get_session()
        try:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = f"complete_{run_id}"
                session.commit()
                logger.info(f"Marked run {run_id} as complete")
        finally:
            session.close()

    def add_evaluation(
        self,
        run_id,
        question_id,
        evaluator_model,
        student_answer,
        correctness,
        completeness,
        clarity,
        overall_score,
        cost,
        raw_evaluation,
    ):
        """Add or update an evaluation. Uses merge to handle duplicates gracefully."""

        def _add():
            session = self.get_session()
            try:
                # Check if evaluation already exists for this run, question, and evaluator
                existing = (
                    session.query(Evaluation)
                    .filter_by(
                        run_id=run_id,
                        question_id=question_id,
                        evaluator_model=evaluator_model,
                    )
                    .first()
                )

                if existing:
                    # Update existing evaluation
                    existing.student_answer = student_answer
                    existing.correctness = correctness
                    existing.completeness = completeness
                    existing.clarity = clarity
                    existing.overall_score = overall_score
                    existing.cost = cost
                    existing.raw_evaluation = json.dumps(raw_evaluation)
                    logger.info(
                        f"Updated evaluation - RunID: {run_id}, QuestionID: {question_id}, EvalID: {existing.id}, Evaluator: {evaluator_model}"
                    )
                else:
                    # Create new evaluation
                    new_eval = Evaluation(
                        run_id=run_id,
                        question_id=question_id,
                        evaluator_model=evaluator_model,
                        student_answer=student_answer,
                        correctness=correctness,
                        completeness=completeness,
                        clarity=clarity,
                        overall_score=overall_score,
                        cost=cost,
                        raw_evaluation=json.dumps(raw_evaluation),
                    )
                    session.add(new_eval)
                    logger.info(
                        f"Saved evaluation - RunID: {run_id}, QuestionID: {question_id}, Evaluator: {evaluator_model}"
                    )

                session.commit()
            except Exception as e:
                session.rollback()
                # Ignore duplicate key errors (race condition)
                if "UNIQUE constraint failed" not in str(
                    e
                ) and "duplicate key" not in str(e):
                    logger.error(
                        f"Error adding/updating evaluation for question {question_id[:8]}: {e}",
                        exc_info=True,
                    )
                    raise
            finally:
                session.close()

        return self._execute_with_retry(_add)

    def add_cost(
        self,
        run_id,
        question_id,
        model,
        role,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost,
        timestamp,
    ):
        def _add():
            session = self.get_session()
            try:
                new_cost = Cost(
                    run_id=run_id,
                    question_id=question_id,
                    model=model,
                    role=role,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                    timestamp=timestamp,
                )
                session.add(new_cost)
                session.commit()
                cost_id = new_cost.id
                logger.debug(
                    f"Saved cost - RunID: {run_id}, QuestionID: {question_id}, CostID: {cost_id}, Model: {model}, Role: {role}, Tokens: {total_tokens}, Cost: ${cost:.4f}"
                )
            except Exception as e:
                session.rollback()
                print(f"Error adding cost for question {question_id}: {e}")
                raise
            finally:
                session.close()

        return self._execute_with_retry(_add)

    def get_run_summary(self, run_id):
        session = self.get_session()
        try:
            from sqlalchemy import func

            results = (
                session.query(
                    Evaluation.evaluator_model,
                    func.count(Evaluation.id),
                    func.avg(Evaluation.correctness),
                    func.avg(Evaluation.completeness),
                    func.avg(Evaluation.clarity),
                    func.avg(Evaluation.overall_score),
                    func.sum(Evaluation.cost),
                )
                .filter(Evaluation.run_id == run_id)
                .group_by(Evaluation.evaluator_model)
                .all()
            )
            return results
        finally:
            session.close()

    def get_total_costs(self, run_id):
        session = self.get_session()
        try:
            from sqlalchemy import func

            results = (
                session.query(
                    Cost.model,
                    Cost.role,
                    func.sum(Cost.prompt_tokens),
                    func.sum(Cost.completion_tokens),
                    func.sum(Cost.cost),
                )
                .filter(Cost.run_id == run_id)
                .group_by(Cost.model, Cost.role)
                .all()
            )
            return results
        finally:
            session.close()

    def get_cached_results(
        self, student_model_name, evaluator_model_names=None, include_run_id=None
    ):
        """
        Fetch existing evaluations for a student model to use as cache.

        Args:
            student_model_name: Name of the student model.
            evaluator_model_names: List of evaluator models required. If provided,
                                  only questions evaluated by ALL these models will be returned.
            include_run_id: Optional run ID to include even if it's marked as incomplete.
        """
        session = self.get_session()
        try:
            # Join Evaluation with Run to filter by student_model and complete status
            # If include_run_id is provided, we also include results from that specific run
            if include_run_id:
                query = (
                    session.query(Evaluation)
                    .join(Run)
                    .filter(
                        Run.student_model == student_model_name,
                        or_(Run.status != "incomplete", Run.id == include_run_id),
                    )
                )
            else:
                query = (
                    session.query(Evaluation)
                    .join(Run)
                    .filter(
                        Run.student_model == student_model_name,
                        Run.status != "incomplete",
                    )
                )

            if evaluator_model_names:
                query = query.filter(
                    Evaluation.evaluator_model.in_(evaluator_model_names)
                )

            # Group by question_id
            temp_cache = {}
            for eval_obj in query.all():
                q_id = eval_obj.question_id
                if q_id not in temp_cache:
                    temp_cache[q_id] = {
                        "question_id": q_id,
                        "student_answer": eval_obj.student_answer,
                        "evaluations": {},
                        "gen_cost": 0.0,
                        "eval_cost": 0.0,
                    }

                # Reconstruct the evaluation data
                raw_eval_str = eval_obj.raw_evaluation
                raw_eval = (
                    json.loads(str(raw_eval_str)) if raw_eval_str is not None else {}
                )
                temp_cache[q_id]["evaluations"][eval_obj.evaluator_model] = {
                    "result": raw_eval,
                    "cost": eval_obj.cost or 0.0,
                }
                temp_cache[q_id]["eval_cost"] += eval_obj.cost or 0.0

            # Filter: Only include questions that have ALL required evaluators
            cache = {}
            if evaluator_model_names:
                required_set = set(evaluator_model_names)
                for q_id, data in temp_cache.items():
                    evaluators_present = set(data["evaluations"].keys())
                    if required_set.issubset(evaluators_present):
                        cache[q_id] = data
            else:
                cache = temp_cache

            # Also fetch student costs for these questions
            if cache:
                cost_query = (
                    session.query(Cost)
                    .join(Run)
                    .filter(Run.student_model == student_model_name)
                    .filter(Cost.role == "student")
                    .filter(Cost.question_id.in_(list(cache.keys())))
                )
                for cost_obj in cost_query.all():
                    if cost_obj.question_id in cache:
                        cache[cost_obj.question_id]["gen_cost"] = cost_obj.cost or 0.0

            return cache
        finally:
            session.close()

    def save_student_answer(self, run_id, question_id, student_answer, cost, timestamp):
        """Save a generated student answer before evaluation. Uses merge to handle duplicates."""

        def _save():
            session = self.get_session()
            try:
                # Use merge to handle potential race conditions where the answer might already exist
                # We need to find if it exists first to get the ID if we want to merge correctly,
                # but StudentAnswer doesn't have a unique constraint on (run_id, question_id) in the schema?
                # Wait, let me check the schema.

                existing = (
                    session.query(StudentAnswer)
                    .filter_by(run_id=run_id, question_id=question_id)
                    .first()
                )

                if existing:
                    existing.student_answer = student_answer
                    existing.cost = cost
                    existing.timestamp = timestamp
                    # Keep evaluated status as is or reset? Usually reset if answer changed.
                    # But here we are likely just re-saving the same answer.
                else:
                    new_answer = StudentAnswer(
                        run_id=run_id,
                        question_id=question_id,
                        student_answer=student_answer,
                        cost=cost,
                        timestamp=timestamp,
                        evaluated=0,
                    )
                    session.add(new_answer)

                session.commit()
                logger.info(
                    f"Saved/Updated student answer - RunID: {run_id}, QuestionID: {question_id}"
                )
            except Exception as e:
                session.rollback()
                logger.error(
                    f"Error saving student answer for question {question_id}: {e}"
                )
                raise
            finally:
                session.close()

        return self._execute_with_retry(_save)

    def mark_answer_evaluated(self, run_id, question_id):
        """Mark a student answer as evaluated."""
        session = self.get_session()
        try:
            answer = (
                session.query(StudentAnswer)
                .filter_by(run_id=run_id, question_id=question_id)
                .first()
            )
            if answer:
                answer.evaluated = 1
                session.commit()
                logger.info(
                    f"Marked answer as evaluated - RunID: {run_id}, QuestionID: {question_id}, AnswerID: {answer.id}"
                )
            else:
                # Answer might not exist if it was from cache, which is fine
                logger.debug(
                    f"No answer found to mark as evaluated (likely from cache) - RunID: {run_id}, QuestionID: {question_id}"
                )
        except Exception as e:
            session.rollback()
            print(f"Error marking answer as evaluated for question {question_id}: {e}")
        finally:
            session.close()

    def get_unevaluated_answers(self, run_id):
        """Get all student answers that haven't been evaluated yet."""
        session = self.get_session()
        try:
            answers = (
                session.query(StudentAnswer).filter_by(run_id=run_id, evaluated=0).all()
            )
            return [
                {
                    "question_id": a.question_id,
                    "student_answer": a.student_answer,
                    "cost": a.cost,
                }
                for a in answers
            ]
        finally:
            session.close()
