from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json

Base = declarative_base()


class Question(Base):
    __tablename__ = "questions"
    id = Column(String, primary_key=True)
    question_text = Column(Text, nullable=False)
    solution = Column(Text)
    source_file = Column(String)
    metadata_json = Column(Text)  # Stored as JSON string

    evaluations = relationship("Evaluation", back_populates="question")
    costs = relationship("Cost", back_populates="question")


class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_model = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)

    evaluations = relationship("Evaluation", back_populates="run")
    costs = relationship("Cost", back_populates="run")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    question_id = Column(String, ForeignKey("questions.id"))
    evaluator_model = Column(String)
    student_answer = Column(Text)
    correctness = Column(Integer)
    completeness = Column(Integer)
    clarity = Column(Integer)
    overall_score = Column(Float)
    cost = Column(Float)
    raw_evaluation = Column(Text)  # Stored as JSON string

    run = relationship("Run", back_populates="evaluations")
    question = relationship("Question", back_populates="evaluations")


class Cost(Base):
    __tablename__ = "costs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    question_id = Column(String, ForeignKey("questions.id"))
    model = Column(String)
    role = Column(String)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost = Column(Float)
    timestamp = Column(String)

    run = relationship("Run", back_populates="costs")
    question = relationship("Question", back_populates="costs")


class StudentAnswer(Base):
    """Store generated student answers before evaluation for resumability."""

    __tablename__ = "student_answers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    question_id = Column(String, ForeignKey("questions.id"))
    student_answer = Column(Text)
    cost = Column(Float)
    timestamp = Column(String)
    evaluated = Column(Integer, default=0)  # 0=pending, 1=evaluated


class Database:
    def __init__(self, db_path="benchmarking.db"):
        self.engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"timeout": 30}
        )
        Base.metadata.create_all(self.engine)
        self._migrate_db()
        self.Session = sessionmaker(bind=self.engine)

    def _migrate_db(self):
        """Simple migration to add cost column to evaluations if it doesn't exist."""
        from sqlalchemy import text

        with self.engine.connect() as conn:
            # Check if cost column exists in evaluations
            try:
                conn.execute(text("SELECT cost FROM evaluations LIMIT 1"))
            except Exception:
                # Column doesn't exist, add it
                try:
                    conn.execute(text("ALTER TABLE evaluations ADD COLUMN cost FLOAT"))
                    conn.commit()
                except Exception as e:
                    print(f"Migration failed: {e}")

    def get_session(self):
        return self.Session()

    def add_question(self, q_id, text, solution, source_file, metadata=None):
        session = self.get_session()
        try:
            exists = session.query(Question).filter_by(id=q_id).first()
            if not exists:
                new_q = Question(
                    id=q_id,
                    question_text=text,
                    solution=solution,
                    source_file=source_file,
                    metadata_json=json.dumps(metadata) if metadata else None,
                )
                session.add(new_q)
                session.commit()
        finally:
            session.close()

    def create_run(self, student_model, timestamp):
        session = self.get_session()
        try:
            new_run = Run(student_model=student_model, timestamp=timestamp)
            session.add(new_run)
            session.commit()
            run_id = new_run.id
            return run_id
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
        session = self.get_session()
        try:
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
            session.commit()
        finally:
            session.close()

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
        finally:
            session.close()

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

    def get_cached_results(self, student_model_name):
        """Fetch existing evaluations for a student model to use as cache."""
        session = self.get_session()
        try:
            # Join Evaluation with Run to filter by student_model
            # We want to group by question_id and reconstruct the 'evaluations' dict
            query = (
                session.query(Evaluation)
                .join(Run)
                .filter(Run.student_model == student_model_name)
            )

            cache = {}
            for eval_obj in query.all():
                q_id = eval_obj.question_id
                if q_id not in cache:
                    cache[q_id] = {
                        "question_id": q_id,
                        "student_answer": eval_obj.student_answer,
                        "evaluations": {},
                        "gen_cost": 0.0,
                        "eval_cost": 0.0,
                    }

                # Reconstruct the evaluation data
                raw_eval = (
                    json.loads(eval_obj.raw_evaluation)
                    if eval_obj.raw_evaluation
                    else {}
                )
                cache[q_id]["evaluations"][eval_obj.evaluator_model] = {
                    "result": raw_eval,
                    "cost": eval_obj.cost or 0.0,
                }
                cache[q_id]["eval_cost"] += eval_obj.cost or 0.0

            # Also fetch student costs for these questions
            cost_query = (
                session.query(Cost)
                .join(Run)
                .filter(Run.student_model == student_model_name)
                .filter(Cost.role == "student")
            )
            for cost_obj in cost_query.all():
                if cost_obj.question_id in cache:
                    cache[cost_obj.question_id]["gen_cost"] = cost_obj.cost or 0.0

            return cache
        finally:
            session.close()

    def save_student_answer(self, run_id, question_id, student_answer, cost, timestamp):
        """Save a generated student answer before evaluation."""
        session = self.get_session()
        try:
            # Check if already exists
            existing = (
                session.query(StudentAnswer)
                .filter_by(run_id=run_id, question_id=question_id)
                .first()
            )
            if not existing:
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
        finally:
            session.close()

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
