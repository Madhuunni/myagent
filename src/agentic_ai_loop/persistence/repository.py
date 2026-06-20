"""SQLAlchemy persistence for prompts and execution steps."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from agentic_ai_loop.agent.state import StepRecord


class Base(DeclarativeBase):
    pass


class PromptRun(Base):
    """One user prompt execution run."""

    __tablename__ = "prompt_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_input: Mapped[str] = mapped_column(Text)
    system_prompts: Mapped[list[str]] = mapped_column(JSON, default=list)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    steps: Mapped[list["ExecutionStep"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class ExecutionStep(Base):
    """One persisted Observer/Think/Act step."""

    __tablename__ = "execution_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("prompt_runs.id"), index=True)
    iteration: Mapped[int] = mapped_column(Integer)
    observation: Mapped[str] = mapped_column(Text)
    thought: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(255))
    tool_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tool_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    tool_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    run: Mapped[PromptRun] = relationship(back_populates="steps")


class PromptExecutionRepository:
    """File or Postgres-backed repository for every prompt/execution step."""

    def __init__(self, database_url: str = "sqlite:///agentic_ai_loop.db") -> None:
        self.engine = create_engine(database_url)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def create_run(self, user_input: str, system_prompts: list[str]) -> int:
        with self.session_factory() as session:
            run = PromptRun(user_input=user_input, system_prompts=system_prompts)
            session.add(run)
            session.commit()
            return run.id

    def record_step(self, run_id: int, step: StepRecord) -> None:
        result = step.tool_result
        with self.session_factory() as session:
            session.add(
                ExecutionStep(
                    run_id=run_id,
                    iteration=step.iteration,
                    observation=step.observation,
                    thought=step.thought,
                    action=step.action,
                    tool_name=result.name if result else None,
                    tool_input=result.input if result else None,
                    tool_output=result.output if result else None,
                    tool_ok=result.ok if result else None,
                    tool_error=result.error if result else None,
                )
            )
            session.commit()

    def finish_run(self, run_id: int, final_answer: str) -> None:
        with self.session_factory() as session:
            run = session.get(PromptRun, run_id)
            if run is not None:
                run.final_answer = final_answer
                run.finished_at = datetime.now(timezone.utc)
                session.commit()
