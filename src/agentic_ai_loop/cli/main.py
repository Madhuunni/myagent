"""Typer CLI for running the agent and monitoring persisted reports."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from agentic_ai_loop.core import AgentLoop
from agentic_ai_loop.observability import record_run_started
from agentic_ai_loop.persistence import PromptExecutionRepository, PromptRun
from agentic_ai_loop.tools import default_registry

app = typer.Typer(help="Run and monitor the agentic AI loop.")
console = Console()


@app.command()
def run(prompt: str, database_url: str = "sqlite:///agentic_ai_loop.db") -> None:
    """Run a prompt through the local agent loop and persist each step."""

    record_run_started(prompt)
    repository = PromptExecutionRepository(database_url)
    state = AgentLoop(tools=default_registry(), repository=repository).run(prompt)
    console.print(state.final_answer)


@app.command()
def report(database_url: str = "sqlite:///agentic_ai_loop.db", limit: int = 10) -> None:
    """Show recent persisted prompt execution reports."""

    repository = PromptExecutionRepository(database_url)
    table = Table(title="Agent Runs")
    table.add_column("ID")
    table.add_column("Prompt")
    table.add_column("Final Answer")
    table.add_column("Finished")
    with repository.session_factory() as session:
        rows = session.scalars(select(PromptRun).order_by(PromptRun.id.desc()).limit(limit)).all()
        for row in rows:
            table.add_row(str(row.id), row.user_input, row.final_answer or "", str(row.finished_at or ""))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
