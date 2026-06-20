"""Run agent loop and tool usage examples with `python -m agentic_ai_loop.demo`."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from .core import AgentLoop
from .tools import default_registry


def print_section(title: str) -> None:
    """Print a readable section heading for the demo output."""

    print(f"\n=== {title} ===")


def run_agent_examples() -> None:
    """Show how the default planner routes prompts to bundled tools."""

    agent = AgentLoop(tools=default_registry())
    examples = [
        ("Calculator through agent", "calculate: 2 + 3 * 4"),
        ("Echo through agent", "Summarize this exact sentence."),
    ]

    print_section("Agent loop examples")
    for label, prompt in examples:
        state = agent.run(prompt, system_prompts=["You are a concise technical assistant."])
        print(f"\n{label}")
        print(f"prompt: {prompt}")
        print(state.final_answer)


def run_tool_registry_examples() -> None:
    """Show direct usage of each tool registered by default_registry()."""

    registry = default_registry()
    examples = [
        ("calculator", "(12 / 3) + 7"),
        ("calculator", "2 ** 5"),
        ("echo", "Echo tool returns the exact input text."),
        ("echo", "Use echo for prompts that do not request calculation."),
    ]

    print_section("Direct tool registry examples")
    for tool_name, tool_input in examples:
        result = registry.run(tool_name, tool_input)
        print(f"{tool_name}({tool_input!r}) -> ok={result.ok}, output={result.output!r}")


def run_report_history_example() -> None:
    """Persist runs and show how to check recent report history."""

    print_section("Report history example")
    if find_spec("sqlalchemy") is None:
        print("Install project dependencies to run the persisted report history example.")
        print("CLI equivalent after installation: agentic-ai-loop report --limit 5")
        return

    from sqlalchemy import select

    from .persistence import PromptExecutionRepository, PromptRun

    with TemporaryDirectory() as tmpdir:
        database_url = f"sqlite:///{Path(tmpdir) / 'agentic_ai_loop_demo.db'}"
        repository = PromptExecutionRepository(database_url)
        agent = AgentLoop(tools=default_registry(), repository=repository)

        for prompt in ["calculate: 10 - 4", "Remember this demo run"]:
            state = agent.run(prompt)
            print(f"saved prompt: {prompt!r} -> {state.final_answer!r}")

        print("\nRecent persisted runs:")
        with repository.session_factory() as session:
            rows = session.scalars(select(PromptRun).order_by(PromptRun.id.desc()).limit(5)).all()
            for row in rows:
                print(f"#{row.id}: prompt={row.user_input!r}, final_answer={row.final_answer!r}")


def main() -> None:
    run_agent_examples()
    run_tool_registry_examples()
    run_report_history_example()


if __name__ == "__main__":
    main()
