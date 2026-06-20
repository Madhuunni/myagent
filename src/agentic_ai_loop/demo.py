"""Run agent loop and tool usage examples with `python -m agentic_ai_loop.demo`."""

from __future__ import annotations

from importlib.util import find_spec
import socket
from pathlib import Path
from tempfile import TemporaryDirectory

from .core import AgentLoop, AgentSettings
from .llm import build_ollama_planner
from .tools import default_registry


def print_section(title: str) -> None:
    """Print a readable section heading for the demo output."""

    print(f"\n=== {title} ===")


def run_agent_examples() -> None:
    """Show how an LLM planner routes prompts to bundled tools."""

    settings = AgentSettings()
    if not _ollama_server_available():
        print_section(f"LLM-backed agent loop examples ({settings.ollama_model})")
        print("Ollama is not reachable at localhost:11434.")
        print("Start Ollama or set AGENTIC_AI_LOOP_OLLAMA_MODEL to an available model before rerunning the demo.")
        return

    agent = AgentLoop(tools=default_registry(), planner=build_ollama_planner(settings.ollama_model))
    examples = [
        ("Calculator through agent", "calculate: 2 + 3 * 4"),
        ("HTTP GET through agent", "http: https://example.com"),
        ("Browser title through agent", "browser: https://example.com"),
        ("Echo through agent", "Summarize this exact sentence."),
    ]

    print_section(f"LLM-backed agent loop examples ({settings.ollama_model})")
    for label, prompt in examples:
        print(f"\n{label}")
        print(f"prompt: {prompt}")
        try:
            state = agent.run(prompt, system_prompts=["You are a concise technical assistant."])
        except Exception as exc:  # pragma: no cover - depends on local Ollama runtime
            print("Unable to reach the configured Ollama model.")
            print("Start Ollama or set AGENTIC_AI_LOOP_OLLAMA_MODEL to an available model.")
            print(f"error: {exc}")
            return
        print(state.final_answer)


def _ollama_server_available(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.25) -> bool:
    """Return whether the local Ollama API socket is reachable."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def run_tool_registry_examples() -> None:
    """Show direct usage of each tool registered by default_registry()."""

    registry = default_registry()
    examples = [
        ("calculator", "(12 / 3) + 7"),
        ("calculator", "2 ** 5"),
        ("http_get", "https://example.com"),
        ("page_text", "https://example.com"),
        ("browser_title", "https://example.com"),
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
