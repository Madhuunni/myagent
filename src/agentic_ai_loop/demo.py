"""Run a minimal agent loop demo with `python -m agentic_ai_loop.demo`."""

from .core import AgentLoop
from .tools import default_registry


def main() -> None:
    agent = AgentLoop(tools=default_registry())
    state = agent.run("calculate: 2 + 3 * 4", system_prompts=["You are a concise technical assistant."])
    print(state.final_answer)


if __name__ == "__main__":
    main()
