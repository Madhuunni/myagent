"""Think stage: planners that convert observations into decisions."""

from __future__ import annotations

from collections.abc import Callable

from .state import AgentState, Decision

Planner = Callable[[AgentState, str], Decision]


class Thinker:
    """Decision component wrapping a planner callable."""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or default_planner

    def think(self, state: AgentState, observation: str) -> Decision:
        return self.planner(state, observation)


def default_planner(state: AgentState, observation: str) -> Decision:
    """Simple bootstrap planner for local testing before wiring a real LLM."""

    if state.previous_tool_results:
        return Decision(kind="finish", thought="A tool result is available; produce final answer.")

    text = state.user_input.strip()
    if text.lower().startswith("calculate:"):
        return Decision(
            kind="tool",
            thought="The user asked for a calculation, so call the calculator tool.",
            tool_name="calculator",
            tool_input=text.split(":", 1)[1].strip(),
        )

    return Decision(kind="tool", thought="Use the echo tool as the default bootstrap action.", tool_name="echo", tool_input=text)
