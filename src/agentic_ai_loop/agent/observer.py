"""Observer stage: builds bounded context for the planner/model."""

from __future__ import annotations

from .config import LoopConfig
from .state import AgentState, ToolResult


class Observer:
    """Build observations from prompts, user input, and tool history."""

    def __init__(self, config: LoopConfig | None = None) -> None:
        self.config = config or LoopConfig()

    def observe(self, state: AgentState) -> str:
        sections = [
            "# System prompts",
            "\n".join(state.system_prompts) or "No system prompts provided.",
            "# User input",
            state.user_input,
            "# Previous tool calls and results",
            self._format_tool_history(state.previous_tool_results),
        ]
        observation = "\n\n".join(sections)
        return observation[-self.config.max_observation_chars :]

    @staticmethod
    def _format_tool_history(results: list[ToolResult]) -> str:
        if not results:
            return "No tools have been called yet."
        return "\n".join(
            f"{idx}. {result.name}({result.input!r}) -> ok={result.ok}, output={result.output!r}"
            for idx, result in enumerate(results, start=1)
        )
