"""Done evaluation and final response formatting."""

from __future__ import annotations

from .config import LoopConfig
from .state import AgentState, ToolResult


class DoneEvaluator:
    """Completion checks and final response rendering."""

    def __init__(self, config: LoopConfig | None = None) -> None:
        self.config = config or LoopConfig()

    def should_finish_after_action(self, result: ToolResult) -> bool:
        return self.config.finish_when_tool_succeeds and result.ok

    def format_final_response(self, state: AgentState, stopped_by_limit: bool = False) -> str:
        latest = state.previous_tool_results[-1] if state.previous_tool_results else None
        status = "stopped_by_iteration_limit" if stopped_by_limit else "completed"
        answer = latest.output if latest else "No tool result was needed."
        return f"status: {status}\nanswer: {answer}\niterations: {len(state.scratchpad)}"
