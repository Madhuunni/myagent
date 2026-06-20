"""Act stage: executes decisions using the configured tool executor."""

from __future__ import annotations

from typing import Protocol

from .state import Decision, ToolResult


class ToolExecutor(Protocol):
    """Minimal protocol implemented by a tool registry."""

    def run(self, name: str, tool_input: str) -> ToolResult:
        """Execute a tool by name."""


class Actor:
    """Execute tool decisions and normalize missing-action failures."""

    def __init__(self, tools: ToolExecutor) -> None:
        self.tools = tools

    def act(self, decision: Decision) -> ToolResult:
        if decision.kind != "tool" or not decision.tool_name:
            return ToolResult(name="none", input="", output="No action requested.", ok=False, error="missing_tool")
        return self.tools.run(decision.tool_name, decision.tool_input or "")
