"""Shared state models for the Observer/Think/Act loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by every tool invocation."""

    name: str
    input: str
    output: str
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class StepRecord:
    """One completed Observer/Think/Act iteration."""

    iteration: int
    observation: str
    thought: str
    action: str
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class Decision:
    """The Think phase output."""

    kind: Literal["tool", "finish"]
    thought: str
    tool_name: str | None = None
    tool_input: str | None = None
    final_answer: str | None = None


@dataclass
class AgentState:
    """Mutable state shared across the loop."""

    user_input: str
    system_prompts: list[str] = field(default_factory=list)
    scratchpad: list[StepRecord] = field(default_factory=list)
    final_answer: str | None = None

    @property
    def previous_tool_results(self) -> list[ToolResult]:
        """Return prior tool results in execution order."""

        return [step.tool_result for step in self.scratchpad if step.tool_result]
