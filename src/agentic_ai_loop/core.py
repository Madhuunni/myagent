"""Core Observer -> Think -> Act agent loop architecture.

The implementation intentionally keeps the first version small and explicit:
- Observer builds the bounded context visible to the model/planner.
- Think decides whether to call a tool or finish.
- Act executes the requested tool and records the result.

A production system can swap the deterministic planner with a LangChain chat model
or graph node without changing the state and tool abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol


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


@dataclass(frozen=True)
class LoopConfig:
    """Runtime limits and completion criteria."""

    max_iterations: int = 6
    max_observation_chars: int = 4_000
    finish_when_tool_succeeds: bool = True


class ToolExecutor(Protocol):
    """Minimal protocol implemented by a tool registry."""

    def run(self, name: str, tool_input: str) -> ToolResult:
        """Execute a tool by name."""


Planner = Callable[[AgentState, str], Decision]


class AgentLoop:
    """A small, testable Observer -> Think -> Act loop."""

    def __init__(self, tools: ToolExecutor, config: LoopConfig | None = None, planner: Planner | None = None) -> None:
        self.tools = tools
        self.config = config or LoopConfig()
        self.planner = planner or self._default_planner

    def run(self, user_input: str, system_prompts: list[str] | None = None) -> AgentState:
        """Run until the planner finishes or the iteration budget is exhausted."""

        state = AgentState(user_input=user_input, system_prompts=system_prompts or [])

        for iteration in range(1, self.config.max_iterations + 1):
            observation = self.observe(state)
            decision = self.think(state, observation)

            if decision.kind == "finish":
                state.final_answer = decision.final_answer or decision.thought
                state.scratchpad.append(
                    StepRecord(iteration=iteration, observation=observation, thought=decision.thought, action="finish")
                )
                return state

            result = self.act(decision)
            state.scratchpad.append(
                StepRecord(
                    iteration=iteration,
                    observation=observation,
                    thought=decision.thought,
                    action=f"tool:{decision.tool_name}",
                    tool_result=result,
                )
            )

            if self._should_finish_after_action(result):
                state.final_answer = self.format_final_response(state)
                return state

        state.final_answer = self.format_final_response(state, stopped_by_limit=True)
        return state

    def observe(self, state: AgentState) -> str:
        """Build a bounded context from prompts, user input, and previous tool calls."""

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

    def think(self, state: AgentState, observation: str) -> Decision:
        """Choose the next action using the configured planner."""

        return self.planner(state, observation)

    def act(self, decision: Decision) -> ToolResult:
        """Execute the tool selected by the Think phase."""

        if decision.kind != "tool" or not decision.tool_name:
            return ToolResult(name="none", input="", output="No action requested.", ok=False, error="missing_tool")
        return self.tools.run(decision.tool_name, decision.tool_input or "")

    def format_final_response(self, state: AgentState, stopped_by_limit: bool = False) -> str:
        """Create a stable response payload for the user."""

        latest = state.previous_tool_results[-1] if state.previous_tool_results else None
        status = "stopped_by_iteration_limit" if stopped_by_limit else "completed"
        answer = latest.output if latest else "No tool result was needed."
        return f"status: {status}\nanswer: {answer}\niterations: {len(state.scratchpad)}"

    def _should_finish_after_action(self, result: ToolResult) -> bool:
        return self.config.finish_when_tool_succeeds and result.ok

    def _default_planner(self, state: AgentState, observation: str) -> Decision:
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

        return Decision(
            kind="tool",
            thought="Use the echo tool as the default bootstrap action.",
            tool_name="echo",
            tool_input=text,
        )

    @staticmethod
    def _format_tool_history(results: list[ToolResult]) -> str:
        if not results:
            return "No tools have been called yet."
        return "\n".join(
            f"{idx}. {result.name}({result.input!r}) -> ok={result.ok}, output={result.output!r}"
            for idx, result in enumerate(results, start=1)
        )
