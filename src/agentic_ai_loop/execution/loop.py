"""Loop controller for the Observer -> Think -> Act architecture."""

from __future__ import annotations

from agentic_ai_loop.agent.act import Actor, ToolExecutor
from agentic_ai_loop.agent.config import LoopConfig
from agentic_ai_loop.agent.evaluator import DoneEvaluator
from agentic_ai_loop.agent.observer import Observer
from agentic_ai_loop.agent.state import AgentState, Decision, StepRecord, ToolResult
from agentic_ai_loop.agent.think import Planner, Thinker
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_ai_loop.persistence.repository import PromptExecutionRepository


class AgentLoop:
    """Composable Observer -> Think -> Act loop controller."""

    def __init__(
        self,
        tools: ToolExecutor,
        config: LoopConfig | None = None,
        planner: Planner | None = None,
        repository: "PromptExecutionRepository" | None = None,
    ) -> None:
        self.config = config or LoopConfig()
        self.observer = Observer(self.config)
        self.thinker = Thinker(planner)
        self.actor = Actor(tools)
        self.evaluator = DoneEvaluator(self.config)
        self.repository = repository

    def run(self, user_input: str, system_prompts: list[str] | None = None) -> AgentState:
        state = AgentState(user_input=user_input, system_prompts=system_prompts or [])
        run_id = self.repository.create_run(user_input, state.system_prompts) if self.repository else None

        for iteration in range(1, self.config.max_iterations + 1):
            observation = self.observe(state)
            decision = self.think(state, observation)

            if decision.kind == "finish":
                state.final_answer = decision.final_answer or decision.thought
                record = StepRecord(iteration=iteration, observation=observation, thought=decision.thought, action="finish")
                state.scratchpad.append(record)
                self._record_step(run_id, record)
                self._finish_run(run_id, state.final_answer)
                return state

            result = self.act(decision)
            record = StepRecord(
                iteration=iteration,
                observation=observation,
                thought=decision.thought,
                action=f"tool:{decision.tool_name}",
                tool_result=result,
            )
            state.scratchpad.append(record)
            self._record_step(run_id, record)

            if self.evaluator.should_finish_after_action(result):
                state.final_answer = self.format_final_response(state)
                self._finish_run(run_id, state.final_answer)
                return state

        state.final_answer = self.format_final_response(state, stopped_by_limit=True)
        self._finish_run(run_id, state.final_answer)
        return state

    def observe(self, state: AgentState) -> str:
        return self.observer.observe(state)

    def think(self, state: AgentState, observation: str) -> Decision:
        return self.thinker.think(state, observation)

    def act(self, decision: Decision) -> ToolResult:
        return self.actor.act(decision)

    def format_final_response(self, state: AgentState, stopped_by_limit: bool = False) -> str:
        return self.evaluator.format_final_response(state, stopped_by_limit)

    def _record_step(self, run_id: int | None, record: StepRecord) -> None:
        if self.repository and run_id is not None:
            self.repository.record_step(run_id, record)

    def _finish_run(self, run_id: int | None, final_answer: str) -> None:
        if self.repository and run_id is not None:
            self.repository.finish_run(run_id, final_answer)
