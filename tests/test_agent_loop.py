from agentic_ai_loop.core import AgentLoop, LoopConfig
from agentic_ai_loop.tools import default_registry


def test_calculator_loop_finishes_with_tool_result():
    agent = AgentLoop(tools=default_registry())

    state = agent.run("calculate: 2 + 3 * 4")

    assert state.final_answer == "status: completed\nanswer: 14\niterations: 1"
    assert state.previous_tool_results[0].name == "calculator"


def test_observer_includes_previous_tool_history():
    agent = AgentLoop(tools=default_registry(), config=LoopConfig(finish_when_tool_succeeds=False))

    state = agent.run("hello")

    second_observation = state.scratchpad[1].observation
    assert "echo('hello')" in second_observation
    assert state.final_answer == "A tool result is available; produce final answer."
