from agentic_ai_loop.agent.state import AgentState, ToolResult, StepRecord
from agentic_ai_loop.llm import build_llm_planner


class FakeChatModel:
    def __init__(self, content):
        self.content = content
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return type("Message", (), {"content": self.content})()


def test_llm_planner_parses_tool_decision():
    model = FakeChatModel('{"kind":"tool","thought":"Use math.","tool_name":"calculator","tool_input":"2 + 2"}')
    planner = build_llm_planner(model)

    decision = planner(AgentState(user_input="calculate 2 + 2"), "observation")

    assert decision.kind == "tool"
    assert decision.tool_name == "calculator"
    assert decision.tool_input == "2 + 2"
    assert "Available tools" in model.prompt


def test_llm_planner_parses_finish_decision_after_tool_result():
    model = FakeChatModel('{"kind":"finish","thought":"Done.","final_answer":"4"}')
    planner = build_llm_planner(model)
    state = AgentState(user_input="calculate 2 + 2")
    state.scratchpad.append(
        StepRecord(
            iteration=1,
            observation="obs",
            thought="Use math.",
            action="tool:calculator",
            tool_result=ToolResult(name="calculator", input="2 + 2", output="4"),
        )
    )

    decision = planner(state, "observation")

    assert decision.kind == "finish"
    assert decision.final_answer == "4"
    assert "previous tool result is available" in model.prompt
