from agentic_ai_loop.agent.observer import Observer
from agentic_ai_loop.agent.think import Thinker
from agentic_ai_loop.agent.act import Actor
from agentic_ai_loop.execution.loop import AgentLoop
from agentic_ai_loop.tools.registry import ToolRegistry, default_registry


def test_components_are_split_into_importable_modules():
    registry = default_registry()

    assert Observer
    assert Thinker
    assert Actor(registry)
    assert AgentLoop(tools=registry)
    assert isinstance(registry, ToolRegistry)
