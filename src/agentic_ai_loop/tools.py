"""Tool registry and sample tools for the basic agent loop."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass, field

from .core import ToolResult

ToolFn = Callable[[str], str]


@dataclass
class ToolRegistry:
    """Small registry that normalizes tool success and failure responses."""

    tools: dict[str, ToolFn] = field(default_factory=dict)

    def register(self, name: str, tool: ToolFn) -> None:
        self.tools[name] = tool

    @classmethod
    def from_langchain_tools(cls, tools: list[object]) -> "ToolRegistry":
        """Adapt LangChain tools that expose ``name`` and ``invoke`` into this registry."""

        registry = cls()
        for tool in tools:
            registry.register(str(getattr(tool, "name")), lambda text, selected=tool: str(selected.invoke(text)))
        return registry

    def run(self, name: str, tool_input: str) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(name=name, input=tool_input, output="", ok=False, error="unknown_tool")
        try:
            return ToolResult(name=name, input=tool_input, output=tool(tool_input), ok=True)
        except Exception as exc:  # Tools are isolated failure boundaries, unlike imports.
            return ToolResult(name=name, input=tool_input, output="", ok=False, error=str(exc))


_ALLOWED_OPERATORS: dict[type[ast.operator | ast.unaryop], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def echo_tool(text: str) -> str:
    """Return the provided text."""

    return text


def calculator_tool(expression: str) -> str:
    """Safely evaluate a small arithmetic expression."""

    parsed = ast.parse(expression, mode="eval")
    return str(_eval_expression(parsed.body))


def default_registry() -> ToolRegistry:
    """Create the starter tool registry."""

    registry = ToolRegistry()
    registry.register("echo", echo_tool)
    registry.register("calculator", calculator_tool)
    return registry


def _eval_expression(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expression(node.left), _eval_expression(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expression(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")
