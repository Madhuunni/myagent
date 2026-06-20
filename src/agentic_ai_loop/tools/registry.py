"""Tool registry and sample tools for the agent loop."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, tool

from agentic_ai_loop.agent.state import ToolResult

from .browser import selenium_title_tool
from .http import http_get_tool, page_text_tool


@dataclass
class ToolRegistry:
    """Small registry that normalizes LangChain tool success and failure responses."""

    tools: dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    @classmethod
    def from_langchain_tools(cls, tools: list[BaseTool]) -> "ToolRegistry":
        registry = cls()
        for selected in tools:
            registry.register(selected)
        return registry

    def run(self, name: str, tool_input: str) -> ToolResult:
        selected = self.tools.get(name)
        if selected is None:
            return ToolResult(name=name, input=tool_input, output="", ok=False, error="unknown_tool")
        try:
            return ToolResult(name=name, input=tool_input, output=str(selected.invoke(tool_input)), ok=True)
        except Exception as exc:  # Tools are isolated failure boundaries, unlike imports.
            return ToolResult(name=name, input=tool_input, output="", ok=False, error=str(exc))


_ALLOWED_OPERATORS: dict[type[ast.operator | ast.unaryop], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


@tool("echo")
def echo_tool(text: str) -> str:
    """Return the exact input text."""

    return text


@tool("calculator")
def calculator_tool(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""

    parsed = ast.parse(expression, mode="eval")
    return str(_eval_expression(parsed.body))


def default_registry() -> ToolRegistry:
    return ToolRegistry.from_langchain_tools(
        [
            echo_tool,
            calculator_tool,
            http_get_tool,
            page_text_tool,
            selenium_title_tool,
        ]
    )


def _eval_expression(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expression(node.left), _eval_expression(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expression(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")
