"""Local LLM support through Ollama/LangChain."""

from __future__ import annotations

import json
from typing import Any

from agentic_ai_loop.agent.state import AgentState, Decision
from agentic_ai_loop.agent.think import Planner


def build_ollama_chat_model(model: str = "llama3.1"):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model)


def build_llm_planner(chat_model: Any) -> Planner:
    """Build a planner that asks a chat model for structured loop decisions."""

    def planner(state: AgentState, observation: str) -> Decision:
        prompt = _format_planner_prompt(state, observation)
        response = chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        return _decision_from_model_content(str(content))

    return planner


def build_ollama_planner(model: str = "llama3.1") -> Planner:
    """Build an Ollama-backed planner for the agent loop."""

    return build_llm_planner(build_ollama_chat_model(model))


def _format_planner_prompt(state: AgentState, observation: str) -> str:
    history_instruction = (
        "If a previous tool result is available, finish with a concise final_answer based on that result."
        if state.previous_tool_results
        else "If a tool can help, choose one tool call."
    )
    return f"""You are the Think step in an Observer-Think-Act agent loop.
Return only a JSON object with this schema:
{{"kind":"tool","thought":"...","tool_name":"calculator|http_get|page_text|browser_title|echo","tool_input":"..."}}
or:
{{"kind":"finish","thought":"...","final_answer":"..."}}

Available tools:
- calculator: evaluate arithmetic expressions.
- http_get: fetch a URL.
- page_text: extract readable page text from a URL.
- browser_title: read a page title from a URL.
- echo: return the input unchanged.

Guidance: {history_instruction}

Observation:
{observation}
"""


def _decision_from_model_content(content: str) -> Decision:
    data = _extract_json_object(content)
    kind = data.get("kind")
    thought = str(data.get("thought") or "LLM planner decision.")
    if kind == "tool":
        return Decision(
            kind="tool",
            thought=thought,
            tool_name=str(data.get("tool_name") or "echo"),
            tool_input=str(data.get("tool_input") or ""),
        )
    return Decision(kind="finish", thought=thought, final_answer=str(data.get("final_answer") or content).strip())


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"kind": "finish", "thought": "The model returned plain text.", "final_answer": content}
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {"kind": "finish", "thought": "The model returned non-JSON text.", "final_answer": content}
    return parsed if isinstance(parsed, dict) else {"kind": "finish", "final_answer": content}
