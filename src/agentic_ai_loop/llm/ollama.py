"""Local LLM support through Ollama/LangChain."""

from __future__ import annotations


def build_ollama_chat_model(model: str = "llama3.1"):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model)
