"""Configuration for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - optional dependency during minimal installs
    BaseSettings = object  # type: ignore[assignment]
    SettingsConfigDict = dict  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class LoopConfig:
    """Runtime limits and completion criteria."""

    max_iterations: int = 6
    max_observation_chars: int = 4_000
    finish_when_tool_succeeds: bool = True


class AgentSettings(BaseSettings):
    """Environment-backed settings for optional integrations."""

    if hasattr(BaseSettings, "model_config"):
        model_config = SettingsConfigDict(env_prefix="AGENTIC_AI_LOOP_")

    ollama_model: str = "llama3.1"
    database_url: str = "sqlite:///agentic_ai_loop.db"
    langsmith_tracing: bool = False
