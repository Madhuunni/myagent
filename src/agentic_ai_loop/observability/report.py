"""Observability helpers for logs, metrics, and run reports."""

from __future__ import annotations

from prometheus_client import Counter
import structlog

logger = structlog.get_logger(__name__)
prompt_runs_total = Counter("agentic_ai_loop_prompt_runs_total", "Total agent prompt runs")


def record_run_started(user_input: str) -> None:
    prompt_runs_total.inc()
    logger.info("agent_run_started", user_input=user_input)
