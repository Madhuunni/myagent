# Agentic AI Loop from Scratch

This repository contains a basic, working Python design for an agentic AI core loop that is compatible with LangChain-style tools and model planners.

## Core decision loop

The architecture is intentionally split into explicit phases:

1. **Observer**: Builds a bounded context window from system prompts, the user request, previous tool calls, and previous tool results.
2. **Think**: Decides whether the agent should call a tool or finish. The default planner is deterministic so the first version runs locally without an API key.
3. **Act**: Executes the selected tool through a registry and records a normalized result.
4. **Repeat**: Returns to Observer until the planner finishes or `max_iterations` is reached. The starter limit is six iterations.
5. **Done**: Stops when a completion criterion is met, such as a successful tool call, an explicit finish decision, or the iteration limit.
6. **Final response**: Formats a stable payload containing status, answer, and iteration count.

## Project layout

```text
src/agentic_ai_loop/core.py   # Agent state, decisions, loop config, and Observer/Think/Act loop
src/agentic_ai_loop/tools.py  # Tool registry plus echo and calculator tools
src/agentic_ai_loop/demo.py   # Runnable demo
tests/test_agent_loop.py      # Basic loop behavior tests
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m agentic_ai_loop.demo
pytest
```

Expected demo output:

```text
status: completed
answer: 14
iterations: 1
```

## How to evolve this design

- Replace the deterministic planner with a LangChain chat model that returns structured `Decision` objects.
- Add a memory policy that summarizes old `StepRecord` entries before the observation exceeds the context budget.
- Add tool routing, retries, timeouts, and permission checks inside `ToolRegistry`.
- Add richer done criteria, such as confidence thresholds, required output schemas, or no-new-information detection.
