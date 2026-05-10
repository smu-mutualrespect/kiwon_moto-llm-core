from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..shape_adapter import adapt_response_plan
from ..tools import build_response_plan_tool, validate_rendered_response_tool
from ..tools.render_tools import serialize_response_tool
from ..tools.request_tools import CanonicalRequest
from .planner import DEFAULT_OUTPUT, AgentOutput, build_agent_prompt, parse_agent_output
from .provider import _load_dotenv_if_present, call_gpt_api_with_meta, select_provider
from .tool_executor import execute_agent_tool_requests
from .tool_registry import get_available_tool_names


@dataclass(frozen=True)
class AgentRunResult:
    agent_output: AgentOutput
    response_body: str
    rendered_meta: dict[str, Any]
    field_values: dict[str, Any]
    planner_meta: dict[str, Any]


def run_agent_loop(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    history_context: str,
    reason: str,
    source: str,
    max_attempts: int = 2,
) -> AgentRunResult:
    latest_observation = ""
    available_tools = get_available_tool_names()
    last_planner_meta: dict[str, Any] = {}
    tool_observations: list[str] = []

    for attempt in range(1, max_attempts + 1):
        agent_output, raw_text, planner_meta = _call_agent_once(
            canonical=canonical,
            world_state=world_state,
            history_context=history_context,
            reason=reason,
            source=source,
            latest_observation=latest_observation,
            available_tools=available_tools,
        )
        last_planner_meta = dict(planner_meta)
        last_planner_meta["attempt"] = attempt
        if tool_observations:
            last_planner_meta["tool_calls_executed"] = len(tool_observations)
            last_planner_meta["tool_observations"] = tool_observations

        if agent_output.tool_requests and attempt < max_attempts:
            latest_observation = execute_agent_tool_requests(
                agent_output.tool_requests,
                canonical=canonical,
                world_state=world_state,
                history_context=history_context,
            )
            if latest_observation:
                tool_observations.append(latest_observation)
            continue

        if agent_output.error_mode != "none":
            return AgentRunResult(
                agent_output=agent_output,
                response_body="",
                rendered_meta={
                    "assets": [],
                    "protocol": "error",
                    "validation_passed": True,
                    "validation_reason": "agent_requested_error_mode",
                    "attempts": attempt,
                },
                field_values={},
                planner_meta=last_planner_meta,
            )

        response_plan = build_response_plan_tool(
            canonical, agent_output, world_state, raw_text
        )
        field_values, plan_meta = adapt_response_plan(
            canonical, response_plan, world_state
        )
        response_body, rendered_meta = serialize_response_tool(canonical, field_values)

        if not response_body:
            latest_observation = "serializer returned empty body; return a safer and more explicit response_plan"
            continue

        validation_passed, validation_reason = validate_rendered_response_tool(
            canonical, response_body, world_state
        )
        merged_meta = {
            **plan_meta,
            **rendered_meta,
            "validation_passed": validation_passed,
            "validation_reason": validation_reason,
            "attempts": attempt,
        }
        if validation_passed:
            return AgentRunResult(
                agent_output=agent_output,
                response_body=response_body,
                rendered_meta=merged_meta,
                field_values=field_values,
                planner_meta=last_planner_meta,
            )

        latest_observation = (
            f"attempt={attempt} validation_failed reason={validation_reason}; "
            "correct the response_plan, preserve core members, and reduce complexity"
        )

    return AgentRunResult(
        agent_output=DEFAULT_OUTPUT,
        response_body="",
        rendered_meta={
            "assets": [],
            "protocol": "unknown",
            "validation_passed": False,
            "validation_reason": "agent_loop_exhausted",
            "attempts": max_attempts,
        },
        field_values={},
        planner_meta=last_planner_meta,
    )


def _call_agent_once(
    *,
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    history_context: str,
    reason: str,
    source: str,
    latest_observation: str,
    available_tools: list[str],
) -> tuple[AgentOutput, str, dict[str, Any]]:
    if os.getenv("MOTO_LLM_OFFLINE_STUB", "").strip().lower() in {"1", "true", "yes"}:
        return (
            DEFAULT_OUTPUT,
            "",
            {
                "provider": "offline_stub",
                "model": "deterministic_response_plan",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )

    _load_dotenv_if_present()
    prompt = build_agent_prompt(
        canonical,
        world_state,
        history_context,
        reason,
        source,
        latest_observation=latest_observation,
        available_tools=available_tools,
    )
    try:
        raw, meta = call_gpt_api_with_meta(prompt)
    except Exception:
        return (
            DEFAULT_OUTPUT,
            "",
            {
                "provider": select_provider(),
                "error": "provider_call_failed",
            },
        )
    return parse_agent_output(raw), raw, meta
