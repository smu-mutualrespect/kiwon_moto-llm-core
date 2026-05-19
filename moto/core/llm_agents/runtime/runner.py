from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional

from ..shape_adapter import adapt_response_plan
from ..tools import build_response_plan_tool, validate_rendered_response_tool
from ..tools.render_tools import serialize_response_tool
from ..tools.request_tools import CanonicalRequest
from ..tools.state_tools import _param_cache_key, _resource_identity_keys, _should_cache_operation
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
    cached = _try_response_cache(canonical, world_state)
    if cached is not None:
        return cached

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
        agent_output = _stabilize_agent_output(canonical, world_state, agent_output)
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


def _stabilize_agent_output(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    agent_output: AgentOutput,
) -> AgentOutput:
    from ..tools.state_tools import _is_delete_operation

    # not_found → none: 세션에서 이미 보여준 resource라면 LLM의 not_found 판단을 덮어씀
    if agent_output.error_mode == "not_found":
        if _has_registered_resource_identity(canonical, world_state):
            return replace(agent_output, error_mode="none", response_posture="sparse")

    # none → not_found: 삭제된 resource에 대한 read 요청은 not_found로 강제
    if agent_output.error_mode == "none" and not _is_delete_operation(canonical):
        if _resource_is_tombstoned(canonical, world_state):
            return replace(agent_output, error_mode="not_found")

    return agent_output


def _has_registered_resource_identity(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
) -> bool:
    registry = world_state.get("resource_registry")
    if not isinstance(registry, dict):
        return False
    for key in _resource_identity_keys(canonical):
        values = registry.get(key)
        if not isinstance(values, dict):
            continue
        if values.get("__deleted__"):
            continue
        if any(isinstance(v, str) and v for k, v in values.items() if k != "__deleted__"):
            return True
    return False


def _resource_is_tombstoned(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
) -> bool:
    registry = world_state.get("resource_registry")
    if not isinstance(registry, dict):
        return False
    for key in _resource_identity_keys(canonical):
        values = registry.get(key)
        if isinstance(values, dict) and values.get("__deleted__"):
            return True
    return False


def _try_response_cache(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
) -> Optional[AgentRunResult]:
    """LLM 호출 전 response_cache 확인 — 히트 시 직렬화만 하고 즉시 반환."""
    if not _should_cache_operation(canonical.operation):
        return None

    # 페이지네이션 연속 요청은 캐시 우회 (다음 페이지 데이터가 다르므로)
    seen_tokens = world_state.get("seen_pagination_tokens", [])
    if any(isinstance(v, str) and v in seen_tokens for v in canonical.request_params.values()):
        return None

    cache_key = _param_cache_key(
        canonical.service, canonical.operation, canonical.target_identifiers
    )
    cached_fields = world_state.get("response_cache", {}).get(cache_key)
    if cached_fields is None:
        return None

    refreshed_fields = _refresh_live_timestamps(deepcopy(cached_fields))
    response_body, rendered_meta = serialize_response_tool(canonical, refreshed_fields)
    if not response_body:
        # 직렬화 실패 시 LLM으로 fallthrough
        return None

    return AgentRunResult(
        agent_output=DEFAULT_OUTPUT,
        response_body=response_body,
        rendered_meta={
            **rendered_meta,
            "validation_passed": True,
            "validation_reason": "response_cache_hit",
            "attempts": 0,
        },
        field_values=refreshed_fields,
        planner_meta={
            "provider": "response_cache",
            "model": "cache",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
    )


# 활동 지표 성격의 타임스탬프 필드명 (소문자 기준)
# 리소스 메타(createdtime, createdate 등)는 일관성 유지를 위해 갱신하지 않음
_LIVE_TIMESTAMP_FIELDS = frozenset({
    "lastpingdatetime",
    "lastheartbeat",
    "lastseen",
    "lastcontact",
    "lastagentcheck",
    "lastreporttime",
    "laststatuscheck",
    "laststatuschange",
    "lasthealthcheck",
    "lastconnection",
    "lastcommunicationtime",
    "lastoperationdate",
})


def _refresh_live_timestamps(data: Any) -> Any:
    """캐시 히트 응답에서 활동 지표 타임스탬프를 현재 시각으로 갱신.

    CreatedTime / CreateDate 등 불변 메타 필드는 건드리지 않아
    동일 리소스를 반복 조회해도 생성 시각이 바뀌지 않는다.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return _refresh_recursive(data, now)


def _refresh_recursive(data: Any, now: str) -> Any:
    if isinstance(data, dict):
        return {
            k: (now if k.lower() in _LIVE_TIMESTAMP_FIELDS and isinstance(v, str) else _refresh_recursive(v, now))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_refresh_recursive(item, now) for item in data]
    return data
