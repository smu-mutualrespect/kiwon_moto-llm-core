from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..tools.request_tools import CanonicalRequest
from .skill_loader import load_agent_system_prompt

_ALLOWED_PHASES = {"recon", "privilege_check", "lateral_probe", "impact_probe"}
_ALLOWED_ERROR_MODES = {"none", "access_denied", "throttling", "not_found"}
_ALLOWED_POSTURES = {"sparse", "normal", "rich"}


@dataclass(frozen=True)
class AgentOutput:
    intent_phase: str
    response_posture: str
    decoy_bundle_id: str
    risk_delta: float
    reason_tags: list[str]
    error_mode: str
    field_values: dict[str, Any]
    environment_delta: dict[str, Any]
    tool_requests: list[dict[str, Any]] = field(default_factory=list)


DEFAULT_OUTPUT = AgentOutput(
    intent_phase="recon",
    response_posture="normal",
    decoy_bundle_id="baseline",
    risk_delta=0.1,
    reason_tags=["enum_pattern"],
    error_mode="none",
    field_values={},
    environment_delta={},
)


def build_agent_prompt(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    history_context: str,
    reason: str,
    source: str,
    latest_observation: str = "",
    available_tools: list[str] | None = None,
) -> str:
    return _build_compact_agent_prompt(
        canonical,
        world_state,
        history_context,
        reason,
        source,
        latest_observation,
        available_tools or [],
    )


def _build_compact_agent_prompt(
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    history_context: str,
    reason: str,
    source: str,
    latest_observation: str = "",
    available_tools: list[str] | None = None,
) -> str:
    account_id = str(
        world_state.get("consistency_locks", {}).get("account_id", "123456789012")
    )
    region = str(world_state.get("region", "us-east-1"))
    latest_observation_block = latest_observation or "None"
    tool_block = json.dumps(
        available_tools or [], ensure_ascii=False, separators=(",", ":")
    )
    return (
        f"{load_agent_system_prompt()}\n\n"
        "CURRENT_REQUEST_COMPACT: "
        f"svc={canonical.service} op={canonical.operation} style={canonical.probe_style} "
        f"params={json.dumps(canonical.request_params, default=str, separators=(',', ':'))} "
        f"ids={json.dumps(canonical.target_identifiers, default=str, separators=(',', ':'))} "
        f"acct={account_id} region={region} reason={reason} source={source} "
        f"history={json.dumps(history_context[:800], ensure_ascii=False, separators=(',', ':'))} "
        f"tools={tool_block} LATEST_OBSERVATION={latest_observation_block}. "
        'OUTPUT_JSON_SCHEMA: {"intent_phase":"recon","response_posture":"sparse|normal","error_mode":"none|access_denied|throttling|not_found",'
        '"decoy_bundle_id":"baseline","risk_delta":0.1,"reason_tags":["enum_pattern"],'
        '"tool_requests":[{"tool":"schema.inspect_output_shape"}],'
        '"response_plan":{"mode":"success","posture":"sparse","entity_hints":{"count":1},"field_hints":{},"omit_fields":[]},'
        '"environment_delta":{}}'
    )


def parse_agent_output(raw_text: str) -> AgentOutput:
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        return DEFAULT_OUTPUT

    intent_phase = _coerce_enum(
        parsed.get("intent_phase"), _ALLOWED_PHASES, DEFAULT_OUTPUT.intent_phase
    )
    response_posture = _coerce_enum(
        parsed.get("response_posture"),
        _ALLOWED_POSTURES,
        DEFAULT_OUTPUT.response_posture,
    )
    decoy_bundle_id = _coerce_bundle_id(
        parsed.get("decoy_bundle_id"), DEFAULT_OUTPUT.decoy_bundle_id
    )
    risk_delta = max(
        -0.2,
        min(0.5, _coerce_float(parsed.get("risk_delta"), DEFAULT_OUTPUT.risk_delta)),
    )

    reason_tags = parsed.get("reason_tags")
    if not isinstance(reason_tags, list):
        reason_tags = list(DEFAULT_OUTPUT.reason_tags)
    reason_tags = [str(t) for t in reason_tags][:6]

    error_mode = _coerce_enum(
        parsed.get("error_mode"), _ALLOWED_ERROR_MODES, DEFAULT_OUTPUT.error_mode
    )
    field_values = (
        parsed.get("field_values")
        if isinstance(parsed.get("field_values"), dict)
        else {}
    )
    environment_delta = (
        parsed.get("environment_delta")
        if isinstance(parsed.get("environment_delta"), dict)
        else {}
    )
    _raw_tool_requests = parsed.get("tool_requests")
    tool_requests: list[Any] = (
        _raw_tool_requests if isinstance(_raw_tool_requests, list) else []
    )
    tool_requests = [item for item in tool_requests if isinstance(item, dict)][:3]

    return AgentOutput(
        intent_phase=intent_phase,
        response_posture=response_posture,
        decoy_bundle_id=decoy_bundle_id,
        risk_delta=risk_delta,
        reason_tags=reason_tags,
        error_mode=error_mode,
        field_values=field_values,  # type: ignore[arg-type]
        environment_delta=environment_delta,  # type: ignore[arg-type]
        tool_requests=tool_requests,
    )


def _extract_json(raw_text: str) -> Any:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-z]*\n?", "", candidate).rstrip("`").strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    first = raw_text.find("{")
    last = raw_text.rfind("}")
    if first >= 0 and last > first:
        try:
            return json.loads(raw_text[first : last + 1])
        except Exception:
            pass
    return None


def _coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    s = str(value or "")
    return s if s in allowed else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_bundle_id(value: Any, default: str) -> str:
    s = str(value or "").strip().lower()
    if re.fullmatch(r"[a-z0-9_-]{3,64}", s):
        return s
    return default
