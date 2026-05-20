from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .runtime.runner import run_agent_loop
from .tools import (
    add_to_session_history_tool,
    build_comparison_points_tool,
    extract_session_id_tool,
    get_session_history_tool,
    get_world_state_tool,
    normalize_request_tool,
    update_world_state_tool,
)
from .tools.request_tools import CanonicalRequest

_ERROR_BODIES: dict[str, Any] = {
    "access_denied": lambda svc, op: json.dumps(
        {
            "__type": "AccessDeniedException",
            "message": f"User is not authorized to perform {svc}:{op}",
        }
    ),
    "throttling": lambda svc, op: json.dumps(
        {
            "__type": "ThrottlingException",
            "message": "Rate exceeded",
        }
    ),
    "not_found": lambda svc, op: json.dumps(
        {
            "__type": "ResourceNotFoundException",
            "message": "Requested resource does not exist",
        }
    ),
}


def _try_xml_fallback(canonical: CanonicalRequest) -> str:
    """XML 프로토콜 서비스에서 agent 응답 실패 시 serializer로 최소 유효 응답 생성.

    JSON 에러를 그대로 반환하면 botocore XML 파서가 실패하므로,
    shape_adapter로 필수 필드를 채운 XML 응답을 생성한다.
    """
    try:
        from moto.core.serialize import get_serializer_class
        from moto.core.utils import get_service_model

        svc_model = get_service_model(canonical.service)
        protocol = svc_model.metadata.get("protocol", "json")
        if protocol not in ("query", "rest-xml", "ec2"):
            return ""

        from .runtime.planner import DEFAULT_OUTPUT
        from .shape_adapter import adapt_response_plan
        from .tools import build_response_plan_tool

        # DEFAULT_OUTPUT(error_mode="none")으로 최소 성공 플랜 생성
        response_plan = build_response_plan_tool(canonical, DEFAULT_OUTPUT, {}, "")
        field_values, _ = adapt_response_plan(canonical, response_plan, {})
        op_model = svc_model.operation_model(canonical.operation)
        serializer_cls = get_serializer_class(canonical.service, protocol)
        serializer = serializer_cls(operation_model=op_model)
        result = serializer.serialize(field_values)
        return str(result.get("body", ""))
    except Exception:
        return ""


def handle_aws_request(
    service: Optional[str],
    action: Optional[str],
    url: str,
    headers: dict[str, Any],
    body: Any,
    reason: str = "Unknown",
    source: str = "Unknown",
    moto_native_body: Optional[str] = None,
) -> str:
    started_perf = time.perf_counter()
    started_at = _utc_iso()

    session_id = extract_session_id_tool(headers)
    canonical = normalize_request_tool(service, action, url, headers, body)
    world_state = get_world_state_tool(session_id, headers)
    history_context = get_session_history_tool(session_id)

    run_result = run_agent_loop(
        canonical=canonical,
        world_state=world_state,
        history_context=history_context,
        reason=reason,
        source=source,
        max_attempts=_max_attempts(),
        moto_native_body=moto_native_body,
    )
    agent_output = run_result.agent_output
    planner_meta = run_result.planner_meta
    field_values = run_result.field_values
    response_body = run_result.response_body
    rendered_meta = run_result.rendered_meta

    if not response_body:
        # XML 프로토콜 서비스(STS, EC2, IAM)에 JSON 에러를 반환하면 botocore XML 파서가 실패한다.
        # serializer로 최소 유효 응답을 생성해 CLI 파싱 오류를 방지한다.
        response_body = _try_xml_fallback(canonical) or _ERROR_BODIES.get(
            agent_output.error_mode, _ERROR_BODIES["access_denied"]
        )(canonical.service, canonical.operation)
        rendered_meta = {"assets": []}  # type: ignore[assignment]

    add_to_session_history_tool(
        session_id,
        f"service={canonical.service}, operation={canonical.operation}, source={source}",
        response_body,
    )
    update_world_state_tool(
        session_id, world_state, canonical, agent_output, rendered_meta, field_values
    )

    validation_passed = rendered_meta.get("validation_passed", bool(response_body))
    validation_reason = rendered_meta.get(
        "validation_reason", "serializer" if response_body else "fallback"
    )
    comparison_points = build_comparison_points_tool(
        canonical=canonical,
        rendered_body=response_body,
        validation_passed=bool(validation_passed),
        validation_reason=str(validation_reason),
    )

    finished_at = _utc_iso()
    total_ms = (time.perf_counter() - started_perf) * 1000.0
    _write_audit_record(
        {
            "timestamps": {"started_at": started_at, "finished_at": finished_at},
            "request": {
                "service": service,
                "action": action,
                "url": url,
                "headers": _redact_value(dict(headers)),
                "body": _redact_value(_safe_body(body)),
                "reason": reason,
                "source": source,
                "canonical": {
                    "service": canonical.service,
                    "operation": canonical.operation,
                    "principal_type": canonical.principal_type,
                    "probe_style": canonical.probe_style,
                    "raw_action": canonical.raw_action,
                },
            },
            "decision": {
                "intent_phase": agent_output.intent_phase,
                "response_posture": getattr(agent_output, "response_posture", "normal"),
                "decoy_bundle_id": getattr(agent_output, "decoy_bundle_id", "baseline"),
                "error_mode": agent_output.error_mode,
                "risk_delta": agent_output.risk_delta,
                "reason_tags": agent_output.reason_tags,
                "environment_delta": agent_output.environment_delta,
                "field_values_keys": list(field_values.keys()),
            },
            "response": {
                "body": response_body,
                "assets": rendered_meta.get("assets", []),
                "protocol": rendered_meta.get("protocol", "unknown"),
            },
            "comparison_points": comparison_points,
            "metrics": {
                "total_duration_ms": round(total_ms, 3),
                "llm": _redact_value(planner_meta),
            },
        }
    )

    _log_fallback_stats(canonical, planner_meta, total_ms)

    return response_body


def _log_fallback_stats(
    canonical: CanonicalRequest,
    planner_meta: dict[str, Any],
    total_ms: float,
) -> None:
    usage = planner_meta.get("usage") or {}
    input_tokens = usage.get("input_tokens", "-")
    output_tokens = usage.get("output_tokens", "-")
    total_tokens = "-"
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = str(input_tokens + output_tokens)

    provider = planner_meta.get("provider", "-")
    model = planner_meta.get("model", "-")
    error = planner_meta.get("error")

    is_cache_hit = provider == "response_cache"
    if is_cache_hit:
        lines = [
            "",
            f"[cache-hit]    {canonical.service}:{canonical.operation}",
            f"  time     : {round(total_ms)}ms",
            "",
        ]
    else:
        lines = [
            "",
            f"[moto-fallback] {canonical.service}:{canonical.operation}",
            f"  provider : {provider} ({model})",
        ]
        if error:
            lines.append(f"  error    : {error}")
        else:
            lines.append(
                f"  tokens   : input={input_tokens}  output={output_tokens}  total={total_tokens}"
            )
        lines.extend(
            [
                f"  time     : {round(total_ms)}ms",
                "",
            ]
        )
    print("\n".join(lines), file=sys.stderr, flush=True)  # noqa: T201


def _max_attempts() -> int:
    try:
        return max(1, int(os.getenv("MOTO_LLM_AGENT_MAX_ATTEMPTS", "2")))
    except ValueError:
        return 2


def _write_audit_record(record: dict[str, Any]) -> None:
    path = os.getenv("MOTO_LLM_AUDIT_FILE")
    if not path:
        return
    try:
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            data = loaded if isinstance(loaded, list) else [loaded]
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(record)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _safe_body(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _redact_value(value: Any) -> Any:
    sensitive_keys = {
        "authorization",
        "x-amz-security-token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "access_key",
        "secret_key",
        "session_token",
    }
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            if str(key).lower() in sensitive_keys:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_value(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        result: str = value
        result = re.sub(r"(AWS_ACCESS_KEY_ID=)[^&\s]+", r"\1<redacted>", result)
        result = re.sub(r"(AWS_SECRET_ACCESS_KEY=)[^&\s]+", r"\1<redacted>", result)
        result = re.sub(r"(AWS_SESSION_TOKEN=)[^&\s]+", r"\1<redacted>", result)
        result = re.sub(r"(X-Amz-Security-Token=)[^&\s]+", r"\1<redacted>", result)
        result = re.sub(
            r"(Authorization:\s*)[^\n\r]+",
            r"\1<redacted>",
            result,
            flags=re.IGNORECASE,
        )
        return result
    return value


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
