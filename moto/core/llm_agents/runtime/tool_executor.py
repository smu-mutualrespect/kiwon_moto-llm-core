from __future__ import annotations

import json
from typing import Any

from moto.core.utils import get_service_model

from ..tools.request_tools import CanonicalRequest
from .schema import build_full_schema
from .skill_loader import load_skill_documents


def execute_agent_tool_requests(
    tool_requests: list[dict[str, Any]],
    *,
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    history_context: str,
) -> str:
    observations: list[str] = []
    for request in tool_requests[:3]:
        if not isinstance(request, dict):
            continue
        name = str(request.get("tool") or request.get("name") or "")
        args = request.get("args") if isinstance(request.get("args"), dict) else {}
        output = _execute_one(name, args, canonical, world_state, history_context)
        observations.append(
            json.dumps(
                {"tool": name, "output": output},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return (
        "TOOL_OBSERVATIONS=" + "[" + ",".join(observations) + "]"
        if observations
        else ""
    )


def _execute_one(
    name: str,
    args: dict[str, Any],
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    history_context: str,
) -> dict[str, Any]:
    if name == "skills.load_skill_document":
        skill_name = str(args.get("skill") or _default_skill_for(canonical))
        skills = load_skill_documents()
        return {"skill": skill_name, "document": skills.get(skill_name, "")[:900]}
    if name == "schema.inspect_output_shape":
        return {
            "service": canonical.service,
            "operation": canonical.operation,
            "shape": build_full_schema(canonical)[:1600],
        }
    if name == "aws_cli.inspect_reference_output":
        return _inspect_reference_output(canonical)
    if name == "runtime.summarize_request_context":
        return {
            "service": canonical.service,
            "operation": canonical.operation,
            "probe_style": canonical.probe_style,
            "request_params": canonical.request_params,
            "target_identifiers": canonical.target_identifiers,
            "region": world_state.get("region", "us-east-1"),
            "history": history_context[:400],
        }
    if name == "state.inspect_consistency":
        return _inspect_consistency(canonical, world_state)
    if name == "latency.estimate_budget":
        return _estimate_latency_budget(args)
    if name == "validator.explain_last_failure":
        return _explain_validation_failure(args)
    return {
        "error": "unknown_tool",
        "available": [
            "skills.load_skill_document",
            "schema.inspect_output_shape",
            "aws_cli.inspect_reference_output",
            "runtime.summarize_request_context",
            "state.inspect_consistency",
            "latency.estimate_budget",
            "validator.explain_last_failure",
        ],
    }


def _default_skill_for(canonical: CanonicalRequest) -> str:
    if canonical.operation.lower().startswith(("list", "describe", "get")):
        return "recon_skill"
    if canonical.operation.lower().startswith(
        ("create", "modify", "monitor", "unmonitor", "purchase", "initiate", "complete")
    ):
        return "write_action_skill"
    return "protocol_repair_skill"


def _inspect_reference_output(canonical: CanonicalRequest) -> dict[str, Any]:
    service = _cli_service_name(canonical.service)
    command = _cli_command_name(canonical.operation)
    url = f"https://docs.aws.amazon.com/cli/latest/reference/{service}/{command}.html"
    try:
        service_model = get_service_model(canonical.service)
        operation_model = service_model.operation_model(canonical.operation)
        output_shape = operation_model.output_shape
        protocol = service_model.metadata.get("protocol", "unknown")
        members = sorted(output_shape.members) if output_shape is not None else []
        return {
            "url": url,
            "service": canonical.service,
            "operation": canonical.operation,
            "protocol": protocol,
            "botocore_output_shape": getattr(output_shape, "name", "")
            if output_shape is not None
            else "",
            "top_level_members": members[:30],
            "guidance": "final response must be rendered by runtime serializer, not written directly by the agent",
        }
    except Exception as exc:
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}


def _inspect_consistency(
    canonical: CanonicalRequest, world_state: dict[str, Any]
) -> dict[str, Any]:
    locks = (
        world_state.get("consistency_locks", {})
        if isinstance(world_state, dict)
        else {}
    )
    return {
        "account_id": str(locks.get("account_id", "123456789012")),
        "region": str(world_state.get("region", "us-east-1")),
        "phase": str(world_state.get("phase", "recon")),
        "risk_score": world_state.get("risk_score", 0.2),
        "recent_actions": list(world_state.get("last_actions", []))[-5:],
        "request_identifiers": canonical.target_identifiers,
        "guidance": "reuse request identifiers when safe; keep ARN account and region aligned with locks",
    }


def _estimate_latency_budget(args: dict[str, Any]) -> dict[str, Any]:
    target_ms = _coerce_int(args.get("target_ms"), 3000)
    current_attempt = _coerce_int(args.get("attempt"), 1)
    requested_tools = _coerce_int(args.get("requested_tools"), 1)
    should_call_more = (
        target_ms >= 4000 and current_attempt == 1 and requested_tools <= 1
    )
    return {
        "target_ms": target_ms,
        "current_attempt": current_attempt,
        "requested_tools": requested_tools,
        "should_call_more_tools": should_call_more,
        "guidance": "prefer direct sparse response_plan for stable AWS shapes; avoid extra tool rounds when target is under 4s",
    }


def _explain_validation_failure(args: dict[str, Any]) -> dict[str, Any]:
    reason = str(args.get("reason") or args.get("validation_reason") or "").strip()
    if not reason:
        return {
            "failure_type": "none",
            "guidance": "no validation failure was provided",
        }
    lowered = reason.lower()
    if "safety pattern" in lowered:
        failure_type = "safety"
        guidance = "remove public URLs, credentials, private keys, and real account-looking secrets"
    elif "protocol" in lowered or "parse" in lowered:
        failure_type = "protocol"
        guidance = "choose sparse response_plan and preserve required AWS protocol wrapper members"
    elif "account_id" in lowered or "world-state" in lowered:
        failure_type = "consistency"
        guidance = "align ARNs and IDs with the locked fake account and region"
    else:
        failure_type = "shape"
        guidance = (
            "reduce optional fields and let the runtime fill botocore-shaped members"
        )
    return {"failure_type": failure_type, "reason": reason[:300], "guidance": guidance}


def _cli_service_name(service: str) -> str:
    aliases = {
        "resource-explorer-2": "resource-explorer-2",
        "codeguru-reviewer": "codeguru-reviewer",
    }
    return aliases.get(service, service)


def _cli_command_name(operation: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(operation):
        if char.isupper() and index > 0:
            chars.append("-")
        chars.append(char.lower())
    return "".join(chars)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
