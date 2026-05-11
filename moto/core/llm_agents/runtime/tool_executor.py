from __future__ import annotations

import json
from typing import Any

from moto.core.utils import get_service_model

from ..tools.request_tools import CanonicalRequest
from .mock_data import get_mock_template
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
        output = _execute_one(name, args, canonical, world_state, history_context)  # type: ignore[arg-type]
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
            "schema": build_full_schema(canonical)[:2000],
        }
    if name == "state.inspect_consistency":
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
    if name == "mock_data.get_mock_template":
        category = str(args.get("category") or "iam_policy")
        return get_mock_template(category)
    return {
        "error": "unknown_tool",
        "available": [
            "skills.load_skill_document",
            "schema.inspect_output_shape",
            "state.inspect_consistency",
            "mock_data.get_mock_template",
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
