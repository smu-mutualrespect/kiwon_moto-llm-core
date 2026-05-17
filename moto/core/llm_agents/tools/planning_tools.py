from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .request_tools import CanonicalRequest


@dataclass(frozen=True)
class ResponsePlan:
    mode: str
    posture: str
    entity_hints: dict[str, Any]
    field_hints: dict[str, Any]
    omit_fields: list[str]


def build_response_plan_tool(
    canonical: CanonicalRequest,
    decision: Any,
    world_state: dict[str, Any],
    raw_text: str,
) -> ResponsePlan:
    parsed = _extract_json_object(raw_text)
    if isinstance(parsed, dict):
        candidate = parsed.get("response_plan")
        if isinstance(candidate, dict):
            return _coerce_plan(candidate, canonical, decision, world_state)
        if any(
            key in parsed
            for key in {"mode", "entity_hints", "field_hints", "omit_fields"}
        ):
            return _coerce_plan(parsed, canonical, decision, world_state)
    return _default_plan(canonical, decision, world_state)


def _coerce_plan(
    payload: dict[str, Any],
    canonical: CanonicalRequest,
    decision: Any,
    world_state: dict[str, Any],
) -> ResponsePlan:
    default = _default_plan(canonical, decision, world_state)
    mode = str(payload.get("mode") or default.mode)
    if mode not in {"success", "empty", "error"}:
        mode = default.mode
    posture = str(
        payload.get("posture") or payload.get("response_posture") or default.posture
    )
    if posture not in {"sparse", "normal", "rich"}:
        posture = default.posture
    entity_hints = (
        payload.get("entity_hints")
        if isinstance(payload.get("entity_hints"), dict)
        else dict(default.entity_hints)
    )
    field_hints = (
        payload.get("field_hints")
        if isinstance(payload.get("field_hints"), dict)
        else dict(default.field_hints)
    )
    omit_fields = payload.get("omit_fields")
    if not isinstance(omit_fields, list):
        omit_fields = list(default.omit_fields)
    else:
        omit_fields = [str(item) for item in omit_fields][:20]
    return stabilize_response_plan(
        canonical,
        decision,
        ResponsePlan(mode, posture, entity_hints, field_hints, omit_fields),  # type: ignore[arg-type]
    )


def _default_plan(
    canonical: CanonicalRequest, decision: Any, world_state: dict[str, Any]
) -> ResponsePlan:
    posture = decision.response_posture
    count = {"sparse": 1, "normal": 2, "rich": 3}.get(posture, 2)
    field_hints = dict(canonical.target_identifiers)
    entity_hints: dict[str, Any] = {"count": count}
    if canonical.target_identifiers:
        entity_hints["echo_inputs"] = True
    return stabilize_response_plan(
        canonical,
        decision,
        ResponsePlan("success", posture, entity_hints, field_hints, []),
    )


def stabilize_response_plan(
    canonical: CanonicalRequest, decision: Any, response_plan: ResponsePlan
) -> ResponsePlan:
    mode = response_plan.mode
    posture = response_plan.posture
    omit_fields = list(response_plan.omit_fields)
    entity_hints = dict(response_plan.entity_hints)
    field_hints = dict(response_plan.field_hints)
    if decision.error_mode == "none" and mode == "error":
        mode = "success"
        posture = "sparse"
    protected = _protected_output_members(canonical)
    omit_fields = [field for field in omit_fields if field not in protected]
    if _requires_non_empty_success(canonical):
        if mode in {"empty", "error"} and decision.error_mode == "none":
            mode = "success"
            posture = "sparse"
    if (canonical.service, canonical.operation) == (
        "ssm",
        "DescribeInstanceInformation",
    ):
        requested_count = entity_hints.get("instance_count")
        if not isinstance(requested_count, int) or requested_count < 1:
            entity_hints["instance_count"] = 1 if posture == "sparse" else 2
        instances = field_hints.get("InstanceInformationList")
        if isinstance(instances, list) and not instances:
            field_hints.pop("InstanceInformationList", None)
    if (canonical.service, canonical.operation) == ("ecr", "GetDownloadUrlForLayer"):
        download_url = field_hints.get("downloadUrl")
        if isinstance(download_url, str) and download_url.lower().startswith(
            ("http://", "https://")
        ):
            field_hints["downloadUrl"] = _sanitize_download_url(download_url, canonical)
    return ResponsePlan(mode, posture, entity_hints, field_hints, omit_fields)


def _sanitize_download_url(download_url: str, canonical: CanonicalRequest) -> str:
    digest = (
        canonical.target_identifiers.get("layerDigest")
        or canonical.request_params.get("layerDigest")
        or "sha256:" + "a" * 64
    )
    repo = (
        canonical.target_identifiers.get("repositoryName")
        or canonical.request_params.get("repositoryName")
        or "demo"
    )
    return f"mock://ecr/{repo}/blobs/{digest.replace(':', '/')}"


def _protected_output_members(canonical: CanonicalRequest) -> set[str]:
    members: set[str] = set()
    key = (canonical.service, canonical.operation)
    ecr_repository_members = {
        "repository",
        "repositories",
        "repositoryArn",
        "registryId",
        "repositoryName",
    }
    if key == ("ecr", "DescribeRepositories"):
        members.update(ecr_repository_members)
    elif key == ("ecr", "CreateRepository"):
        members.update(ecr_repository_members)
    elif key == ("ecr", "DeleteRepository"):
        members.update(ecr_repository_members)
    elif key == ("ecr", "InitiateLayerUpload"):
        members.update({"uploadId", "partSize"})
    elif key == ("ecr", "GetDownloadUrlForLayer"):
        members.update({"downloadUrl", "layerDigest"})
    elif key == ("ecr", "CompleteLayerUpload"):
        members.update({"uploadId", "layerDigest", "repositoryName"})
    elif key == ("ecr", "BatchCheckLayerAvailability"):
        members.update({"layers"})
    elif key == ("ssm", "DescribeInstanceInformation"):
        members.update({"InstanceInformationList"})
    elif key == ("ssm", "StartSession"):
        members.update({"SessionId", "TokenValue", "StreamUrl"})
    elif key == ("ecs", "ExecuteCommand"):
        members.update({"session"})
    elif key == ("sts", "DecodeAuthorizationMessage"):
        members.update({"DecodedMessage"})
    elif key == ("secretsmanager", "ValidateResourcePolicy"):
        members.update({"PolicyValidationPassed"})
    elif canonical.operation.startswith("List"):
        members.add("NextToken")
    return members


def _requires_non_empty_success(canonical: CanonicalRequest) -> bool:
    return (canonical.service, canonical.operation) in {
        ("ecr", "BatchCheckLayerAvailability"),
        ("ecr", "GetDownloadUrlForLayer"),
        ("ecr", "InitiateLayerUpload"),
        ("ecr", "CompleteLayerUpload"),
        ("ssm", "DescribeInstanceInformation"),
        ("sts", "DecodeAuthorizationMessage"),
        ("secretsmanager", "ValidateResourcePolicy"),
    }


def _extract_json_object(raw_text: str) -> Any:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").replace("json", "", 1).strip()
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
            return None
    return None
