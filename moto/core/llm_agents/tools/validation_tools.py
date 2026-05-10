from __future__ import annotations

import json
import re
from typing import Any
from xml.etree import ElementTree

from moto.core.utils import get_service_model

from .request_tools import CanonicalRequest

_SAFE_DENY_PATTERNS = [
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]
_SAFE_XML_NAMESPACE_PATTERNS = [
    re.compile(r'xmlns="https?://[^"]+amazonaws\.com/[^"]*"', re.IGNORECASE),
    re.compile(r"xmlns='https?://[^']+amazonaws\.com/[^']*'", re.IGNORECASE),
]

# botocore required_members 외에 허니팟 응답에서 반드시 있어야 하는 핵심 필드
# XML 프로토콜 서비스(ec2/sts/iam)는 태그명 문자열 포함 여부로 검사
_HONEYPOT_CORE_MEMBERS: dict[tuple[str, str], list[str]] = {
    ("ecr", "InitiateLayerUpload"): ["uploadId", "partSize"],
    ("ecr", "GetDownloadUrlForLayer"): ["downloadUrl", "layerDigest"],
    ("ecr", "CompleteLayerUpload"): ["uploadId", "layerDigest", "repositoryName"],
    ("ecr", "BatchCheckLayerAvailability"): ["layers"],
    ("ssm", "DescribeInstanceInformation"): ["InstanceInformationList"],
    ("sts", "DecodeAuthorizationMessage"): ["DecodedMessage"],
    ("secretsmanager", "ValidateResourcePolicy"): ["PolicyValidationPassed"],
    ("iam", "GetContextKeysForPrincipalPolicy"): ["ContextKeyNames"],
    ("iam", "ListServiceSpecificCredentials"): ["ServiceSpecificCredentials"],
    ("iam", "GenerateServiceLastAccessedDetails"): ["JobId"],
    ("bedrock", "ListFoundationModels"): ["modelSummaries"],
}

# 내용과 무관하게 항상 Agent가 응답해야 하는 고위험·고상호작용 명령
_HIGH_INTERACTION_OPERATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ssm", "StartSession"),
        ("ssm", "ResumeSession"),
        ("ssm", "TerminateSession"),
        ("ecs", "ExecuteCommand"),
    }
)


def validate_rendered_response_tool(
    canonical: CanonicalRequest,
    rendered_body: str,
    world_state: dict[str, Any],
) -> tuple[bool, str]:
    """Validate an LLM-generated response: safety + botocore shape + world-state."""
    is_safe, safety_reason = _validate_safety(rendered_body)
    if not is_safe:
        return False, safety_reason

    shape_ok, shape_reason = _validate_against_shape(
        canonical.service, canonical.operation, rendered_body, check_empty=True
    )
    if not shape_ok:
        return False, shape_reason

    return _validate_world_state_consistency(rendered_body, world_state)


def validate_moto_native_response(
    service: str,
    operation: str,
    body: Any,
    status_code: int,
) -> bool:
    """Check a Moto-native response. Returns True if the agent should be called instead."""
    # high-interaction 명령은 내용과 무관하게 항상 Agent가 응답
    if (service, operation) in _HIGH_INTERACTION_OPERATIONS:
        return True

    if status_code >= 400:
        return False
    body_str = _to_str(body)
    if not body_str.strip():
        return False

    is_safe, _ = _validate_safety(body_str)
    if not is_safe:
        return True

    ok, _ = _validate_against_shape(service, operation, body_str, check_empty=False)
    return not ok


def build_comparison_points_tool(
    canonical: CanonicalRequest,
    rendered_body: str,
    validation_passed: bool,
    validation_reason: str,
) -> dict[str, Any]:
    protocol_family = (
        "xml" if canonical.service in {"ec2", "iam", "sts", "s3"} else "json"
    )
    stripped = rendered_body.lstrip()
    detected_format = (
        "xml"
        if stripped.startswith("<")
        else "json"
        if stripped.startswith("{") or stripped.startswith("[")
        else "text"
    )
    parseability = _parseability(detected_format, rendered_body)
    return {
        "protocol_family_expected": protocol_family,
        "response_format_detected": detected_format,
        "format_match": protocol_family == detected_format,
        "response_parseable": parseability["parseable"],
        "parse_error": parseability["parse_error"],
        "xml_namespace_present": parseability["xml_namespace_present"],
        "xml_root_tag": parseability["xml_root_tag"],
        "json_top_level_type": parseability["json_top_level_type"],
        "validator_passed": validation_passed,
        "validator_reason": validation_reason,
        "fallback_triggered": not validation_passed,
    }


def _validate_against_shape(
    service: str,
    operation: str,
    rendered_body: str,
    *,
    check_empty: bool,
) -> tuple[bool, str]:
    """Universal validation using botocore output shape for any service/operation."""
    try:
        service_model = get_service_model(service)
        operation_model = service_model.operation_model(operation)
        output_shape = operation_model.output_shape
        protocol = service_model.metadata.get("protocol", "json")
    except Exception:
        return True, "ok"

    is_xml_protocol = protocol in ("query", "rest-xml", "ec2")
    stripped = rendered_body.lstrip()
    is_xml_body = stripped.startswith("<")
    is_json_body = stripped.startswith("{") or stripped.startswith("[")

    if is_xml_protocol and not is_xml_body:
        return False, f"protocol '{protocol}' expects XML but got non-XML"
    if not is_xml_protocol and output_shape is not None and not is_json_body:
        return False, f"protocol '{protocol}' expects JSON but got non-JSON"

    if output_shape is None:
        return True, "ok"

    required = list(getattr(output_shape, "required_members", []) or [])

    if is_xml_protocol:
        try:
            ElementTree.fromstring(rendered_body)
        except Exception as exc:
            return False, f"XML parse failed: {exc}"
        for member in required:
            if member not in rendered_body:
                return False, f"XML missing required member: {member}"
    else:
        try:
            payload = json.loads(rendered_body)
        except json.JSONDecodeError as exc:
            return False, f"JSON parse failed: {exc}"
        if not isinstance(payload, dict):
            return False, "response must be a JSON object"
        if check_empty and rendered_body.strip() == "{}":
            top_members = list(output_shape.members.keys())
            if top_members:
                return (
                    False,
                    f"empty object response; expected members: {top_members[:3]}",
                )
        for member in required:
            if member not in payload:
                return False, f"missing required member: {member}"
        # botocore required_members에 없더라도 허니팟 필수 핵심 필드 확인
        for member in _HONEYPOT_CORE_MEMBERS.get((service, operation), []):
            if member not in payload:
                return False, f"missing honeypot core member: {member}"
        # 범용 output shape 멤버 검증: 모든 서비스에 자동 적용
        if output_shape.members:
            valid_keys = set(output_shape.members.keys())
            response_keys = set(payload.keys())
            if response_keys and not (response_keys & valid_keys):
                return False, (
                    f"response keys {sorted(response_keys)} don't match "
                    f"any AWS schema members; expected one of: {sorted(valid_keys)[:3]}"
                )
            if check_empty and response_keys:
                unknown = response_keys - valid_keys
                if unknown:
                    return (
                        False,
                        f"response contains fields not in AWS schema: {sorted(unknown)}",
                    )
        # LLM 응답 경로에서만 값 형식(value format) 검사
        if check_empty:
            ok, reason = _validate_value_formats(payload)
            if not ok:
                return False, reason

    if is_xml_protocol:
        # XML 프로토콜 서비스의 허니팟 핵심 필드를 태그 시작 패턴으로 검사
        # (부분 문자열 오탐 방지: "ServiceSpecificCredentials" in "<ListServiceSpecificCredentials>" 차단)
        for member in _HONEYPOT_CORE_MEMBERS.get((service, operation), []):
            if f"<{member}" not in rendered_body and f"<{member}>" not in rendered_body:
                return False, f"XML missing honeypot core member: {member}"
        # 범용 XML output shape 멤버 검증: body 위치 필드 중 하나라도 있어야 함
        if check_empty and output_shape.members:
            valid_tags = _get_xml_body_wire_names(output_shape)
            if valid_tags and not any(f"<{t}" in rendered_body for t in valid_tags):
                return False, (
                    f"XML response contains no valid AWS body members; "
                    f"expected one of: {sorted(valid_tags)[:3]}"
                )

    return True, "ok"


def _get_xml_body_wire_names(output_shape: Any) -> set[str]:
    """Collect XML body element names from botocore output shape (excludes headers/uri)."""
    _skip_locations = {"header", "headers", "uri", "querystring", "statusCode"}
    names: set[str] = set()
    for member_name, member_shape in output_shape.members.items():
        loc = member_shape.serialization.get("location", "body")
        if loc in _skip_locations:
            continue
        wire_name = member_shape.serialization.get("name", member_name)
        names.add(wire_name)
        if wire_name != member_name:
            names.add(member_name)  # SDK 이름도 허용 (fallback)
    return names


_INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_VOLUME_ID_RE = re.compile(r"^vol-[0-9a-f]{8,17}$")
_ACCOUNT_ID_RE = re.compile(r"^\d{12}$")
_ARN_RE = re.compile(r"^arn:aws[a-z-]*:[a-z0-9-]+:[a-z0-9-]*:\d{12}:")

# 필드명 → 검증 규칙 (소문자 키 기준)
_FIELD_FORMAT_RULES: dict[str, tuple[re.Pattern[str], str]] = {
    "instanceid": (_INSTANCE_ID_RE, "invalid InstanceId (expected i-[0-9a-f]{8,17})"),
    "volumeid": (_VOLUME_ID_RE, "invalid VolumeId (expected vol-[0-9a-f]{8,17})"),
}
_ARN_SUFFIX = "arn"  # 필드명이 'arn'으로 끝나면 ARN 형식 검사
_ACCOUNT_FIELDS = {"ownerid", "accountid", "awsaccountid"}


def _validate_value_formats(payload: dict[str, Any]) -> tuple[bool, str]:
    """JSON 응답의 핵심 필드 값이 실제 AWS 형식과 일치하는지 검사."""
    for key, value in _iter_leaf_strings(payload):
        lowered = key.lower()
        # 필드명 직접 매칭
        rule = _FIELD_FORMAT_RULES.get(lowered)
        if rule and value:
            pattern, msg = rule
            if not pattern.match(value):
                return False, f"{key}: {msg}, got '{value}'"
        # ARN 필드 검사
        if lowered.endswith(_ARN_SUFFIX) and value and value.startswith("arn:"):
            if not _ARN_RE.match(value):
                return False, f"{key}: invalid ARN format, got '{value}'"
        # Account ID 필드 검사
        if lowered in _ACCOUNT_FIELDS and value:
            if not _ACCOUNT_ID_RE.match(value):
                return (
                    False,
                    f"{key}: invalid account ID (expected 12 digits), got '{value}'",
                )
    return True, "ok"


def _iter_leaf_strings(obj: Any, _depth: int = 0) -> list[tuple[str, str]]:
    """dict/list 재귀 순회하며 (key, str_value) 쌍 반환 (최대 깊이 4)."""
    results: list[tuple[str, str]] = []
    if _depth > 4:
        return results
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                results.append((k, v))
            else:
                results.extend(_iter_leaf_strings(v, _depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_iter_leaf_strings(item, _depth + 1))
    return results


def _validate_safety(rendered_body: str) -> tuple[bool, str]:
    for pattern in _SAFE_DENY_PATTERNS:
        if pattern.search(rendered_body):
            if pattern.pattern == r"https?://" and _is_allowed_xml_namespace(
                rendered_body
            ):
                continue
            return False, f"safety pattern denied: {pattern.pattern}"
    return True, "ok"


def _is_allowed_xml_namespace(rendered_body: str) -> bool:
    stripped = rendered_body.lstrip()
    return stripped.startswith("<") and any(
        p.search(rendered_body) for p in _SAFE_XML_NAMESPACE_PATTERNS
    )


def _validate_world_state_consistency(
    rendered_body: str,
    world_state: dict[str, Any],
) -> tuple[bool, str]:
    account_id = str(world_state.get("consistency_locks", {}).get("account_id", ""))
    if "arn:aws:" in rendered_body and account_id and account_id not in rendered_body:
        return False, "world-state validation failed: account_id lock mismatch"
    return True, "ok"


def _to_str(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body) if body is not None else ""


def _parseability(body_format: str, rendered_body: str) -> dict[str, Any]:
    if body_format == "xml":
        try:
            root = ElementTree.fromstring(rendered_body)
            return {
                "parseable": True,
                "parse_error": "",
                "xml_namespace_present": root.tag.startswith("{"),
                "xml_root_tag": root.tag.split("}", 1)[-1],
                "json_top_level_type": "",
            }
        except Exception as exc:
            return {
                "parseable": False,
                "parse_error": str(exc),
                "xml_namespace_present": False,
                "xml_root_tag": "",
                "json_top_level_type": "",
            }
    if body_format == "json":
        try:
            parsed = json.loads(rendered_body)
            return {
                "parseable": True,
                "parse_error": "",
                "xml_namespace_present": False,
                "xml_root_tag": "",
                "json_top_level_type": type(parsed).__name__,
            }
        except Exception as exc:
            return {
                "parseable": False,
                "parse_error": str(exc),
                "xml_namespace_present": False,
                "xml_root_tag": "",
                "json_top_level_type": "",
            }
    return {
        "parseable": False,
        "parse_error": "unsupported response format",
        "xml_namespace_present": False,
        "xml_root_tag": "",
        "json_top_level_type": "",
    }
