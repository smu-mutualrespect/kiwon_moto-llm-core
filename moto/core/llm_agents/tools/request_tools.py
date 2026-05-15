from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# SigV4 signing service 이름과 botocore service id가 다른 경우 fallback normalizer에서 보정한다.
_SIGNING_SERVICE_ALIASES = {
    # access-analyzer는 signing name이고 accessanalyzer는 botocore service id다.
    "access-analyzer": "accessanalyzer",
}

# X-Amz-Target prefix만 보고 service를 복구해야 하는 JSON RPC 계열 service 목록이다.
_TARGET_PREFIX_SERVICES = {
    # FraudDetector의 JSON RPC target prefix다.
    "AWSHawksNestServiceFacade": "frauddetector",
    # Resource Explorer v2의 JSON RPC target prefix다.
    "AWSResourceExplorer": "resource-explorer-2",
    # Backup Gateway의 JSON RPC target prefix다.
    "BackupOnPremises_v20210101": "backup-gateway",
}


@dataclass(frozen=True)
class CanonicalRequest:
    service: str
    operation: str
    principal_type: str
    probe_style: str
    raw_action: str
    request_params: dict[str, Any]
    target_identifiers: dict[str, str]
    body_format: str


def normalize_request_tool(
    service: Optional[str],
    action: Optional[str],
    url: str,
    headers: dict[str, Any],
    body: Any,
) -> CanonicalRequest:
    # 호출자가 service를 이미 넘겼으면 그 값을 최우선으로 쓴다.
    normalized_service = (
        service
        # service가 없으면 X-Amz-Target prefix에서 먼저 복구한다.
        or _service_from_target(headers)
        # target으로 안 되면 SigV4 Authorization credential scope에서 복구한다.
        or _service_from_authorization(headers)
        # 그래도 안 되면 URL host의 첫 component에서 복구한다.
        or _service_from_host(url)
        # 모든 추론이 실패하면 audit/debug가 가능하도록 unknown으로 남긴다.
        or "unknown"
    ).lower()
    # body/query/json을 통합 request parameter dict로 정규화한다.
    request_params, body_format = _extract_request_params(headers, body)
    # 명시 action, X-Amz-Target, body Action= 순서로 raw action을 뽑는다.
    raw_action = _extract_action(action, headers, body)
    # raw action을 botocore operation 이름 형태로 정규화한다.
    operation = _canonical_operation(raw_action)

    probe_style = "enumeration"
    lower_operation = operation.lower()
    if lower_operation.startswith(("get", "describe", "list")):
        probe_style = "enumeration"
    elif lower_operation.startswith(("assume", "run", "start")):
        probe_style = "execution"

    principal_type = "unknown"
    auth_header = str(headers.get("Authorization", ""))
    if "AKIA" in auth_header or "Credential=" in auth_header:
        principal_type = "iam_user_or_role"

    target_identifiers = _extract_target_identifiers(request_params)

    return CanonicalRequest(
        service=normalized_service,
        operation=operation,
        principal_type=principal_type,
        probe_style=probe_style,
        raw_action=raw_action,
        request_params=request_params,
        target_identifiers=target_identifiers,
        body_format=body_format,
    )


def _extract_action(action: Optional[str], headers: dict[str, Any], body: Any) -> str:
    if action:
        return str(action)
    target = headers.get("X-Amz-Target") or headers.get("x-amz-target")
    if target:
        return str(target).split(".")[-1]
    body_text = (
        body.decode("utf-8", errors="ignore")
        if isinstance(body, bytes)
        else str(body or "")
    )
    if "Action=" in body_text:
        params = parse_qs(body_text, keep_blank_values=True)
        values = params.get("Action")
        if values and values[0]:
            return values[0]
    return "UnknownAction"


def _extract_request_params(
    headers: dict[str, Any], body: Any
) -> tuple[dict[str, Any], str]:
    body_text = (
        body.decode("utf-8", errors="ignore")
        if isinstance(body, bytes)
        else str(body or "")
    )
    stripped = body_text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except Exception:
            return {}, "json-invalid"
        if isinstance(parsed, dict):
            return parsed, "json"
        return {"_root": parsed}, "json"
    if "=" in body_text and "&" in body_text or body_text.startswith("Action="):
        params = parse_qs(body_text, keep_blank_values=True)
        normalized: dict[str, Any] = {}
        for key, values in params.items():
            normalized[key] = values[0] if len(values) == 1 else values
        return normalized, "query"
    target = headers.get("X-Amz-Target") or headers.get("x-amz-target")
    if target and stripped:
        try:
            parsed = json.loads(stripped)
        except Exception:
            return {}, "target-text"
        if isinstance(parsed, dict):
            return parsed, "target-json"
    return {}, "text"


def _extract_target_identifiers(request_params: dict[str, Any]) -> dict[str, str]:
    identifiers: dict[str, str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(key if not prefix else f"{prefix}.{key}", nested)
            return
        if isinstance(value, list):
            if value:
                visit(prefix, value[0])
            return
        parts = prefix.split(".")
        key = parts[-1]
        # EC2 쿼리 프로토콜: VolumeId.1, InstanceId.1 등 끝이 숫자인 indexed param 처리
        # 마지막 컴포넌트가 순수 숫자면 바로 앞 컴포넌트를 실제 키로 사용한다.
        if key.isdigit() and len(parts) >= 2:
            key = parts[-2]
        if not key:
            return
        lowered = key.lower()
        if any(
            token in lowered
            for token in ("arn", "name", "id", "digest", "secret", "repository", "user")
        ):
            identifiers[key] = str(value)
            if lowered.endswith("s") and len(key) > 1:
                identifiers.setdefault(key[:-1], str(value))
            if lowered.endswith("arns") or lowered.endswith("digests"):
                identifiers.setdefault(key[:-1], str(value))

    visit("", request_params)
    return identifiers


def _canonical_operation(raw_action: str) -> str:
    action = re.sub(
        r"[^A-Za-z0-9]+", " ", raw_action.split(":")[-1].split(".")[-1].strip()
    )
    parts = [p for p in action.split() if p]
    if not parts:
        return "UnknownAction"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _service_from_host(url: str) -> Optional[str]:
    host = urlparse(url).netloc.lower()
    if not host:
        return None
    parts = host.split(".")
    first = parts[0]
    if first in {"api", "data", "control", "runtime"} and len(parts) > 1:
        return parts[1]
    if first in {"ec2", "ssm", "iam", "sts", "s3", "ecr", "secretsmanager"}:
        return first
    return None


def _service_from_authorization(headers: dict[str, Any]) -> Optional[str]:
    # 헤더 키 대소문자 차이를 흡수해서 Authorization 값을 읽는다.
    auth = str(headers.get("Authorization") or headers.get("authorization") or "")
    # SigV4 Credential scope의 service segment를 추출한다.
    match = re.search(r"Credential=[^,/]+/\d{8}/[^/]+/([^/]+)/aws4_request", auth)
    # SigV4 형식이 아니면 service를 알 수 없으므로 None을 반환한다.
    if not match:
        return None
    # 추출한 signing service를 lower-case로 통일한다.
    service = match.group(1).lower()
    # signing service와 botocore service id가 다르면 alias로 바꾼다.
    return _SIGNING_SERVICE_ALIASES.get(service, service)


def _service_from_target(headers: dict[str, Any]) -> Optional[str]:
    # JSON RPC 요청의 X-Amz-Target 헤더를 대소문자 차이까지 고려해 읽는다.
    target = headers.get("X-Amz-Target") or headers.get("x-amz-target")
    # Prefix.Operation 형태가 아니면 이 방법으로 service를 복구할 수 없다.
    if not target or "." not in str(target):
        return None
    # 점 앞 prefix가 service family를 나타낸다.
    prefix = str(target).split(".", 1)[0]
    # 알려진 prefix면 botocore service id를 반환하고, 아니면 None을 반환한다.
    return _TARGET_PREFIX_SERVICES.get(prefix)
