from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse


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
    normalized_service = (service or _service_from_host(url) or "unknown").lower()
    request_params, body_format = _extract_request_params(headers, body)
    raw_action = _extract_action(action, headers, body)
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
        key = prefix.split(".")[-1]
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
