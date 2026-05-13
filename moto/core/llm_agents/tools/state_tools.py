from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .request_tools import CanonicalRequest

_session_storage: dict[str, list[dict[str, str]]] = {}
_session_state: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()


def has_cached_agent_response_tool(
    session_id: str, service: str, operation: str
) -> bool:
    """에이전트가 이 operation을 이전에 응답한 적 있는지 확인 (agent_responses 기반)."""
    with _lock:
        state = _session_state.get(session_id, {})
    return f"{service}:{operation}" in state.get("agent_responses", [])


def get_session_history_tool(session_id: str) -> str:
    with _lock:
        history = list(_session_storage.get(session_id, []))
    if not history:
        return "No previous interactions in this session."
    formatted: list[str] = []
    for idx, item in enumerate(history, start=1):
        formatted.append(f"Request {idx}: {item['request']}")
        formatted.append(f"Response {idx}: {item['response'][:280]}")
    return "\n".join(formatted)


def add_to_session_history_tool(
    session_id: str, request_info: str, response: str
) -> None:
    with _lock:
        _session_storage.setdefault(session_id, []).append(
            {"request": request_info, "response": response}
        )
        if len(_session_storage[session_id]) > 8:
            _session_storage[session_id].pop(0)


def extract_session_id_tool(headers: dict[str, Any]) -> str:
    # Derive session from the credential (access key) in the SigV4 Authorization header.
    # This keeps the same attacker in one session regardless of IP/proxy changes,
    # and separates different attackers that share an egress IP.
    auth = str(headers.get("Authorization") or headers.get("authorization") or "")
    if auth:
        match = re.search(r"Credential=([A-Z0-9]+)/", auth)
        if match:
            return match.group(1)
    return str(
        headers.get("X-Forwarded-For")
        or headers.get("x-forwarded-for")
        or headers.get("X-Amzn-Trace-Id")
        or headers.get("x-amzn-trace-id")
        or "default_session"
    )


def get_world_state_tool(session_id: str, headers: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        if session_id not in _session_state:
            _session_state[session_id] = {
                "session_id": session_id,
                "persona": "mid-size-prod-account",
                "region": headers.get("X-Amz-Region", "us-east-1"),
                "phase": "recon",
                "exposed_assets": [],
                "exposed_roles": ["ReadOnlyOpsRole"],
                "credibility_level": "medium",
                "risk_score": 0.2,
                "last_actions": [],
                "consistency_locks": {
                    "account_id": _derive_account_id(session_id),
                    "os_family": "Amazon Linux 2",
                },
                "known_names": {},
                "agent_responses": [],
                "response_cache": {},
                "seen_pagination_tokens": [],
                "session_start_time": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        else:
            # 매 요청마다 region 갱신 — 공격자가 다른 리전 탐색 시 ARN 일관성 유지
            current_region = (
                headers.get("X-Amz-Region") or headers.get("x-amz-region") or ""
            ).strip()
            if current_region:
                _session_state[session_id]["region"] = current_region
        return deepcopy(_session_state[session_id])


def update_world_state_tool(
    session_id: str,
    current: dict[str, Any],
    canonical: CanonicalRequest,
    agent_output: Any,
    rendered_meta: dict[str, Any],
    field_values: dict[str, Any] | None = None,
) -> None:
    next_state = deepcopy(current)

    next_state["phase"] = agent_output.intent_phase
    risk_score = float(next_state.get("risk_score", 0.2)) + float(
        agent_output.risk_delta
    )
    next_state["risk_score"] = max(0.0, min(1.0, risk_score))

    action_key = f"{canonical.service}:{canonical.operation}"
    last_actions = list(next_state.get("last_actions", []))
    last_actions.append(action_key)
    next_state["last_actions"] = last_actions[-10:]

    # 에이전트가 응답한 operation 추적 (has_cached_agent_response_tool 의 broad routing 용)
    agent_responses = list(next_state.get("agent_responses", []))
    if action_key not in agent_responses:
        agent_responses.append(action_key)
    next_state["agent_responses"] = agent_responses

    # Merge environment_delta into world_state
    for key, value in agent_output.environment_delta.items():
        if isinstance(value, list):
            existing = list(next_state.get(key, []))
            for item in value:
                if item not in existing:
                    existing.append(item)
            next_state[key] = existing
        else:
            next_state[key] = value

    # Track exposed assets from the serialized response
    exposed_assets = list(next_state.get("exposed_assets", []))
    for asset in rendered_meta.get("assets", []):
        if asset not in exposed_assets:
            exposed_assets.append(asset)
    next_state["exposed_assets"] = exposed_assets

    # Store name-type field values for cross-call consistency (scoped by service)
    if field_values:
        known_names = dict(next_state.get("known_names", {}))
        _merge_known_names(known_names, field_values, canonical.service)
        next_state["known_names"] = known_names

    # Cache full field_values for read operations so repeated calls return identical results
    if field_values and _should_cache_operation(canonical.operation):
        response_cache = dict(next_state.get("response_cache", {}))
        cache_key = _param_cache_key(
            canonical.service, canonical.operation, canonical.target_identifiers
        )
        if cache_key not in response_cache:
            response_cache[cache_key] = deepcopy(field_values)
        next_state["response_cache"] = response_cache

    with _lock:
        _session_state[session_id] = next_state


def record_native_interaction_tool(
    session_id: str,
    canonical: CanonicalRequest,
    response_body: str,
    *,
    status_code: int = 200,
) -> None:
    add_to_session_history_tool(
        session_id,
        (
            f"service={canonical.service}, operation={canonical.operation}, "
            f"source=moto_native, status={status_code}"
        ),
        response_body,
    )

    with _lock:
        current = _session_state.get(session_id, {})
        next_state = deepcopy(current)
        next_state.setdefault("session_id", session_id)
        next_state.setdefault("persona", "mid-size-prod-account")
        next_state.setdefault("region", "us-east-1")
        next_state.setdefault("phase", "recon")
        next_state.setdefault("exposed_roles", ["ReadOnlyOpsRole"])
        next_state.setdefault("credibility_level", "medium")
        next_state.setdefault("risk_score", 0.2)
        next_state.setdefault(
            "consistency_locks",
            {
                "account_id": _derive_account_id(session_id),
                "os_family": "Amazon Linux 2",
            },
        )

        action_key = f"{canonical.service}:{canonical.operation}"
        last_actions = list(next_state.get("last_actions", []))
        last_actions.append(action_key)
        next_state["last_actions"] = last_actions[-10:]

        exposed_assets = list(next_state.get("exposed_assets", []))
        for asset in _extract_assets_from_response(response_body):
            if asset not in exposed_assets:
                exposed_assets.append(asset)
        next_state["exposed_assets"] = exposed_assets[-50:]

        # native 응답에서도 이름 필드 추출 → 에이전트 응답과 이름 일관성 유지
        known_names = dict(next_state.get("known_names", {}))
        try:
            parsed = json.loads(response_body)
            _merge_known_names(known_names, parsed, canonical.service)
            _merge_aws_tags(known_names, parsed, canonical.service)
            # native 응답의 NextToken을 기록 → 에이전트가 해당 토큰 수신 시 빈 결과 반환
            _record_pagination_tokens(next_state, parsed)
        except Exception:
            # JSON 파싱 실패 시 XML로 재시도 (EC2/S3 등 XML 프로토콜 서비스)
            _merge_known_names_from_xml(known_names, response_body, canonical.service)
            _record_pagination_tokens_from_xml(next_state, response_body)
        next_state["known_names"] = known_names

        _session_state[session_id] = next_state


def _extract_assets_from_response(response_body: str) -> list[str]:
    patterns = [
        r"arn:aws:[A-Za-z0-9-]+:[^\s\"',<]+",
        r"\bi-[0-9a-f]{8,17}\b",
        r"\bvol-[0-9a-f]{8,17}\b",
        r"\bami-[0-9a-f]{8,17}\b",
        r"\bsnap-[0-9a-f]{8,17}\b",
        r"\bvpc-[0-9a-f]{8,17}\b",
        r"\bsubnet-[0-9a-f]{8,17}\b",
        r"\bsg-[0-9a-f]{8,17}\b",
        r"\bsha256:[0-9a-fA-F]{3,64}\b",
        r"\bupload-[A-Za-z0-9-]+\b",
    ]
    assets: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, response_body):
            cleaned = match.rstrip(".,)]}")
            if cleaned and cleaned not in assets:
                assets.append(cleaned)
            if len(assets) >= 50:
                return assets
    return assets


_SKIP_NAME_KEYS = {"nexttoken", "marker", "requestid", "token"}
_CACHE_OPERATION_PREFIXES = (
    "get",
    "describe",
    "list",
    "batch",
    "query",
    "head",
    "scan",
)


def _should_cache_operation(operation: str) -> bool:
    return operation.lower().startswith(_CACHE_OPERATION_PREFIXES)


def _param_cache_key(
    service: str, operation: str, target_identifiers: dict[str, Any]
) -> str:
    """target_identifiers 해시를 포함한 응답 캐시 키 — 파라미터가 다른 동일 operation 구분."""
    if target_identifiers:
        params_hash = hashlib.sha256(
            json.dumps(
                sorted(target_identifiers.items()), separators=(",", ":")
            ).encode()
        ).hexdigest()[:8]
        return f"{service}:{operation}:{params_hash}"
    return f"{service}:{operation}"


def _merge_known_names(
    known_names: dict[str, Any], field_values: Any, service: str = ""
) -> None:
    """Recursively extract *Name string fields into known_names, scoped by service."""
    if isinstance(field_values, dict):
        for key, value in field_values.items():
            lowered = key.lower()
            if lowered in _SKIP_NAME_KEYS:
                continue
            if (
                isinstance(value, str)
                and value
                and (lowered == "name" or lowered.endswith("name"))
            ):
                scoped = f"{service}:{key}" if service else key
                known_names.setdefault(scoped, value)
            else:
                _merge_known_names(known_names, value, service)
    elif isinstance(field_values, list):
        for item in field_values:
            _merge_known_names(known_names, item, service)


_PAGINATION_KEYS = {"nexttoken", "marker", "nextpage", "nextmarker", "paginationtoken"}


def _record_pagination_tokens(state: dict[str, Any], parsed: Any) -> None:
    """JSON 응답에서 페이지네이션 토큰 값을 세션에 기록."""
    if not isinstance(parsed, dict):
        return
    tokens: list[str] = list(state.get("seen_pagination_tokens", []))
    for key, value in parsed.items():
        if key.lower() in _PAGINATION_KEYS and isinstance(value, str) and value:
            if value not in tokens:
                tokens.append(value)
    state["seen_pagination_tokens"] = tokens[-50:]


def _record_pagination_tokens_from_xml(state: dict[str, Any], xml_body: str) -> None:
    """XML 응답에서 페이지네이션 토큰 값을 세션에 기록."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_body.strip())
    except Exception:
        return
    tokens: list[str] = list(state.get("seen_pagination_tokens", []))
    for el in root.iter():
        local = el.tag.split("}")[-1].lower()
        if local in _PAGINATION_KEYS and el.text and el.text.strip():
            val = el.text.strip()
            if val not in tokens:
                tokens.append(val)
    state["seen_pagination_tokens"] = tokens[-50:]


def _merge_known_names_from_xml(
    known_names: dict[str, Any], xml_body: str, service: str = ""
) -> None:
    """EC2/S3 등 XML 응답에서 Name 필드와 AWS tagSet 태그를 known_names에 추출."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_body.strip())
    except Exception:
        return

    # <value> 바로 앞 형제가 <key>Name</key> 인 tagSet 패턴 처리
    for item in root.iter():
        children = list(item)
        for i, child in enumerate(children):
            tag_local = child.tag.split("}")[-1].lower()
            if tag_local == "key" and child.text and child.text.strip() == "Name":
                if i + 1 < len(children):
                    val_el = children[i + 1]
                    val_local = val_el.tag.split("}")[-1].lower()
                    if val_local == "value" and val_el.text and val_el.text.strip():
                        scoped = f"{service}:Name" if service else "Name"
                        known_names.setdefault(scoped, val_el.text.strip())

    # 직접 <Name> 또는 <*name> 요소 처리
    for el in root.iter():
        local = el.tag.split("}")[-1]
        lowered = local.lower()
        if (
            (lowered == "name" or lowered.endswith("name"))
            and el.text
            and el.text.strip()
        ):
            scoped = f"{service}:{local}" if service else local
            known_names.setdefault(scoped, el.text.strip())


def _merge_aws_tags(known_names: dict[str, Any], data: Any, service: str = "") -> None:
    """AWS Tags 배열([{"Key":"Name","Value":"..."}]) 에서 Name 태그 값을 추출."""
    if isinstance(data, list):
        name_val = None
        for item in data:
            if isinstance(item, dict):
                key = item.get("Key") or item.get("key")
                val = item.get("Value") or item.get("value")
                if (
                    isinstance(key, str)
                    and key == "Name"
                    and isinstance(val, str)
                    and val
                ):
                    name_val = val
                else:
                    _merge_aws_tags(known_names, item, service)
        if name_val:
            scoped = f"{service}:Name" if service else "Name"
            known_names.setdefault(scoped, name_val)
    elif isinstance(data, dict):
        for value in data.values():
            _merge_aws_tags(known_names, value, service)


def _derive_account_id(session_id: str) -> str:
    """Derive a deterministic 12-digit fake AWS account ID from the session key.

    Different attackers get different account IDs; the same attacker always
    sees the same one across requests.
    """
    digest = int(hashlib.sha256(session_id.encode()).hexdigest()[:10], 16)
    return str(100000000000 + (digest % 900000000000))
