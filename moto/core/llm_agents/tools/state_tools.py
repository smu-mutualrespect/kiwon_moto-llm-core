from __future__ import annotations

import hashlib
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .request_tools import CanonicalRequest

_session_storage: dict[str, list[dict[str, str]]] = {}
_session_state: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()


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
                "response_cache": {},
                "session_start_time": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
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
        cache_key = f"{canonical.service}:{canonical.operation}"
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


def _derive_account_id(session_id: str) -> str:
    """Derive a deterministic 12-digit fake AWS account ID from the session key.

    Different attackers get different account IDs; the same attacker always
    sees the same one across requests.
    """
    digest = int(hashlib.sha256(session_id.encode()).hexdigest()[:10], 16)
    return str(100000000000 + (digest % 900000000000))
