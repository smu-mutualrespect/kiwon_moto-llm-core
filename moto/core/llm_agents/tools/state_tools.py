from __future__ import annotations

import hashlib
import json
import re
import threading
import time as _time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .request_tools import CanonicalRequest

_session_storage: dict[str, list[dict[str, str]]] = {}
_session_state: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()
_action_log: dict[str, list[dict[str, Any]]] = {}
_last_activity: dict[str, float] = {}
_reported_sessions: set[str] = set()
_reported_action_counts: dict[str, int] = {}
_session_summary: dict[str, dict[str, Any]] = {}  # 보고 완료 세션의 요약 정보


def _append_action_log(
    session_id: str,
    service: str,
    operation: str,
    phase: str,
    risk_score: float,
    source: str,
    *,
    status_code: int = 200,
    new_assets: list[str] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "service": service,
        "operation": operation,
        "phase": phase,
        "risk_score": round(risk_score, 3),
        "source": source,
        "status_code": status_code,
    }
    if new_assets:
        entry["new_assets"] = new_assets
    with _lock:
        _action_log.setdefault(session_id, []).append(entry)
        _last_activity[session_id] = _time.time()


def has_any_agent_response(session_id: str) -> bool:
    """True if the agent has responded at least once in this session."""
    with _lock:
        state = _session_state.get(session_id, {})
    return bool(state.get("agent_responses"))


def has_session_history(session_id: str) -> bool:
    """True if this session has made any prior request (agent or moto-native)."""
    with _lock:
        return session_id in _session_state


def has_cached_agent_response_tool(
    session_id: str, service: str, operation: str
) -> bool:
    """에이전트가 이 operation을 이전에 응답한 적 있는지 확인 (agent_responses 기반).

    같은 세션에서 agent가 write 연산을 처리했고 그 write가 현재 read operation에
    영향을 줄 수 있으면 agent/native-patch 경로로 라우팅해 일관성을 유지한다.
    """
    with _lock:
        state = _session_state.get(session_id, {})
    if f"{service}:{operation}" in state.get("agent_responses", []):
        return True
    # 서비스 단위 broad routing은 너무 넓다. write operation이 실제로 영향을 주는
    # read operation만 라우팅한다.
    affected_reads = state.get("affected_read_operations", [])
    if f"{service}:{operation}" in affected_reads:
        return True
    return False


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
    profile_state = _honeypot_profile_state(session_id)
    with _lock:
        if session_id not in _session_state:
            _session_state[session_id] = {
                "session_id": session_id,
                "persona": profile_state.get("persona", "mid-size-prod-account"),
                "region": profile_state.get(
                    "region", headers.get("X-Amz-Region", "us-east-1")
                ),
                "phase": "recon",
                "exposed_assets": list(profile_state.get("exposed_assets", [])),
                "exposed_roles": list(
                    profile_state.get("exposed_roles", ["ReadOnlyOpsRole"])
                ),
                "credibility_level": "medium",
                "risk_score": 0.2,
                "last_actions": [],
                "consistency_locks": {
                    "account_id": profile_state.get(
                        "account_id", _derive_account_id(session_id)
                    ),
                    "os_family": profile_state.get("os_family", "Amazon Linux 2"),
                },
                **(
                    {"honeypot_profile": profile_state["honeypot_profile"]}
                    if "honeypot_profile" in profile_state
                    else {}
                ),
                "known_names": {},
                "agent_responses": [],
                "response_cache": {},
                "resource_registry": {},
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


def _honeypot_profile_state(session_id: str) -> dict[str, Any]:
    try:
        from moto.core.llm_agents import honeypot_profile as profile
        from moto.core.llm_agents.honeypot_aws_mocks import is_honeypot_access_key
    except Exception:
        return {}
    if not is_honeypot_access_key(session_id):
        return {}
    return {
        "persona": "nexora-prod-devops-bastion",
        "region": profile.REGION,
        "account_id": profile.ACCOUNT_ID,
        "os_family": profile.OS_RELEASE,
        "exposed_roles": [profile.INSTANCE_PROFILE],
        "exposed_assets": [
            profile.IAM_USER_ARN,
            profile.BASTION_ROLE_ARN,
            f"arn:aws:eks:{profile.REGION}:{profile.ACCOUNT_ID}:cluster/{profile.EKS_CLUSTER}",
            profile.BACKEND_IMAGE,
            f"s3://{profile.BACKUP_BUCKET}/daily",
        ],
        "honeypot_profile": {
            "company": profile.COMPANY_PREFIX,
            "environment": profile.ENVIRONMENT,
            "account_id": profile.ACCOUNT_ID,
            "region": profile.REGION,
            "iam_user": profile.IAM_USER,
            "iam_arn": profile.IAM_USER_ARN,
            "hostname": profile.HOSTNAME,
            "linux_user": profile.LINUX_USER,
            "eks_cluster": profile.EKS_CLUSTER,
            "ecr_registry": profile.ECR_REGISTRY,
            "backup_bucket": profile.BACKUP_BUCKET,
        },
    }


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

    # agent가 write 연산을 처리했음을 기록 — field_values 유무와 무관하게 항상 실행
    if not _should_cache_operation(canonical.operation):
        agent_modified = list(next_state.get("agent_modified_services", []))
        if canonical.service not in agent_modified:
            agent_modified.append(canonical.service)
        next_state["agent_modified_services"] = agent_modified
        _invalidate_read_cache(next_state, canonical)

        affected_reads = list(next_state.get("affected_read_operations", []))
        for read_service, read_operation in _affected_read_operations_for_write(
            canonical
        ):
            read_key = f"{read_service}:{read_operation}"
            if read_key not in affected_reads:
                affected_reads.append(read_key)
        next_state["affected_read_operations"] = affected_reads[-50:]

    # Store name-type field values for cross-call consistency (scoped by service)
    if field_values:
        known_names = dict(next_state.get("known_names", {}))
        _merge_known_names(known_names, field_values, canonical.service)
        next_state["known_names"] = known_names
        resource_registry = dict(next_state.get("resource_registry", {}))
        _merge_resource_registry(resource_registry, canonical, field_values)
        next_state["resource_registry"] = resource_registry
        if not _should_cache_operation(canonical.operation):
            # write 연산 결과에서 리소스별 상태 변경 추출 → resource_state_patches 저장
            # _try_native_as_base가 moto native baseline에 이 패치를 덮어써서 일관성 유지
            patches = _extract_resource_patches(field_values)
            if patches:
                rsp = dict(next_state.get("resource_state_patches", {}))
                svc_patches = dict(rsp.get(canonical.service, {}))
                for rid, patch in patches.items():
                    _prev = svc_patches.get(rid)
                    _entry: dict[str, Any] = (
                        dict(_prev) if isinstance(_prev, dict) else {}
                    )
                    _entry.update(patch)
                    svc_patches[rid] = _entry
                rsp[canonical.service] = svc_patches
                next_state["resource_state_patches"] = rsp

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

    _append_action_log(
        session_id,
        canonical.service,
        canonical.operation,
        agent_output.intent_phase,
        next_state["risk_score"],
        source="agent",
        new_assets=rendered_meta.get("assets") or [],
    )


def record_native_interaction_tool(
    session_id: str,
    canonical: CanonicalRequest,
    response_body: str,
    *,
    status_code: int = 200,
    include_history: bool = True,
) -> None:
    if include_history:
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
        profile_state = _honeypot_profile_state(session_id)
        next_state = deepcopy(current)
        next_state.setdefault("session_id", session_id)
        next_state.setdefault(
            "persona", profile_state.get("persona", "mid-size-prod-account")
        )
        next_state.setdefault("region", profile_state.get("region", "us-east-1"))
        next_state.setdefault("phase", "recon")
        next_state.setdefault(
            "exposed_roles", profile_state.get("exposed_roles", ["ReadOnlyOpsRole"])
        )
        next_state.setdefault("credibility_level", "medium")
        next_state.setdefault("risk_score", 0.2)
        next_state.setdefault("resource_registry", {})
        next_state.setdefault(
            "exposed_assets", list(profile_state.get("exposed_assets", []))
        )
        next_state.setdefault(
            "consistency_locks",
            {
                "account_id": profile_state.get(
                    "account_id", _derive_account_id(session_id)
                ),
                "os_family": profile_state.get("os_family", "Amazon Linux 2"),
            },
        )
        if "honeypot_profile" in profile_state:
            next_state.setdefault("honeypot_profile", profile_state["honeypot_profile"])

        action_key = f"{canonical.service}:{canonical.operation}"
        last_actions = list(next_state.get("last_actions", []))
        last_actions.append(action_key)
        next_state["last_actions"] = last_actions[-10:]

        newly_found: list[str] = []
        if include_history:
            exposed_assets = list(next_state.get("exposed_assets", []))
            for asset in _extract_assets_from_response(response_body):
                if asset not in exposed_assets:
                    exposed_assets.append(asset)
                    newly_found.append(asset)
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

        # moto native 쓰기 연산 성공 시 캐시 무효화 (agent 경로와 동일한 정책)
        if status_code < 300 and not _should_cache_operation(canonical.operation):
            _invalidate_read_cache(next_state, canonical)

        _session_state[session_id] = next_state

    _append_action_log(
        session_id,
        canonical.service,
        canonical.operation,
        next_state.get("phase", "recon"),
        float(next_state.get("risk_score", 0.2)),
        source="moto_native",
        status_code=status_code,
        new_assets=newly_found or [],
    )


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
    "query",
    "head",
    "scan",
    # "batch" 제외 — BatchWriteItem/BatchDeleteItem 등 쓰기 연산이 포함되어 캐시 오분류 발생
)


def _should_cache_operation(operation: str) -> bool:
    if operation in {"ListServiceSpecificCredentials"}:
        return False
    return operation.lower().startswith(_CACHE_OPERATION_PREFIXES)


_WRITE_TO_AFFECTED_READS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("ec2", "MonitorInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "UnmonitorInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "RunInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "StartInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "StopInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "TerminateInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "RebootInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "CreateSecurityGroup"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "AuthorizeSecurityGroupIngress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "AuthorizeSecurityGroupEgress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "RevokeSecurityGroupIngress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "RevokeSecurityGroupEgress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "CreateVolume"): (("ec2", "DescribeVolumes"),),
    ("ec2", "AttachVolume"): (
        ("ec2", "DescribeVolumes"),
        ("ec2", "DescribeInstances"),
    ),
    ("ec2", "DetachVolume"): (
        ("ec2", "DescribeVolumes"),
        ("ec2", "DescribeInstances"),
    ),
    ("ec2", "DeleteVolume"): (("ec2", "DescribeVolumes"),),
}


def _affected_read_operations_for_write(
    canonical: CanonicalRequest,
) -> tuple[tuple[str, str], ...]:
    return _WRITE_TO_AFFECTED_READS.get((canonical.service, canonical.operation), ())


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


_DELETE_OPERATION_PREFIXES = (
    "delete",
    "remove",
    "terminate",
    "deregister",
    "detach",
    "revoke",
    "disassociate",
    "unsubscribe",
)

_AWS_REGION_RE = re.compile(
    r"^(us|eu|ap|sa|ca|me|af|il|mx)-(north|south|east|west|central|"
    r"northeast|southeast|northwest|southwest)-\d+[a-z]?$"
)


def _is_delete_operation(canonical: CanonicalRequest) -> bool:
    return canonical.operation.lower().startswith(_DELETE_OPERATION_PREFIXES)


def _merge_resource_registry(
    registry: dict[str, Any], canonical: CanonicalRequest, field_values: Any
) -> None:
    identity_keys = _resource_identity_keys(canonical)
    if not identity_keys:
        return

    is_delete = _is_delete_operation(canonical)

    extracted: dict[str, str] = {}
    for key, value in _iter_resource_leaf_strings(field_values):
        lowered = key.lower()
        if not value:
            continue
        if lowered in _SKIP_NAME_KEYS:
            continue
        if lowered.endswith("arn") and value.startswith("arn:aws:"):
            extracted.setdefault(key, value)
            extracted.setdefault("arn", value)
        elif lowered.endswith("id") and _looks_like_resource_id(value):
            extracted.setdefault(key, value)
            extracted.setdefault("id", value)
        elif (lowered == "name" or lowered.endswith("name")) and value:
            extracted.setdefault(key, value)
            extracted.setdefault("name", value)

    if not extracted and not is_delete:
        return

    for identity_key in identity_keys:
        current = dict(registry.get(identity_key, {}))
        for key, value in extracted.items():
            current.setdefault(key, value)
        if is_delete:
            current["__deleted__"] = True
        registry[identity_key] = current


def _invalidate_read_cache(state: dict[str, Any], canonical: CanonicalRequest) -> None:
    """Delete 이후 같은 서비스의 read cache를 제거해서 삭제 후 조회가 stale 응답을 반환하지 않게 한다."""
    response_cache = dict(state.get("response_cache", {}))
    if not response_cache:
        return
    prefix = f"{canonical.service}:"
    keys_to_remove = [k for k in response_cache if k.startswith(prefix)]
    for key in keys_to_remove:
        del response_cache[key]
    state["response_cache"] = response_cache


def _resource_identity_keys(canonical: CanonicalRequest) -> list[str]:
    keys: list[str] = []
    for source in (canonical.target_identifiers, canonical.request_params):
        for key, value in source.items():
            if isinstance(value, (dict, list)) or value in (None, ""):
                continue
            lowered = str(key).lower().split(".")[-1]
            if lowered.isdigit():
                continue
            if any(
                token in lowered
                for token in ("name", "id", "arn", "repository", "secret", "user")
            ):
                text = str(value)
                for registry_key in (
                    f"{canonical.service}:{lowered}:{text}",
                    f"{canonical.service}:primary:{text}",
                ):
                    if registry_key not in keys:
                        keys.append(registry_key)
    return keys


def _iter_resource_leaf_strings(
    obj: Any, parent_key: str = ""
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                results.append((str(key), value))
            else:
                results.extend(_iter_resource_leaf_strings(value, str(key)))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_iter_resource_leaf_strings(item, parent_key))
    return results


def _looks_like_resource_id(value: str) -> bool:
    if _AWS_REGION_RE.match(value):
        return False
    return bool(
        re.match(r"^[a-z][a-z0-9-]*-[A-Za-z0-9-]{4,}$", value)
        or re.match(r"^[0-9a-f]{8}-[0-9a-f-]{13,}$", value)
    )


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


_RESOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9-]*-[0-9a-f]{8,}$")


def _extract_resource_patches(field_values: Any) -> dict[str, dict[str, Any]]:
    """write 연산 결과에서 리소스 ID별 상태 변경 정보를 추출.

    배열 내 항목에서 *Id 필드를 key로, 나머지 필드를 patch로 반환한다.
    예: MonitorInstances → {"i-xxx": {"Monitoring": {"State": "enabled"}}}
    """
    patches: dict[str, dict[str, Any]] = {}
    if not isinstance(field_values, dict):
        return patches
    for value in field_values.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            resource_id: str | None = None
            patch: dict[str, Any] = {}
            for k, v in item.items():
                if k.endswith("Id") and isinstance(v, str) and _RESOURCE_ID_RE.match(v):
                    resource_id = v
                else:
                    patch[k] = v
            if resource_id and patch:
                existing = patches.get(resource_id, {})
                existing.update(patch)
                patches[resource_id] = existing
    return patches


def get_full_action_log(session_id: str) -> list[dict[str, Any]]:
    with _lock:
        return list(_action_log.get(session_id, []))


def get_session_state_snapshot(session_id: str) -> dict[str, Any]:
    with _lock:
        return deepcopy(_session_state.get(session_id, {}))


def get_idle_sessions(idle_seconds: float) -> list[str]:
    """Return inactive sessions with unreported activity."""
    now = _time.time()
    with _lock:
        return [
            sid
            for sid, last in _last_activity.items()
            if now - last >= idle_seconds
            and len(_action_log.get(sid, [])) > _reported_action_counts.get(sid, 0)
        ]


def mark_session_reported(session_id: str, report_path: str = "") -> None:
    """세션을 보고 완료로 표시합니다.

    보고서 생성 이후 같은 세션 ID로 활동이 이어질 수 있으므로 원본 action/state는
    유지하고, 마지막 보고서가 포함한 action 개수만 저장합니다.
    """
    with _lock:
        _reported_sessions.add(session_id)
        log = _action_log.get(session_id, [])
        action_count = len(log)
        _reported_action_counts[session_id] = action_count
        _session_summary[session_id] = {
            "action_count": action_count,
            "services": sorted({e["service"] for e in log}),
            "reported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "report_path": report_path,
        }


def is_session_reported(session_id: str) -> bool:
    with _lock:
        return session_id in _reported_sessions


def get_reported_action_count(session_id: str) -> int:
    with _lock:
        return _reported_action_counts.get(session_id, 0)


def get_report_path(session_id: str) -> str:
    with _lock:
        return str(_session_summary.get(session_id, {}).get("report_path") or "")
