"""STIX 2.1 번들 생성기.

`stix2` 라이브러리 없이 STIX 2.1 스펙을 직접 구현합니다.
생성된 JSON은 MITRE ATT&CK Navigator, SIEM, OpenCTI, MISP 등에 import 가능합니다.

참고 스펙:
  https://docs.oasis-open.org/cti/stix/v2.1/stix-v2.1.html
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


# uuid5(NAMESPACE_DNS, "phantomgate.honeypot") — 고정 결정론적 UUID
_PHANTOMGATE_IDENTITY_ID = f"identity--{uuid.uuid5(uuid.NAMESPACE_DNS, 'phantomgate.honeypot')}"
_MITRE_IDENTITY_ID = "identity--c78cb6e5-0c4b-4611-8297-d1b8b55e40b5"  # MITRE 공식 ID


def generate_stix_bundle(
    session_id: str,
    iocs: dict[str, Any],
    ttp_map: dict[str, dict[str, Any]],
    state: dict[str, Any],
    action_log: list[dict[str, Any]],  # noqa: ARG001 — used by sub-builders
) -> dict[str, Any]:
    """세션 데이터를 STIX 2.1 Bundle 형식으로 변환합니다.

    포함 객체:
    - identity (PhantomGate)
    - threat-actor (공격자)
    - attack-pattern (기법별 1개)
    - indicator (자격증명/행위 기반 IOC)
    - observed-data (원시 관찰 기록)
    - relationship (연결 관계)
    - sighting (관찰 → 기법 연결)
    """
    now_ts = _now_str()
    actor_id = _det_uuid("threat-actor", session_id)
    observed_id = _det_uuid("observed-data", session_id)

    # indicator 먼저 빌드 — observed-data.object_refs에 참조하기 위해
    indicators = _build_indicators(session_id, iocs, action_log, now_ts)
    indicator_ids = [ind["id"] for ind in indicators]

    objects: list[dict[str, Any]] = [
        _phantomgate_identity(),
        _threat_actor(session_id, actor_id, state, action_log, now_ts),
        _observed_data(observed_id, session_id, action_log, indicator_ids, now_ts),
    ]

    attack_pattern_ids: dict[str, str] = {}
    for tech_id, info in ttp_map.items():
        ap_id = _det_uuid("attack-pattern", tech_id)
        attack_pattern_ids[tech_id] = ap_id
        objects.append(_attack_pattern(ap_id, tech_id, info, now_ts))
        # threat-actor -[uses]-> attack-pattern
        objects.append(_relationship(
            _det_uuid("relationship", f"{actor_id}-uses-{ap_id}"),
            "uses", f"threat-actor--{actor_id}", f"attack-pattern--{ap_id}", now_ts,
        ))

    for ind in indicators:
        objects.append(ind)
        # indicator -[indicates]-> threat-actor
        objects.append(_relationship(
            _det_uuid("relationship", f"{ind['id']}-indicates-{actor_id}"),
            "indicates", ind["id"], f"threat-actor--{actor_id}", now_ts,
        ))

    # observed-data → attack-pattern sighting
    for tech_id, ap_id in attack_pattern_ids.items():
        evidence = ttp_map[tech_id].get("evidence", [])
        objects.append(_sighting(
            _det_uuid("sighting", f"{observed_id}-{ap_id}"),
            f"observed-data--{observed_id}", f"attack-pattern--{ap_id}",
            len(evidence), action_log, now_ts,
        ))

    return {
        "type": "bundle",
        "id": f"bundle--{_det_uuid_raw(session_id)}",
        "spec_version": "2.1",
        "objects": objects,
    }


# ──────────────────────────────────────────────
# STIX 객체 생성 함수들
# ──────────────────────────────────────────────

def _phantomgate_identity() -> dict[str, Any]:
    return {
        "type": "identity",
        "spec_version": "2.1",
        "id": _PHANTOMGATE_IDENTITY_ID,
        "created": "2025-01-01T00:00:00Z",
        "modified": "2025-01-01T00:00:00Z",
        "name": "PhantomGate Honeypot",
        "identity_class": "system",
        "description": "AWS 클라우드 허니팟 위협 인텔리전스 시스템",
    }


def _threat_actor(
    session_id: str,
    actor_id: str,
    state: dict[str, Any],
    action_log: list[dict[str, Any]],
    now_ts: str,
) -> dict[str, Any]:
    risk = float(state.get("risk_score", 0.0))
    sophistication = (
        "advanced" if risk >= 0.8
        else "intermediate" if risk >= 0.5
        else "novice"
    )
    services = ", ".join(sorted({e["service"] for e in action_log})) or "unknown"
    return {
        "type": "threat-actor",
        "spec_version": "2.1",
        "id": f"threat-actor--{actor_id}",
        "created": now_ts,
        "modified": now_ts,
        "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
        "name": f"Unknown Actor [{session_id[:16]}]",
        "threat_actor_types": ["criminal"],
        "sophistication": sophistication,
        "resource_level": "individual",
        "primary_motivation": "organizational-gain",
        "confidence": _risk_to_confidence(risk),
        "labels": ["honeypot-observed"],
        "description": (
            f"허니팟 세션 {session_id}에서 관찰된 공격자. "
            f"최종 위험 점수: {risk:.2f}/1.00. "
            f"대상 서비스: {services}"
        ),
    }


def _attack_pattern(
    ap_id: str,
    tech_id: str,
    info: dict[str, Any],
    now_ts: str,
) -> dict[str, Any]:
    url = info.get("url") or f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}/"
    return {
        "type": "attack-pattern",
        "spec_version": "2.1",
        "id": f"attack-pattern--{ap_id}",
        "created": now_ts,
        "modified": now_ts,
        "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
        "name": f"{tech_id} - {info.get('name', tech_id)}",
        "description": info.get("description", ""),
        "kill_chain_phases": [
            {
                "kill_chain_name": "mitre-attack",
                "phase_name": info.get("tactic", "unknown").lower().replace(" ", "-"),
            }
        ],
        "external_references": [
            {
                "source_name": "mitre-attack",
                "external_id": tech_id,
                "url": url,
            }
        ],
        "x_mitre_platforms": info.get("platforms", ["IaaS"]),
        "x_mitre_detection": info.get("detection", ""),
    }


def _observed_data(
    obs_id: str,
    session_id: str,
    action_log: list[dict[str, Any]],
    object_refs: list[str],
    now_ts: str,
) -> dict[str, Any]:
    first_ts = action_log[0]["timestamp"] if action_log else now_ts
    last_ts = action_log[-1]["timestamp"] if action_log else now_ts
    # STIX 2.1 스펙: object_refs는 non-empty 필요 — indicator ID 또는 identity를 참조
    refs = object_refs if object_refs else [_PHANTOMGATE_IDENTITY_ID]
    return {
        "type": "observed-data",
        "spec_version": "2.1",
        "id": f"observed-data--{obs_id}",
        "created": now_ts,
        "modified": now_ts,
        "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
        "first_observed": first_ts,
        "last_observed": last_ts,
        "number_observed": len(action_log),
        "object_refs": refs,
        "description": f"세션 {session_id}에서 관찰된 {len(action_log)}건의 AWS API 호출",
        "labels": ["honeypot-session"],
        "x_phantomgate_session_id": session_id,
        "x_phantomgate_api_calls": [
            {
                "seq": i + 1,
                "timestamp": e["timestamp"],
                "service": e["service"],
                "operation": e["operation"],
                "phase": e["phase"],
                "risk_score": e.get("risk_score", 0),
                "source": e["source"],
            }
            for i, e in enumerate(action_log)
        ],
    }


def _build_indicators(
    session_id: str,
    iocs: dict[str, Any],
    action_log: list[dict[str, Any]],
    now_ts: str,
) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []

    # AccessKeyId 지표
    access_key = iocs.get("access_key_id") or iocs.get("session_identifier")
    if access_key:
        ind_id = f"indicator--{_det_uuid('indicator-cred', access_key)}"
        indicators.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": ind_id,
            "created": now_ts,
            "modified": now_ts,
            "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
            "name": f"악성 AWS 자격증명 — {access_key[:16]}",
            "description": (
                f"허니팟에서 공격에 사용된 AWS 자격증명. "
                f"유형: {iocs.get('credential_type', '알 수 없음')}"
            ),
            "indicator_types": ["malicious-activity", "compromised"],
            "pattern": f"[user-account:account_type = 'aws' AND user-account:user_id = '{access_key}']",
            "pattern_type": "stix",
            "valid_from": now_ts,
            "confidence": 90,
            "labels": ["aws-access-key"],
        })

    # 행위 기반 지표 — 고위험 operation 시퀀스
    high_risk_ops = iocs.get("high_risk_operations", [])
    if len(high_risk_ops) >= 2:
        seq_str = " → ".join(high_risk_ops[:4])
        ind_id = f"indicator--{_det_uuid('indicator-behavior', session_id)}"
        indicators.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": ind_id,
            "created": now_ts,
            "modified": now_ts,
            "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
            "name": "클라우드 정찰 → 민감 자산 접근 행위 패턴",
            "description": f"관찰된 고위험 작업 시퀀스: {seq_str}",
            "indicator_types": ["malicious-activity", "anomalous-activity"],
            "pattern": (
                "[network-traffic:dst_ref.type = 'domain-name' "
                "AND network-traffic:dst_ref.value MATCHES 'amazonaws\\\\.com']"
            ),
            "pattern_type": "stix",
            "valid_from": now_ts,
            "confidence": 75,
            "labels": ["behavioral-ioc", "cloud-attack-chain"],
            "x_phantomgate_operations": high_risk_ops,
        })

    # 자동화 도구 징후 지표
    if iocs.get("automation_indicator"):
        ind_id = f"indicator--{_det_uuid('indicator-automation', session_id)}"
        indicators.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": ind_id,
            "created": now_ts,
            "modified": now_ts,
            "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
            "name": "자동화 공격 도구 사용 징후",
            "description": iocs["automation_indicator"],
            "indicator_types": ["anomalous-activity"],
            "pattern": "[process:command_line MATCHES 'aws\\s+(iam|s3|secretsmanager|ec2)']",
            "pattern_type": "stix",
            "valid_from": now_ts,
            "confidence": 60,
            "labels": ["automated-tool", "behavioral-ioc"],
        })

    return indicators


def _relationship(
    rel_id: str,
    rel_type: str,
    source_ref: str,
    target_ref: str,
    now_ts: str,
) -> dict[str, Any]:
    return {
        "type": "relationship",
        "spec_version": "2.1",
        "id": f"relationship--{rel_id}",
        "created": now_ts,
        "modified": now_ts,
        "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
        "relationship_type": rel_type,
        "source_ref": source_ref,
        "target_ref": target_ref,
    }


def _sighting(
    sighting_id: str,
    obs_ref: str,
    ap_ref: str,
    count: int,
    action_log: list[dict[str, Any]],
    now_ts: str,
) -> dict[str, Any]:
    first_ts = action_log[0]["timestamp"] if action_log else now_ts
    last_ts = action_log[-1]["timestamp"] if action_log else now_ts
    return {
        "type": "sighting",
        "spec_version": "2.1",
        "id": f"sighting--{sighting_id}",
        "created": now_ts,
        "modified": now_ts,
        "created_by_ref": _PHANTOMGATE_IDENTITY_ID,
        "sighting_of_ref": ap_ref,
        "observed_data_refs": [obs_ref],
        "first_seen": first_ts,
        "last_seen": last_ts,
        "count": max(1, count),
        "where_sighted_refs": [_PHANTOMGATE_IDENTITY_ID],
    }


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _det_uuid(namespace: str, value: str) -> str:
    """namespace + value 기반의 결정론적 UUID v5를 반환합니다."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"phantomgate.{namespace}.{value}"))


def _det_uuid_raw(value: str) -> str:
    """bundle ID용 결정론적 UUID v5 (RFC 4122 준수)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"phantomgate.bundle.{value}"))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _risk_to_confidence(risk: float) -> int:
    """위험 점수(0-1)를 STIX 신뢰도(0-100)로 변환합니다."""
    return min(100, int(risk * 100))
