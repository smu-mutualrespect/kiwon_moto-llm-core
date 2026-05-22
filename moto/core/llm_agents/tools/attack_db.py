"""MITRE ATT&CK Enterprise 기법 데이터베이스.

첫 호출 시 MITRE GitHub에서 enterprise-attack STIX JSON을 다운로드하고,
필요한 필드만 추출해 로컬 캐시 파일(_CACHE_PATH)에 저장합니다.
이후 호출은 캐시에서만 읽으므로 네트워크 불필요.

오프라인 환경 또는 다운로드 실패 시 내장 정적 데이터로 폴백합니다.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ATT&CK enterprise STIX JSON (GitHub raw)
_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data"
    "/master/enterprise-attack/enterprise-attack-14.1.json"
)

_CACHE_PATH = Path(
    os.getenv("MOTO_ATTACK_DB_CACHE", str(Path.home() / ".cache" / "phantomgate" / "attack_db.json"))
)

_db: dict[str, dict[str, Any]] | None = None
_lock = threading.Lock()


def get_technique(technique_id: str) -> dict[str, Any]:
    """기법 ID(예: 'T1555.006')로 ATT&CK 메타데이터를 반환합니다.

    반환 형식::

        {
            "id": "T1555.006",
            "name": "...",
            "tactic": "Credential Access",
            "description": "...",
            "detection": "...",
            "url": "https://attack.mitre.org/techniques/...",
            "platforms": ["IaaS", ...],
            "data_sources": [...],
        }

    데이터가 없으면 id/name만 담긴 최소 dict을 반환합니다.
    """
    db = _get_db()
    return db.get(technique_id, {"id": technique_id, "name": technique_id, "tactic": "Unknown"})


def refresh(force: bool = False) -> int:
    """ATT&CK 데이터를 MITRE에서 다시 내려받아 캐시를 갱신합니다.

    성공 시 로드된 기법 수, 실패 시 0을 반환합니다.
    """
    global _db
    with _lock:
        built = _build_db(force=force)
        if built:
            _db = built
            return len(built)
        return 0


# ──────────────────────────────────────────────
# 내부 구현
# ──────────────────────────────────────────────

def _get_db() -> dict[str, dict[str, Any]]:
    global _db
    if _db is not None:
        return _db
    with _lock:
        if _db is None:
            _db = _build_db() or _static_fallback()
    return _db


def _build_db(force: bool = False) -> dict[str, dict[str, Any]] | None:
    """캐시 파일에서 로드하거나, 없으면 MITRE에서 다운로드해 빌드합니다."""
    if not force and _CACHE_PATH.exists():
        try:
            data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                _log.debug("ATT&CK DB loaded from cache (%d techniques)", len(data))
                return data
        except Exception as exc:
            _log.warning("ATT&CK cache read failed: %s", exc)

    return _download_and_build()


def _download_and_build() -> dict[str, dict[str, Any]] | None:
    """MITRE GitHub에서 STIX JSON을 다운로드하고 compact dict으로 변환합니다."""
    try:
        import requests as _req  # noqa: PLC0415

        _log.info("Downloading ATT&CK enterprise data from MITRE GitHub…")
        resp = _req.get(_STIX_URL, timeout=30, stream=True)
        resp.raise_for_status()
        stix_data = resp.json()
    except Exception as exc:
        _log.warning("ATT&CK download failed (%s) — using fallback", exc)
        return None

    db: dict[str, dict[str, Any]] = {}
    for obj in stix_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("x_mitre_deprecated") or obj.get("revoked"):
            continue

        ext_refs = obj.get("external_references", [])
        att_id = next(
            (r["external_id"] for r in ext_refs if r.get("source_name") == "mitre-attack"),
            None,
        )
        att_url = next(
            (r.get("url", "") for r in ext_refs if r.get("source_name") == "mitre-attack"),
            "",
        )
        if not att_id:
            continue

        # 전술 목록
        kill_chain = obj.get("kill_chain_phases", [])
        tactics = [
            p["phase_name"].replace("-", " ").title()
            for p in kill_chain
            if p.get("kill_chain_name") == "mitre-attack"
        ]

        # detection 필드 추출
        detection = obj.get("x_mitre_detection", "")
        if len(detection) > 500:
            detection = detection[:497] + "…"

        description = obj.get("description", "")
        if len(description) > 600:
            description = description[:597] + "…"

        db[att_id] = {
            "id": att_id,
            "name": obj.get("name", att_id),
            "tactic": tactics[0] if tactics else "Unknown",
            "tactics": tactics,
            "description": description,
            "detection": detection,
            "url": att_url,
            "platforms": obj.get("x_mitre_platforms", []),
            "data_sources": obj.get("x_mitre_data_sources", []),
            "is_subtechnique": "." in att_id,
        }

    if not db:
        return None

    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
        _log.info("ATT&CK DB cached (%d techniques) → %s", len(db), _CACHE_PATH)
    except Exception as exc:
        _log.warning("Failed to write ATT&CK cache: %s", exc)

    return db


def _static_fallback() -> dict[str, dict[str, Any]]:
    """다운로드 실패 시 핵심 클라우드 기법만 담은 내장 데이터를 반환합니다."""
    return {
        "T1087.004": {
            "id": "T1087.004", "name": "Account Discovery: Cloud Account",
            "tactic": "Discovery",
            "description": "공격자가 클라우드 환경의 계정 목록을 수집합니다.",
            "detection": "GetCallerIdentity, ListUsers 등 IAM 열거 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1087/004/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Enumeration"],
        },
        "T1069.003": {
            "id": "T1069.003", "name": "Permission Groups Discovery: Cloud Groups",
            "tactic": "Discovery",
            "description": "공격자가 클라우드 권한 그룹 및 역할을 열거합니다.",
            "detection": "ListRoles, ListPolicies 등 IAM 탐색 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1069/003/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Enumeration"],
        },
        "T1098": {
            "id": "T1098", "name": "Account Manipulation",
            "tactic": "Privilege Escalation",
            "description": "공격자가 계정 또는 권한을 수정해 접근을 유지하거나 확대합니다.",
            "detection": "AttachUserPolicy, PutRolePolicy 등 IAM 쓰기 작업 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1098/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Modification"],
        },
        "T1136.003": {
            "id": "T1136.003", "name": "Create Account: Cloud Account",
            "tactic": "Persistence",
            "description": "공격자가 클라우드 계정을 신규 생성해 지속성을 확보합니다.",
            "detection": "CreateUser, CreateRole API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1136/003/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Modification"],
        },
        "T1580": {
            "id": "T1580", "name": "Cloud Infrastructure Discovery",
            "tactic": "Discovery",
            "description": "공격자가 클라우드 인프라 구성 요소를 열거합니다.",
            "detection": "DescribeInstances, DescribeVpcs 등 Describe* API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1580/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Enumeration"],
        },
        "T1619": {
            "id": "T1619", "name": "Cloud Storage Object Discovery",
            "tactic": "Discovery",
            "description": "공격자가 클라우드 스토리지 버킷/객체를 열거합니다.",
            "detection": "ListBuckets, GetBucketPolicy 등 S3 탐색 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1619/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Storage: Cloud Storage Enumeration"],
        },
        "T1530": {
            "id": "T1530", "name": "Data from Cloud Storage Object",
            "tactic": "Collection",
            "description": "공격자가 클라우드 스토리지에서 데이터를 수집합니다.",
            "detection": "GetObject, SelectObjectContent 등 S3 읽기 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1530/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Storage: Cloud Storage Access"],
        },
        "T1537": {
            "id": "T1537", "name": "Transfer Data to Cloud Account",
            "tactic": "Exfiltration",
            "description": "공격자가 데이터를 외부 클라우드 계정으로 전송합니다.",
            "detection": "PutObject, 교차 계정 복사 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1537/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Storage: Cloud Storage Modification"],
        },
        "T1526": {
            "id": "T1526", "name": "Cloud Service Discovery",
            "tactic": "Discovery",
            "description": "공격자가 사용 가능한 클라우드 서비스 목록을 수집합니다.",
            "detection": "ListSecrets, ListFunctions 등 List* API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1526/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Enumeration"],
        },
        "T1555.006": {
            "id": "T1555.006",
            "name": "Credentials from Password Stores: Cloud Secrets Management Services",
            "tactic": "Credential Access",
            "description": "공격자가 클라우드 시크릿 관리 서비스에서 자격증명을 탈취합니다.",
            "detection": "GetSecretValue, GetParameter 등 비밀 조회 API 모니터링. 비정상 주체나 시간대의 호출에 경보 설정.",
            "url": "https://attack.mitre.org/techniques/T1555/006/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Enumeration"],
        },
        "T1552.001": {
            "id": "T1552.001", "name": "Unsecured Credentials: Credentials In Files",
            "tactic": "Credential Access",
            "description": "공격자가 파일 또는 설정에서 평문 자격증명을 획득합니다.",
            "detection": "SSM GetParameter, S3 GetObject 등 설정 조회 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1552/001/",
            "platforms": ["IaaS"], "data_sources": ["File: File Access"],
        },
        "T1562.008": {
            "id": "T1562.008", "name": "Impair Defenses: Disable Cloud Logs",
            "tactic": "Defense Evasion",
            "description": "공격자가 CloudTrail 등 로깅을 비활성화해 탐지를 방해합니다.",
            "detection": "StopLogging, DeleteTrail API 모니터링. 로그 중단 알람 설정.",
            "url": "https://attack.mitre.org/techniques/T1562/008/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Modification"],
        },
        "T1562": {
            "id": "T1562", "name": "Impair Defenses",
            "tactic": "Defense Evasion",
            "description": "공격자가 보안 도구나 로깅을 비활성화합니다.",
            "detection": "보안 설정 변경 API 모니터링.",
            "url": "https://attack.mitre.org/techniques/T1562/",
            "platforms": ["IaaS"], "data_sources": ["Cloud Service: Cloud Service Modification"],
        },
    }
