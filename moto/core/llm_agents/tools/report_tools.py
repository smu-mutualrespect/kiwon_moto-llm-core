from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moto.core.llm_agents.runtime.provider import call_gpt_api_with_meta
from moto.core.llm_agents.tools.attack_db import get_technique
from moto.core.llm_agents.tools.state_tools import (
    get_full_action_log,
    get_idle_sessions,
    get_session_state_snapshot,
    is_session_reported,
    mark_session_reported,
)
from moto.core.llm_agents.tools.stix_export import generate_stix_bundle

_log = logging.getLogger(__name__)

_REPORT_DIR = Path(os.getenv("MOTO_HONEYPOT_REPORT_DIR", "reports"))
_IDLE_SECONDS = float(os.getenv("MOTO_HONEYPOT_SESSION_TIMEOUT", "300"))
_REPORT_MODEL = os.getenv("MOTO_LLM_REPORT_MODEL") or None
_REPORT_MAX_TOKENS = int(os.getenv("MOTO_LLM_REPORT_MAX_TOKENS", "5000"))

# TLP(Traffic Light Protocol) 등급 — 조직 설정에 따라 변경
_TLP_LEVEL = os.getenv("MOTO_HONEYPOT_TLP", "TLP:AMBER")

# 작업(operation) → MITRE ATT&CK 기법 ID 정적 매핑
# ref: https://attack.mitre.org/matrices/enterprise/cloud/
_OP_TO_TECHNIQUE: dict[tuple[str, str], str] = {
    ("sts", "GetCallerIdentity"): "T1087.004",
    ("iam", "ListUsers"): "T1087.004",
    ("iam", "ListRoles"): "T1069.003",
    ("iam", "ListPolicies"): "T1069.003",
    ("iam", "GetPolicy"): "T1069.003",
    ("iam", "GetPolicyVersion"): "T1069.003",
    ("iam", "ListAttachedUserPolicies"): "T1069.003",
    ("iam", "ListAttachedRolePolicies"): "T1069.003",
    ("iam", "AttachUserPolicy"): "T1098",
    ("iam", "AttachRolePolicy"): "T1098",
    ("iam", "CreateUser"): "T1136.003",
    ("iam", "CreateAccessKey"): "T1098",
    ("iam", "PutUserPolicy"): "T1098",
    ("ec2", "DescribeInstances"): "T1580",
    ("ec2", "DescribeSecurityGroups"): "T1580",
    ("ec2", "DescribeVpcs"): "T1580",
    ("ec2", "DescribeSubnets"): "T1580",
    ("s3", "ListBuckets"): "T1619",
    ("s3", "GetBucketPolicy"): "T1619",
    ("s3", "GetBucketAcl"): "T1619",
    ("s3", "GetObject"): "T1530",
    ("s3", "PutObject"): "T1537",
    ("secretsmanager", "ListSecrets"): "T1526",
    ("secretsmanager", "GetSecretValue"): "T1555.006",
    ("secretsmanager", "DescribeSecret"): "T1526",
    ("ssm", "GetParameter"): "T1552.001",
    ("ssm", "GetParameters"): "T1552.001",
    ("lambda", "ListFunctions"): "T1526",
    ("lambda", "GetFunction"): "T1526",
    ("eks", "ListClusters"): "T1580",
    ("rds", "DescribeDBInstances"): "T1580",
    ("cloudtrail", "DescribeTrails"): "T1580",
    ("cloudtrail", "StopLogging"): "T1562.008",
    ("guardduty", "ListDetectors"): "T1580",
    ("guardduty", "DeleteDetector"): "T1562",
}


def generate_attack_report(session_id: str) -> str:
    """세션의 TTP 매핑 Markdown 보고서를 생성하고 디스크에 저장합니다.

    저장된 보고서 파일 경로를 반환하며, 기록된 활동이 없으면 빈 문자열을 반환합니다.
    """
    if is_session_reported(session_id):
        return ""
    state = get_session_state_snapshot(session_id)
    action_log = get_full_action_log(session_id)
    if not action_log:
        return ""

    timing = _compute_timing_analysis(action_log)
    iocs = _extract_iocs(session_id, state, action_log, timing)
    ttp_map = _map_ttps(action_log)

    prompt = _build_report_prompt(session_id, state, action_log, timing, iocs, ttp_map)
    report_md, llm_meta = call_gpt_api_with_meta(
        prompt,
        model=_REPORT_MODEL,
        timeout=120.0,
        max_tokens=_REPORT_MAX_TOKENS,
    )

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{session_id[:16]}_{timestamp}"

    md_path = _REPORT_DIR / f"{base}.md"
    md_path.write_text(report_md, encoding="utf-8")

    # 보고서 생성 메트릭 로그
    usage = llm_meta.get("usage") or {}
    _log.info(
        "보고서 생성 완료 | session=%s | model=%s | "
        "input_tokens=%s | output_tokens=%s | duration_ms=%s | path=%s",
        session_id[:16],
        llm_meta.get("model"),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        llm_meta.get("duration_ms"),
        md_path,
    )

    # 메트릭 파일 저장 (측정용)
    metrics_path = _REPORT_DIR / f"{base}.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "session_id": session_id[:16],
                "model": llm_meta.get("model"),
                "prompt_chars": len(prompt),
                "output_chars": len(report_md),
                "usage": usage,
                "duration_ms": llm_meta.get("duration_ms"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ATT&CK Navigator 레이어 JSON
    navigator_path = _REPORT_DIR / f"{base}_navigator.json"
    navigator_path.write_text(
        json.dumps(
            generate_navigator_layer(session_id, ttp_map), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    # STIX 2.1 번들 (SIEM/OpenCTI/MISP import용)
    stix_path = _REPORT_DIR / f"{base}.stix.json"
    stix_bundle = generate_stix_bundle(session_id, iocs, ttp_map, state, action_log)
    stix_path.write_text(
        json.dumps(stix_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mark_session_reported(session_id)
    return str(md_path)


def generate_navigator_layer(
    session_id: str,
    ttp_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """관찰된 TTP를 MITRE ATT&CK Navigator 레이어 JSON으로 반환합니다.

    https://mitre-attack.github.io/attack-navigator/ 에서 import 해서 시각화할 수 있습니다.
    """
    if ttp_map is None:
        action_log = get_full_action_log(session_id)
        ttp_map = _map_ttps(action_log)

    techniques = []
    for tech_id, info in ttp_map.items():
        score = min(100, len(info["evidence"]) * 25)  # evidence 수에 비례
        techniques.append(
            {
                "techniqueID": tech_id,
                "score": score,
                "color": _score_to_color(score),
                "comment": f"관찰 횟수: {len(info['evidence'])}회 | 작업: {', '.join(info['evidence'][:3])}",
                "enabled": True,
            }
        )

    return {
        "name": f"PhantomGate — {session_id[:16]}",
        "versions": {"attack": "14", "navigator": "4.9", "layer": "4.5"},
        "domain": "enterprise-attack",
        "description": f"PhantomGate 허니팟 세션 {session_id} 에서 관찰된 TTP",
        "filters": {"platforms": ["AWS"]},
        "gradient": {
            "colors": ["#ffffff", "#ff6666"],
            "minValue": 0,
            "maxValue": 100,
        },
        "techniques": techniques,
    }


_watcher_started = False
_watcher_lock = threading.Lock()


def start_report_watcher(idle_seconds: float = _IDLE_SECONDS) -> None:
    """비활성 세션에 대해 자동으로 보고서를 생성하는 백그라운드 데몬 스레드를 시작합니다.

    idle_seconds(기본 300초 / 5분) 동안 요청이 없으면 세션 종료로 판단합니다.
    MOTO_HONEYPOT_SESSION_TIMEOUT 환경변수로 조정 가능합니다.
    중복 호출에 안전합니다 — 스레드는 최초 1회만 시작됩니다.
    """
    global _watcher_started
    with _watcher_lock:
        if _watcher_started:
            return
        _watcher_started = True

    def _watch() -> None:
        while True:
            time.sleep(60)
            for sid in get_idle_sessions(idle_seconds):
                try:
                    generate_attack_report(sid)
                except Exception:
                    _log.exception("보고서 생성 실패 — session_id=%s", sid)

    threading.Thread(target=_watch, daemon=True, name="honeypot-report-watcher").start()


# ──────────────────────────────────────────────
# 내부 헬퍼 함수들
# ──────────────────────────────────────────────


def _compute_timing_analysis(action_log: list[dict[str, Any]]) -> dict[str, Any]:
    """요청 간격 분석으로 자동화 도구 사용 여부를 추정합니다."""
    if len(action_log) < 2:
        return {}
    timestamps: list[datetime] = []
    for entry in action_log:
        try:
            ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
            timestamps.append(ts)
        except Exception:
            pass
    if len(timestamps) < 2:
        return {}
    intervals = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    avg = sum(intervals) / len(intervals)
    deviation = (sum((x - avg) ** 2 for x in intervals) / len(intervals)) ** 0.5
    total_sec = (timestamps[-1] - timestamps[0]).total_seconds()
    # 평균 간격 10초 이하이거나, 충분한 샘플(5개 이상 간격)에서 편차가 매우 작으면 자동화 의심
    # 2개짜리 세션은 interval이 1개뿐이라 deviation=0 → 오탐 방지를 위해 샘플 수 조건 추가
    is_automated = avg <= 10.0 or (len(intervals) >= 5 and deviation < 2.0)
    return {
        "avg_interval_sec": round(avg, 1),
        "min_interval_sec": round(min(intervals), 1),
        "max_interval_sec": round(max(intervals), 1),
        "std_deviation_sec": round(deviation, 1),
        "total_duration_sec": round(total_sec, 1),
        "is_automated": is_automated,
    }


def _extract_iocs(
    session_id: str,
    state: dict[str, Any],
    action_log: list[dict[str, Any]],
    timing: dict[str, Any],
) -> dict[str, Any]:
    """액션 로그와 세션 상태에서 침해지표(IOC)를 추출합니다."""
    iocs: dict[str, Any] = {}

    # 자격증명 식별자
    # AKIA = 장기 IAM User Key, ASIA = 임시 STS 세션 토큰
    # AROA는 Role Principal ID (ARN 내부 식별자)이므로 자격증명으로 분류하지 않음
    if session_id.startswith(("AKIA", "ASIA")):
        iocs["access_key_id"] = session_id
        iocs["credential_type"] = (
            "장기 자격증명 (IAM User Key)"
            if session_id.startswith("AKIA")
            else "임시 자격증명 (STS Session Token)"
        )
    else:
        iocs["session_identifier"] = session_id

    # 타겟 계정
    account_id = state.get("consistency_locks", {}).get("account_id")
    if account_id:
        iocs["target_account_id"] = account_id

    # 탐색한 서비스 목록
    iocs["targeted_services"] = sorted({e["service"] for e in action_log})

    # 관찰된 전체 작업 순서 (중복 제거)
    iocs["observed_operations"] = list(
        dict.fromkeys(
            f"{e['service']}:{e['operation']}"
            for e in action_log
        )
    )

    # 발견한 자산 ARN
    iocs["discovered_arns"] = [
        a for a in state.get("exposed_assets", []) if str(a).startswith("arn:aws:")
    ][:20]

    # 자동화 도구 징후
    if timing.get("is_automated"):
        iocs["automation_indicator"] = (
            f"평균 요청 간격 {timing['avg_interval_sec']}초 / 표준편차 {timing['std_deviation_sec']}초 "
            f"— 자동화 스크립트 또는 공격 도구 사용 의심"
        )

    # 공격 지속시간
    if timing.get("total_duration_sec"):
        iocs["attack_duration"] = (
            f"{timing['total_duration_sec']}초 ({round(timing['total_duration_sec'] / 60, 1)}분)"
        )

    return iocs


def _map_ttps(action_log: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """관찰된 작업을 MITRE ATT&CK 기법으로 매핑합니다.

    attack_db에서 기법 상세 정보(description, detection, url, platforms, data_sources)를 조회합니다.
    반환 형식: {technique_id: {tactic, name, description, detection, url, evidence, confidence}}
    """
    result: dict[str, dict[str, Any]] = {}
    for entry in action_log:
        key = (entry["service"], entry["operation"])
        tech_id = _OP_TO_TECHNIQUE.get(key)
        if not tech_id:
            continue
        op_str = f"{entry['service']}:{entry['operation']} [{entry['timestamp']}]"
        if tech_id not in result:
            # attack_db에서 풍부한 메타데이터 조회 (캐시/오프라인 폴백 자동 처리)
            meta = get_technique(tech_id)
            result[tech_id] = {
                "tactic": meta.get("tactic", "Unknown"),
                "name": meta.get("name", tech_id),
                "description": meta.get("description", ""),
                "detection": meta.get("detection", ""),
                "url": meta.get(
                    "url",
                    f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}/",
                ),
                "platforms": meta.get("platforms", []),
                "data_sources": meta.get("data_sources", []),
                "evidence": [],
            }
        if op_str not in result[tech_id]["evidence"]:
            result[tech_id]["evidence"].append(op_str)

    # IC 스타일 신뢰도:
    #   High   — 동일 기법 2회 이상 직접 관찰
    #   Medium — 1회 관찰, 정황 증거
    #   Low    — 간접 추론
    for info in result.values():
        ev_count = len(info["evidence"])
        info["confidence"] = (
            "High" if ev_count >= 2 else "Medium" if ev_count == 1 else "Low"
        )

    return result


def _build_report_prompt(
    session_id: str,
    state: dict[str, Any],
    action_log: list[dict[str, Any]],
    timing: dict[str, Any],
    iocs: dict[str, Any],
    ttp_map: dict[str, dict[str, Any]],
) -> str:
    start_time = state.get("session_start_time", "unknown")
    account_id = state.get("consistency_locks", {}).get("account_id", "unknown")

    total_ops = len(action_log)
    services_used = sorted({e["service"] for e in action_log})
    phases_seen = list(dict.fromkeys(e["phase"] for e in action_log))

    # 타임라인 (증거 연결 — 점수 제외)
    timeline_lines: list[str] = []
    for i, entry in enumerate(action_log, 1):
        assets = entry.get("new_assets", [])
        line = (
            f"[{i:02d}] {entry['timestamp']} | [{entry['phase']}] "
            f"{entry['service']}:{entry['operation']} | 출처={entry['source']}"
        )
        if assets:
            line += f" | 발견자산={', '.join(str(a) for a in assets[:2])}"
        timeline_lines.append(line)

    # TTP 사전 매핑 블록 (탐지 가이드 포함)
    ttp_lines: list[str] = []
    for tech_id, info in ttp_map.items():
        evidence_str = " / ".join(info["evidence"][:3])
        detection = info.get("detection", "")[:200]
        ttp_lines.append(
            f"- {tech_id} | {info['tactic']} | {info['name']} | "
            f"신뢰도={info['confidence']} | 증거={evidence_str}"
            + (f" | 탐지가이드={detection}" if detection else "")
        )

    # IOC 블록
    ioc_lines = [f"- {k}: {v}" for k, v in iocs.items()]

    # 타이밍 분석
    auto_note = ""
    if timing:
        auto_note = (
            f"평균 요청 간격: {timing.get('avg_interval_sec')}초, "
            f"표준편차: {timing.get('std_deviation_sec')}초, "
            f"전체 세션 지속: {timing.get('total_duration_sec')}초, "
            f"자동화 의심: {'예' if timing.get('is_automated') else '아니오'}"
        )

    return f"""당신은 AWS 클라우드 허니팟에서 수집된 공격 세션을 분석하는 위협 인텔리전스 분석가입니다.
이 보고서의 목적은 "공격자가 AWS 환경에서 어떤 흐름으로 움직이는가"를 기록하는 것입니다.
점수나 피해 평가 없이, 관찰된 행동 순서와 TTP를 중심으로 **한국어**로 작성하세요.

══════════════ 관찰 데이터 ══════════════

[세션 정보]
- 세션 ID: {session_id}
- 허니팟 계정 ID (가상): {account_id}
- 세션 시작: {start_time}
- 총 API 호출 수: {total_ops}
- 탐색한 서비스: {", ".join(services_used)}
- 관찰된 공격 단계 흐름: {" → ".join(phases_seen)}

[요청 타이밍 분석]
{auto_note or "데이터 없음"}

[전체 공격 타임라인 (증거 번호 포함)]
{chr(10).join(timeline_lines)}

[TTP 매핑]
{chr(10).join(ttp_lines) or "매핑 없음"}

[공격자 식별 지표 (IOC)]
{chr(10).join(ioc_lines)}

══════════════ 보고서 형식 ══════════════

아래 구조를 **반드시** 따르세요. 각 섹션에서 위 데이터의 실제 증거 번호([01], [02] 등)를 인용하세요.
점수, 위험도 수치, 피해 평가는 절대 포함하지 마세요.

---

# AWS 허니팟 공격 흐름 분석 보고서

**문서 등급**: {_TLP_LEVEL}
**작성일**: {datetime.now(timezone.utc).strftime("%Y년 %m월 %d일")}
**세션**: {session_id[:16]}

---

## 1. 세션 요약

| 항목 | 내용 |
|------|------|
| 공격자 식별자 | |
| 자격증명 유형 | |
| 세션 시작 시각 | |
| 세션 지속 시간 | |
| 탐색한 서비스 수 | |
| 총 API 호출 수 | |
| 자동화 도구 사용 여부 | |

## 2. 공격 흐름 분석

(허니팟에서 관찰된 공격자의 행동을 단계별로 서술하세요.
각 단계가 왜 수행됐는지, 다음 단계로 어떻게 이어지는지 흐름 중심으로 작성합니다.
증거 번호([01], [02] 등)를 각 단계마다 인용하세요.)

## 3. MITRE ATT&CK TTP 매핑

| 전술 | 기법 ID | 기법 이름 | 공격자가 한 행위 | 증거 | 신뢰도 |
|------|--------|----------|----------------|------|--------|
(TTP 매핑 데이터를 기반으로 작성. Procedure는 이번 세션에서 구체적으로 관찰된 행위로 서술)

## 4. 공격자 식별 지표 (IOC)

### 4-1. 자격증명 지표
### 4-2. 탐색한 자산 목록
### 4-3. 행위 패턴 지표

## 5. 탐지 권고

(이 공격 패턴이 실제 AWS 환경에서 발생했을 때 어떻게 탐지할 수 있는지 서술하세요.
CloudTrail, GuardDuty 등 AWS 네이티브 탐지 수단을 중심으로 작성합니다.
각 권고는 위 TTP 또는 타임라인 증거와 연결해서 서술하세요.)

## 6. 결론 및 보안 대책 권고

### 공격자 특징 요약
(이 세션에서 관찰된 공격자의 특징을 서술하세요 — 자격증명 유형, 탐색 범위, 행동 패턴, 자동화 여부 등)

### 실제 환경이었다면
(허니팟이 아니라 실제 AWS 운영 환경이었을 경우를 가정하여, 이 공격 흐름이 어떤 결과로 이어질 수 있었는지 서술하세요.)

### 보안 대책 권고
(위 공격 패턴을 방어하기 위한 AWS 보안 설정 및 정책을 구체적으로 권고하세요.
각 권고마다 근거가 되는 공식 레퍼런스를 아래 형식으로 반드시 포함하세요:
> 참고: [문서명](URL) — 한 줄 설명)

반드시 포함할 레퍼런스 출처:
- AWS 공식 보안 문서 (docs.aws.amazon.com/security)
- AWS Well-Architected Framework Security Pillar
- CIS AWS Foundations Benchmark
- NIST SP 800-53 또는 CSF
- MITRE ATT&CK for Cloud (attack.mitre.org)

---
*본 보고서는 PhantomGate 허니팟 시스템에 의해 자동 생성되었습니다.*
"""


def _score_to_color(score: int) -> str:
    if score >= 75:
        return "#ff0000"
    if score >= 50:
        return "#ff6600"
    if score >= 25:
        return "#ffaa00"
    return "#ffdd00"
