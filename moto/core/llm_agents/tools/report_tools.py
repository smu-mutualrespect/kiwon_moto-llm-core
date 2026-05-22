from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moto.core.llm_agents.runtime.provider import call_gpt_api_with_meta
from moto.core.llm_agents.tools.state_tools import (
    get_full_action_log,
    get_idle_sessions,
    get_session_state_snapshot,
    mark_session_reported,
)

_REPORT_DIR = Path(os.getenv("MOTO_HONEYPOT_REPORT_DIR", "reports"))
_IDLE_SECONDS = float(os.getenv("MOTO_HONEYPOT_SESSION_TIMEOUT", "300"))
_REPORT_MODEL = os.getenv("MOTO_LLM_REPORT_MODEL") or None
_REPORT_MAX_TOKENS = int(os.getenv("MOTO_LLM_REPORT_MAX_TOKENS", "3000"))


def generate_attack_report(session_id: str) -> str:
    """Generate a TTP-mapped Markdown report for the session and save it to disk.

    Returns the path of the saved report file, or "" if no activity was recorded.
    """
    state = get_session_state_snapshot(session_id)
    action_log = get_full_action_log(session_id)
    if not action_log:
        return ""

    prompt = _build_report_prompt(session_id, state, action_log)
    report_md, _ = call_gpt_api_with_meta(
        prompt,
        model=_REPORT_MODEL,
        timeout=120.0,
        max_tokens=_REPORT_MAX_TOKENS,
    )

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _REPORT_DIR / f"{session_id[:16]}_{timestamp}.md"
    path.write_text(report_md, encoding="utf-8")
    mark_session_reported(session_id)
    return str(path)


def start_report_watcher(idle_seconds: float = _IDLE_SECONDS) -> None:
    """Start a background daemon thread that auto-generates reports for idle sessions.

    A session is considered finished when it has had no new requests for
    idle_seconds (default 300 s / 5 min, configurable via MOTO_HONEYPOT_SESSION_TIMEOUT).
    """
    def _watch() -> None:
        while True:
            time.sleep(60)
            for sid in get_idle_sessions(idle_seconds):
                try:
                    generate_attack_report(sid)
                except Exception:
                    pass

    threading.Thread(target=_watch, daemon=True, name="honeypot-report-watcher").start()


def _build_report_prompt(
    session_id: str,
    state: dict[str, Any],
    action_log: list[dict[str, Any]],
) -> str:
    start_time = state.get("session_start_time", "unknown")
    risk_score = float(state.get("risk_score", 0.0))
    exposed_assets = state.get("exposed_assets", [])
    account_id = state.get("consistency_locks", {}).get("account_id", "unknown")

    total_ops = len(action_log)
    services_used = sorted({e["service"] for e in action_log})
    phases_seen = list(dict.fromkeys(e["phase"] for e in action_log))

    # Compute approximate session duration
    if total_ops >= 2:
        first_ts = action_log[0]["timestamp"]
        last_ts = action_log[-1]["timestamp"]
        duration_str = f"{first_ts} → {last_ts}"
    else:
        duration_str = start_time

    # Build timeline
    timeline_lines: list[str] = []
    for entry in action_log:
        ts = entry["timestamp"]
        phase = entry["phase"]
        svc = entry["service"]
        op = entry["operation"]
        src = entry["source"]
        risk = entry.get("risk_score", 0.0)
        assets = entry.get("new_assets", [])
        line = f"[{ts}] [{phase}] {svc}:{op} ({src}) risk={risk:.2f}"
        if assets:
            line += f"  → {', '.join(str(a) for a in assets[:3])}"
        timeline_lines.append(line)

    assets_block = "\n".join(f"- {a}" for a in exposed_assets[:60]) or "None detected"

    return f"""당신은 AWS 클라우드 허니팟 공격 세션을 분석하는 위협 인텔리전스 분석가입니다.
아래 데이터를 분석하여 완전하고 전문적인 Markdown 위협 인텔리전스 보고서를 **한국어**로 작성하세요.

## Session Metadata
- Session ID: {session_id}
- Target Account (fake): {account_id}
- Session Start: {start_time}
- Activity Window: {duration_str}
- Total API Operations: {total_ops}
- Services Probed: {", ".join(services_used)}
- Attack Phase Progression: {" → ".join(phases_seen)}
- Final Risk Score: {risk_score:.2f} / 1.00

## Full Attack Timeline
{chr(10).join(timeline_lines)}

## Exposed / Discovered Assets
{assets_block}

---

아래 형식의 Markdown 보고서를 **한국어**로 작성하세요. 타임라인의 실제 작업을 구체적으로 언급하세요.

# 위협 인텔리전스 보고서 — {session_id[:16]}

## 요약
(공격자가 누구이며 무엇을 했는지, 전반적인 심각도를 2–3문장으로 요약)

## 공격 타임라인
(타임스탬프 순서대로 주요 이벤트 목록, 단계별로 그룹화)

## MITRE ATT&CK TTP 매핑
| 전술(Tactic) | 기법 ID | 기법 이름 | 관찰된 작업 |
|------------|--------|----------|-----------|
(관찰된 각 기법마다 행 추가)

## 노출 자산 및 영향 평가
(발견된 리소스 목록, 민감도 및 잠재적 피해 범위 평가)

## 위험도 평가
**전체 심각도: 심각 / 높음 / 중간 / 낮음**
(타임라인 근거와 함께 판단 이유 설명)

## 권고 사항
(구체적이고 실행 가능한 방어 조치 3–5개)
"""
