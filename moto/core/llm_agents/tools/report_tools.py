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

    return f"""You are a threat intelligence analyst reviewing an AWS cloud honeypot attack session.
Analyze the data below and produce a complete, professional Markdown threat intelligence report.

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

Write the following Markdown report. Be specific and reference actual operations from the timeline.

# Threat Intelligence Report — {session_id[:16]}

## Executive Summary
(2–3 sentences covering who attacked, what they did, and the overall severity)

## Attack Timeline
(chronological list of key events with timestamps; group by phase)

## MITRE ATT&CK TTP Mapping
| Tactic | Technique ID | Technique Name | Observed Operation |
|--------|-------------|----------------|-------------------|
(fill in rows for each distinct technique observed)

## Exposed Assets & Impact Assessment
(list discovered resources; assess sensitivity and potential blast radius)

## Risk Assessment
**Overall Severity: Critical / High / Medium / Low**
(justify with evidence from the timeline)

## Recommendations
(3–5 concrete, actionable defensive measures)
"""
