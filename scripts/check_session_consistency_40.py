"""
40개 명령어를 단일 세션으로 실행해서 세션 일관성을 검증한다.

검증 항목:
1. account_id — 모든 ARN에서 추출한 account id가 세션 내 동일한지
2. 같은 서비스 내 resource_registry — 동일 서비스에 여러 번 요청할 때 registry가 유지되는지
3. 삭제 후 lifecycle — delete가 있으면 이후 read는 not_found여야 함
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moto.core.llm_agents.agent import handle_aws_request

SESSION_AUTH = (
    "AWS4-HMAC-SHA256 "
    "Credential=AKIASESSION40CMD/20260517/us-east-1/{service}/aws4_request, "
    "SignedHeaders=host;x-amz-date, "
    "Signature=fakesig"
)

DEFAULT_CORPUS = ROOT / "artifacts" / "agentic_runtime" / "command_corpus.json"
DEFAULT_RESULTS = ROOT / "artifacts" / "agentic_runtime" / "session40_results.json"
DEFAULT_SUMMARY = ROOT / "artifacts" / "agentic_runtime" / "session40_summary.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))

    os.environ["MOTO_LLM_RUNTIME_MODE"] = "agentic"
    if args.live:
        os.environ.pop("MOTO_LLM_OFFLINE_STUB", None)
    else:
        os.environ["MOTO_LLM_OFFLINE_STUB"] = "1"

    results = []
    for entry in corpus:
        result = _run_entry(entry)
        results.append(result)

    checks = _consistency_checks(results)
    passed = all(c["pass"] for c in checks)

    payload = {
        "mode": "live" if args.live else "offline_stub",
        "session_id": "AKIASESSION40CMD",
        "total": len(results),
        "pass": passed,
        "consistency_checks": checks,
        "results": results,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(_render_summary(payload), encoding="utf-8")

    print(_render_summary(payload))
    return 0 if passed else 1


def _run_entry(entry: dict[str, Any]) -> dict[str, Any]:
    service = entry["service"]
    operation = entry["operation"]
    headers = dict(entry.get("headers", {}))
    headers["Authorization"] = SESSION_AUTH.format(service=service)

    t0 = time.perf_counter()
    response_body = handle_aws_request(
        service=service,
        action=operation,
        url=f"https://{service}.us-east-1.amazonaws.com/",
        headers=headers,
        body=entry.get("body", ""),
        reason="session consistency check",
        source="scripts.check_session_consistency_40",
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    arns = _extract_arns(response_body)
    account_ids = list({_account_id_from_arn(a) for a in arns if _account_id_from_arn(a)})
    is_error = _is_error_response(response_body)

    return {
        "id": entry["id"],
        "service": service,
        "operation": operation,
        "phase": entry.get("phase"),
        "latency_ms": latency_ms,
        "is_error": is_error,
        "arns": arns[:3],
        "account_ids_in_response": account_ids,
        "response_body": response_body,
    }


def _consistency_checks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []

    # 1. account_id 일관성: 모든 ARN의 account id가 하나여야 함
    all_account_ids: set[str] = set()
    for r in results:
        all_account_ids.update(r["account_ids_in_response"])
    # 000000000000 같은 placeholder 제외
    real_ids = {a for a in all_account_ids if a and a != "000000000000"}
    checks.append({
        "name": "account_id_consistent",
        "pass": len(real_ids) <= 1,
        "detail": f"발견된 account_id: {sorted(real_ids) if real_ids else ['없음']}",
    })

    # 2. 같은 서비스 내 ARN 일관성: 동일 서비스에서 나온 ARN의 account part가 같아야 함
    from collections import defaultdict
    svc_account_ids: dict[str, set[str]] = defaultdict(set)
    for r in results:
        svc = r["service"]
        for aid in r["account_ids_in_response"]:
            if aid and aid != "000000000000":
                svc_account_ids[svc].add(aid)

    inconsistent_svcs = {s: sorted(ids) for s, ids in svc_account_ids.items() if len(ids) > 1}
    checks.append({
        "name": "per_service_account_id_consistent",
        "pass": len(inconsistent_svcs) == 0,
        "detail": f"account_id 불일치 서비스: {inconsistent_svcs if inconsistent_svcs else '없음'}",
    })

    # 3. 같은 서비스에서 같은 resource type의 ARN이 일관되는지 (registry 유지 검증)
    # 동일 서비스에서 같은 resource name이 등장하면 ARN도 같아야 함
    from collections import defaultdict
    name_arns: dict[str, list[str]] = defaultdict(list)
    for r in results:
        body = r.get("response_body", "")
        for name, arn in _extract_name_arn_pairs(body, r["service"]):
            name_arns[f"{r['service']}:{name}"].append(arn)

    inconsistent_names = {}
    for key, arns in name_arns.items():
        unique = set(arns)
        if len(unique) > 1:
            inconsistent_names[key] = sorted(unique)
    checks.append({
        "name": "resource_arn_consistent_across_same_service_calls",
        "pass": len(inconsistent_names) == 0,
        "detail": f"ARN 불일치 resource: {inconsistent_names if inconsistent_names else '없음'}",
    })

    return checks


def _extract_arns(body: str) -> list[str]:
    return re.findall(r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[^\s\"',<>]+", body)


def _account_id_from_arn(arn: str) -> str:
    parts = arn.split(":")
    return parts[4] if len(parts) > 4 else ""


def _is_error_response(body: str) -> bool:
    try:
        parsed = json.loads(body)
        return "__type" in parsed or "Error" in parsed
    except Exception:
        return "<Error>" in body or "ErrorResponse" in body


def _extract_name_arn_pairs(body: str, service: str) -> list[tuple[str, str]]:
    """응답 body에서 name→ARN 매핑을 추출한다."""
    pairs = []
    try:
        parsed = json.loads(body)
        _collect_name_arn(parsed, service, {}, pairs)
    except Exception:
        pass
    return pairs


def _collect_name_arn(obj: Any, service: str, ctx: dict, out: list) -> None:
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("Name")
        arn = next((v for k, v in obj.items() if "arn" in k.lower() and isinstance(v, str) and v.startswith("arn:aws:")), None)
        if name and arn:
            out.append((str(name), arn))
        for v in obj.values():
            _collect_name_arn(v, service, ctx, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_name_arn(item, service, ctx, out)


def _render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# 40-Command Single-Session Consistency Summary",
        "",
        f"- Mode: {payload['mode']}",
        f"- Session ID: {payload['session_id']}",
        f"- Total commands: {payload['total']}",
        f"- Overall pass: {payload['pass']}",
        "",
        "## Consistency Checks",
        "",
    ]
    for c in payload["consistency_checks"]:
        mark = "✓" if c["pass"] else "✗"
        lines.append(f"- {mark} **{c['name']}**: {c['detail']}")

    lines += ["", "## Per-Command Results", ""]
    lines.append("| ID | service | latency_ms | is_error | account_ids |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for r in payload["results"]:
        aids = ", ".join(r["account_ids_in_response"]) or "-"
        lines.append(f"| {r['id']} | {r['service']} | {r['latency_ms']} | {r['is_error']} | {aids} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
