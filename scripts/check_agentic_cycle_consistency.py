from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moto.core.llm_agents.agent import handle_aws_request

DEFAULT_CORPUS = (
    ROOT / "artifacts" / "agentic_runtime" / "ecr_repository_cycle_corpus.json"
)
DEFAULT_RESULTS = (
    ROOT / "artifacts" / "agentic_runtime" / "ecr_repository_cycle_results.json"
)
DEFAULT_SUMMARY = (
    ROOT / "artifacts" / "agentic_runtime" / "ecr_repository_cycle_summary.md"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run read/create/read/delete/read consistency checks."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    entries = json.loads(args.corpus.read_text(encoding="utf-8"))
    previous_stub = os.environ.get("MOTO_LLM_OFFLINE_STUB")
    previous_runtime = os.environ.get("MOTO_LLM_RUNTIME_MODE")
    os.environ["MOTO_LLM_RUNTIME_MODE"] = "agentic"
    if args.live:
        os.environ.pop("MOTO_LLM_OFFLINE_STUB", None)
    else:
        os.environ["MOTO_LLM_OFFLINE_STUB"] = "1"

    try:
        results = [_run_entry(entry) for entry in entries]
    finally:
        if previous_stub is None:
            os.environ.pop("MOTO_LLM_OFFLINE_STUB", None)
        else:
            os.environ["MOTO_LLM_OFFLINE_STUB"] = previous_stub
        if previous_runtime is None:
            os.environ.pop("MOTO_LLM_RUNTIME_MODE", None)
        else:
            os.environ["MOTO_LLM_RUNTIME_MODE"] = previous_runtime

    checks = _consistency_checks(results)
    payload = {
        "mode": "live" if args.live else "offline_stub",
        "corpus": str(args.corpus),
        "results": results,
        "checks": checks,
        "pass": all(item["pass"] for item in checks),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary.write_text(_render_summary(payload), encoding="utf-8")
    return 0 if payload["pass"] else 1


def _run_entry(entry: dict[str, Any]) -> dict[str, Any]:
    headers = dict(entry.get("headers", {}))
    headers.setdefault(
        "Authorization",
        "AWS4-HMAC-SHA256 Credential=AKIACYCLECORPUS/20260517/us-east-1/ecr/aws4_request",
    )
    started = time.perf_counter()
    body = handle_aws_request(
        service=entry["service"],
        action=entry["operation"],
        url="https://api.ecr.us-east-1.amazonaws.com/",
        headers=headers,
        body=entry["body"],
        reason="cycle consistency check",
        source="scripts.check_agentic_cycle_consistency",
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    parsed = json.loads(body)
    repository = _extract_repository(parsed)
    return {
        "id": entry["id"],
        "phase": entry["phase"],
        "command": entry["command"],
        "latency_ms": latency_ms,
        "repository": repository,
        "response_body": body,
    }


def _extract_repository(parsed: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed.get("repository"), dict):
        return parsed["repository"]
    repositories = parsed.get("repositories")
    if isinstance(repositories, list) and repositories and isinstance(repositories[0], dict):
        return repositories[0]
    return {}


def _consistency_checks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity_phases = {"create", "read_after_create", "delete"}
    repositories = [
        item["repository"] for item in results if item.get("phase") in identity_phases
    ]
    return [
        _same_field("repositoryName", repositories),
        _same_field("repositoryArn", repositories),
        _same_field("registryId", repositories),
    ]


def _same_field(field: str, repositories: list[dict[str, Any]]) -> dict[str, Any]:
    values = [repo.get(field, "") for repo in repositories]
    return {
        "field": field,
        "values": values,
        "pass": bool(values) and all(value == values[0] and value for value in values),
    }


def _render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Agentic Runtime Cycle Consistency",
        "",
        f"- Mode: {payload['mode']}",
        f"- Pass: {payload['pass']}",
        "",
        "| Phase | Command | RepositoryName | RepositoryArn | RegistryId | Latency ms |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for item in payload["results"]:
        repo = item["repository"]
        lines.append(
            f"| {item['phase']} | `{item['command']}` | "
            f"`{repo.get('repositoryName', '')}` | `{repo.get('repositoryArn', '')}` | "
            f"`{repo.get('registryId', '')}` | {item['latency_ms']} |"
        )
    lines.extend(["", "| Field | Pass | Values |", "| --- | --- | --- |"])
    for check in payload["checks"]:
        values = ", ".join(f"`{value}`" for value in check["values"])
        lines.append(f"| {check['field']} | {check['pass']} | {values} |")
    lines.extend(
        [
            "",
            "Note: identity checks compare create/read_after_create/delete only.",
            "read_before_create and read_after_delete are reported for lifecycle review,",
            "but are not part of the identity equality check.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
