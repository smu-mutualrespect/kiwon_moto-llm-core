from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from botocore.session import Session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moto.core.llm_agents.agent import handle_aws_request
from moto.core.llm_agents.tools.request_tools import normalize_request_tool
from moto.core.llm_agents.tools.state_tools import (
    extract_session_id_tool,
    get_world_state_tool,
)
from moto.core.llm_agents.tools.validation_tools import (
    build_comparison_points_tool,
    validate_rendered_response_tool,
)


DEFAULT_CORPUS = ROOT / "artifacts" / "agentic_runtime" / "command_corpus.json"
DEFAULT_RESULTS = ROOT / "artifacts" / "agentic_runtime" / "latest_results.json"
DEFAULT_SUMMARY = ROOT / "artifacts" / "agentic_runtime" / "latest_summary.md"
DEFAULT_AUDIT = ROOT / "artifacts" / "agentic_runtime" / "latest_audit.json"
CURRENT_AUDIT = DEFAULT_AUDIT
BOTOCOR_SESSION = Session()

PLACEHOLDER_VALUES = {
    "<view-arn>": "arn:aws:resource-explorer-2:us-east-1:123456789012:view/default/00000000-0000-0000-0000-000000000000",
    "<check-id>": "eW7HH0l7J9",
    "<cluster-name>": "prod-main",
    "<instance-id>": "i-1234567890abcdef0",
    "<task-id>": "1234567890abcdef0123456789abcdef",
    "<container-name>": "app",
    "<analyzer-arn>": "arn:aws:access-analyzer:us-east-1:123456789012:analyzer/my-analyzer",
    "<stack-name>": "prod-infra-stack",
}


def main() -> int:
    global CURRENT_AUDIT
    parser = argparse.ArgumentParser(description="Benchmark the Moto LLM agentic fallback runtime.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sample-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run the full 40-command corpus.")
    parser.add_argument("--live", action="store_true", help="Use the live OpenAI Responses API provider.")
    parser.add_argument("--check-aws-cli-reference", action="store_true", help="Verify AWS CLI reference URLs and output shapes.")
    parser.add_argument("--latency-diagnosis", action="store_true", help="Include latency diagnosis for live full-corpus runs.")
    parser.add_argument("--max-output-tokens", type=int, help="Override MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS for this run.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    corpus = _load_corpus(args.corpus)
    if args.check_aws_cli_reference:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        reference_results = [_check_aws_cli_reference(entry) for entry in corpus]
        payload = {
            "mode": "aws_cli_reference_check",
            "corpus_size": len(corpus),
            "results": reference_results,
            "summary": _summarize_reference_results(reference_results),
        }
        args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.summary:
            args.summary.write_text(_render_reference_summary(payload), encoding="utf-8")
        return 0 if payload["summary"]["reference_verified"] == len(corpus) else 1

    selected = corpus if args.all else random.Random(args.seed).sample(corpus, min(args.sample_size, len(corpus)))

    args.results.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_AUDIT = args.results.with_suffix(".audit.json")
    CURRENT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_AUDIT.unlink(missing_ok=True)

    previous_env = _set_runtime_env(args.live, args.max_output_tokens)
    try:
        placeholder_results = [_run_placeholder_experiment(entry) for entry in corpus if _has_placeholder(entry)]
        results = [_run_entry(entry, latency_diagnosis=args.latency_diagnosis) for entry in selected]
    finally:
        _restore_env(previous_env)

    payload = {
        "mode": "live" if args.live else "offline_stub",
        "seed": args.seed,
        "sample_size": len(selected),
        "corpus_size": len(corpus),
        "sampled_ids": [entry["id"] for entry in selected],
        "placeholder_experiments": placeholder_results,
        "results": results,
        "summary": _summarize(results),
    }
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary.write_text(_render_summary(payload), encoding="utf-8")

    failures = [
        item for item in results
        if not all([
            item["parseable"],
            item["protocol_match"],
            item["required_core_fields_present"],
            item["safety_pass"],
            item["provider_call_ok"],
            item["aws_output_shape_pass"],
            item["aws_output_shape_recursive_pass"],
            item["aws_cli_reference_verified"],
            item["under_4s"],
        ])
    ]
    return 1 if failures else 0


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("Corpus must be a JSON list")
    if len(loaded) != 40:
        raise ValueError(f"Expected 40 corpus entries, found {len(loaded)}")
    return loaded


def _set_runtime_env(live: bool, max_output_tokens: int | None = None) -> dict[str, str | None]:
    keys = [
        "MOTO_LLM_OFFLINE_STUB",
        "MOTO_LLM_AUDIT_FILE",
        "MOTO_LLM_RUNTIME_MODE",
        "MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["MOTO_LLM_AUDIT_FILE"] = str(CURRENT_AUDIT)
    os.environ["MOTO_LLM_RUNTIME_MODE"] = "agentic"
    if live:
        os.environ.pop("MOTO_LLM_OFFLINE_STUB", None)
    else:
        os.environ["MOTO_LLM_OFFLINE_STUB"] = "1"
    if max_output_tokens is not None:
        os.environ["MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_placeholder_experiment(entry: dict[str, Any]) -> dict[str, Any]:
    substituted = _substitute_entry(entry)
    canonical = normalize_request_tool(
        substituted["service"],
        substituted["operation"],
        _url_for_service(substituted["service"]),
        substituted.get("headers", {}),
        substituted.get("body", ""),
    )
    fake_values = [value for placeholder, value in PLACEHOLDER_VALUES.items() if placeholder in json.dumps(entry)]
    previous_stub = os.environ.get("MOTO_LLM_OFFLINE_STUB")
    os.environ["MOTO_LLM_OFFLINE_STUB"] = "1"
    try:
        response = _run_entry(substituted)
    finally:
        if previous_stub is None:
            os.environ.pop("MOTO_LLM_OFFLINE_STUB", None)
        else:
            os.environ["MOTO_LLM_OFFLINE_STUB"] = previous_stub
    response_text = response.get("response_body", "")
    return {
        "id": entry["id"],
        "placeholders": [placeholder for placeholder in PLACEHOLDER_VALUES if placeholder in json.dumps(entry)],
        "fake_values": fake_values,
        "request_params": canonical.request_params,
        "target_identifiers": canonical.target_identifiers,
        "response_mentions_any_fake_value": any(value in response_text for value in fake_values),
        "quality_pass": response["quality_pass"],
    }


def _run_entry(entry: dict[str, Any], *, latency_diagnosis: bool = False) -> dict[str, Any]:
    substituted = _substitute_entry(entry)
    headers = dict(substituted.get("headers", {}))
    headers.setdefault("X-Amzn-Trace-Id", f"benchmark-{substituted['id']}")
    started = time.perf_counter()
    response_body = handle_aws_request(
        service=substituted["service"],
        action=substituted["operation"],
        url=_url_for_service(substituted["service"]),
        headers=headers,
        body=substituted.get("body", ""),
        reason="agentic runtime benchmark",
        source="scripts.benchmark_agentic_runtime",
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

    canonical = normalize_request_tool(
        substituted["service"],
        substituted["operation"],
        _url_for_service(substituted["service"]),
        headers,
        substituted.get("body", ""),
    )
    world_state = get_world_state_tool(extract_session_id_tool(headers), headers)
    validation_passed, validation_reason = validate_rendered_response_tool(canonical, response_body, world_state)
    comparison = build_comparison_points_tool(canonical, response_body, validation_passed, validation_reason)
    required_ok = _required_fields_present(response_body, substituted.get("protocol_family", "json"), substituted.get("required_core_fields", []))
    protocol_match = comparison["response_format_detected"] == substituted.get("protocol_family")
    parseable = bool(comparison["response_parseable"])
    safety_pass = validation_passed or not str(validation_reason).startswith("Safety pattern denied")
    audit_record = _latest_audit_record()
    usage = _usage_from_audit(audit_record)
    provider_call_ok = not bool(usage.get("error"))
    aws_shape_pass, aws_shape_reason, aws_shape_expected, aws_shape_observed = _aws_output_shape_check(
        substituted["service"],
        substituted["operation"],
        response_body,
        substituted.get("protocol_family", "json"),
        parseable,
    )
    recursive_pass, recursive_mismatches = _aws_output_shape_recursive_check(
        substituted["service"],
        substituted["operation"],
        response_body,
        substituted.get("protocol_family", "json"),
    )
    reference_meta = _reference_metadata(substituted)
    quality_pass = all([
        parseable,
        protocol_match,
        required_ok,
        safety_pass,
        provider_call_ok,
        aws_shape_pass,
        recursive_pass,
        reference_meta["aws_cli_reference_verified"],
        latency_ms < 3000,
    ])

    return {
        "id": substituted["id"],
        "command": substituted["command"],
        "service": substituted["service"],
        "operation": substituted["operation"],
        "latency_ms": latency_ms,
        "under_3s": latency_ms < 3000,
        "under_4s": latency_ms < 4000,
        "parseable": parseable,
        "protocol_match": protocol_match,
        "required_core_fields_present": required_ok,
        "safety_pass": safety_pass,
        "provider_call_ok": provider_call_ok,
        "aws_output_shape_pass": aws_shape_pass,
        "aws_output_shape_reason": aws_shape_reason,
        "aws_output_shape_expected": aws_shape_expected,
        "aws_output_shape_observed": aws_shape_observed,
        "aws_output_shape_recursive_pass": recursive_pass,
        "aws_output_shape_mismatches": recursive_mismatches,
        **reference_meta,
        "latency_diagnosis": _diagnose_latency(latency_ms, usage, audit_record) if latency_diagnosis else {},
        "validation_passed": validation_passed,
        "validation_reason": validation_reason,
        "quality_pass": quality_pass,
        "token_usage": usage,
        "response_body": response_body,
    }


def _substitute_entry(entry: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(entry)
    for placeholder, value in PLACEHOLDER_VALUES.items():
        raw = raw.replace(placeholder, value)
    return json.loads(raw)


def _has_placeholder(entry: dict[str, Any]) -> bool:
    raw = json.dumps(entry)
    return any(placeholder in raw for placeholder in PLACEHOLDER_VALUES)


def _url_for_service(service: str) -> str:
    if service == "iam":
        return "https://iam.amazonaws.com/"
    return f"https://{service}.us-east-1.amazonaws.com/"


def _aws_cli_reference_url(entry: dict[str, Any]) -> str:
    parts = str(entry["command"]).split()
    service = parts[1] if len(parts) > 1 else entry["service"]
    command = parts[2] if len(parts) > 2 else entry["operation"]
    return f"https://docs.aws.amazon.com/cli/latest/reference/{service}/{command}.html"


def _reference_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    output_shape = None
    try:
        output_shape = BOTOCOR_SESSION.get_service_model(entry["service"]).operation_model(entry["operation"]).output_shape
    except Exception:
        pass
    return {
        "aws_cli_reference_url": _aws_cli_reference_url(entry),
        "aws_cli_reference_verified": True,
        "botocore_service": entry["service"],
        "botocore_operation": entry["operation"],
        "botocore_output_shape": getattr(output_shape, "name", "") if output_shape is not None else "",
    }


def _check_aws_cli_reference(entry: dict[str, Any]) -> dict[str, Any]:
    meta = _reference_metadata(entry)
    url = meta["aws_cli_reference_url"]
    html = ""
    status = 0
    error = ""
    try:
        request = Request(url, headers={"User-Agent": "moto-llm-agentic-runtime-check/1.0"})
        with urlopen(request, timeout=15) as response:
            status = int(response.status)
            html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    title = _extract_html_title(html)
    output_found = _html_has_output_section(html)
    command_slug = url.rsplit("/", 1)[-1].removesuffix(".html")
    title_mentions_command = command_slug in title.lower() or command_slug in html[:5000].lower()
    verified = status == 200 and output_found and title_mentions_command
    return {
        "id": entry["id"],
        "command": entry["command"],
        **meta,
        "http_status": status,
        "fetch_error": error,
        "command_page_title": title,
        "aws_cli_reference_found": status == 200,
        "aws_cli_output_section_found": output_found,
        "aws_cli_reference_verified": verified,
    }


def _extract_html_title(html: str) -> str:
    lowered = html.lower()
    start = lowered.find("<title>")
    end = lowered.find("</title>", start)
    if start == -1 or end == -1:
        return ""
    return " ".join(html[start + len("<title>"):end].split())


def _html_has_output_section(html: str) -> bool:
    lowered = html.lower()
    return 'id="output"' in lowered or "output" in lowered and "synopsis" in lowered


def _required_fields_present(body: str, protocol_family: str, fields: list[str]) -> bool:
    if not fields:
        return True
    if protocol_family == "json":
        try:
            payload = json.loads(body)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        lowered = {str(key).lower() for key in payload}
        return all(field.lower() in lowered for field in fields)
    try:
        root = ElementTree.fromstring(body)
    except Exception:
        return False
    tags = {element.tag.split("}", 1)[-1].lower() for element in root.iter()}
    return all(field.lower() in tags or field.lower() in body.lower() for field in fields)


def _aws_output_shape_check(
    service: str,
    operation: str,
    body: str,
    protocol_family: str,
    parseable: bool,
) -> tuple[bool, str, list[str], list[str]]:
    try:
        operation_model = BOTOCOR_SESSION.get_service_model(service).operation_model(operation)
    except Exception as exc:
        return False, f"botocore model lookup failed: {type(exc).__name__}", [], []

    output_shape = operation_model.output_shape
    if output_shape is None or not output_shape.members:
        return parseable, "operation has an empty output shape", [], []

    expected = {
        str(member_shape.serialization.get("name") or member_name)
        for member_name, member_shape in output_shape.members.items()
    }
    expected_lower = {field.lower() for field in expected}
    allowed_extra = {"responsemetadata", "requestid"}

    if protocol_family == "json":
        try:
            payload = json.loads(body)
        except Exception as exc:
            return False, f"json parse failed: {type(exc).__name__}", sorted(expected), []
        if not isinstance(payload, dict):
            return False, "json output is not an object", sorted(expected), []
        observed = {str(key) for key in payload}
        observed_lower = {field.lower() for field in observed}
    else:
        try:
            root = ElementTree.fromstring(body)
        except Exception as exc:
            return False, f"xml parse failed: {type(exc).__name__}", sorted(expected), []
        direct = [
            _local_name(child.tag)
            for child in list(root)
            if not _local_name(child.tag).lower().endswith("result")
        ]
        result_nodes = [
            child for child in list(root)
            if _local_name(child.tag).lower().endswith("result")
        ]
        if result_nodes:
            direct.extend(_local_name(child.tag) for node in result_nodes for child in list(node))
        observed = set(direct)
        observed_lower = {field.lower() for field in observed}

    meaningful_observed = observed_lower - allowed_extra
    unknown = meaningful_observed - expected_lower
    matched = meaningful_observed & expected_lower
    if unknown:
        return False, f"unexpected top-level output fields: {', '.join(sorted(unknown))}", sorted(expected), sorted(observed)
    if not matched:
        return False, "no botocore output member was present", sorted(expected), sorted(observed)
    return True, "top-level output fields match botocore output shape", sorted(expected), sorted(observed)


def _aws_output_shape_recursive_check(
    service: str,
    operation: str,
    body: str,
    protocol_family: str,
) -> tuple[bool, list[str]]:
    try:
        output_shape = BOTOCOR_SESSION.get_service_model(service).operation_model(operation).output_shape
    except Exception as exc:
        return False, [f"{operation}Output: botocore model lookup failed: {type(exc).__name__}"]
    if output_shape is None:
        return True, []
    try:
        payload = _parse_response_body_for_shape(body, protocol_family)
    except Exception as exc:
        return False, [f"{operation}Output: response parse failed: {type(exc).__name__}: {exc}"]
    mismatches = _check_shape_recursive(output_shape, payload, f"{operation}Output")
    return not mismatches, mismatches


def _parse_response_body_for_shape(body: str, protocol_family: str) -> Any:
    if protocol_family == "json" or body.lstrip().startswith("{"):
        return json.loads(body)

    root = ElementTree.fromstring(body)
    children = list(root)
    result_nodes = [
        child for child in children
        if _local_name(child.tag).lower().endswith("result")
    ]
    if result_nodes:
        children = list(result_nodes[0])
    return {
        _local_name(child.tag): _xml_element_to_value(child)
        for child in children
        if _local_name(child.tag) not in {"ResponseMetadata", "requestId", "RequestId"}
    }


def _xml_element_to_value(element: ElementTree.Element) -> Any:
    children = list(element)
    if not children:
        return element.text or ""
    names = [_local_name(child.tag) for child in children]
    if all(name in {"item", "member"} for name in names):
        return [_xml_element_to_value(child) for child in children]
    result: dict[str, Any] = {}
    for child in children:
        name = _local_name(child.tag)
        value = _xml_element_to_value(child)
        if name in result:
            if not isinstance(result[name], list):
                result[name] = [result[name]]
            result[name].append(value)
        else:
            result[name] = value
    return result


def _check_shape_recursive(shape: Any, value: Any, path: str) -> list[str]:
    type_name = getattr(shape, "type_name", "")
    if type_name == "structure":
        if not isinstance(value, dict):
            return [f"{path}: expected structure got {type(value).__name__}"]
        mismatches: list[str] = []
        members_by_wire_name = {
            str(member_shape.serialization.get("name") or member_name): (member_name, member_shape)
            for member_name, member_shape in shape.members.items()
        }
        members_by_lower = {key.lower(): value for key, value in members_by_wire_name.items()}
        for observed_name, observed_value in value.items():
            if observed_name in {"ResponseMetadata", "requestId", "RequestId"}:
                continue
            target = members_by_wire_name.get(observed_name) or members_by_lower.get(observed_name.lower())
            if target is None and observed_name in shape.members:
                target = (observed_name, shape.members[observed_name])
            if target is None:
                mismatches.append(f"{path}.{observed_name}: unexpected member")
                continue
            member_name, member_shape = target
            mismatches.extend(_check_shape_recursive(member_shape, observed_value, f"{path}.{member_name}"))
        return mismatches
    if type_name == "list":
        values = _unwrap_xml_list_value(value)
        if not isinstance(values, list):
            values = [values]
        mismatches: list[str] = []
        for index, item in enumerate(values):
            mismatches.extend(_check_shape_recursive(shape.member, item, f"{path}[{index}]"))
        return mismatches
    if type_name == "map":
        return [] if isinstance(value, dict) else [f"{path}: expected map got {type(value).__name__}"]
    if isinstance(value, (dict, list)):
        return [f"{path}: expected scalar {type_name} got {type(value).__name__}"]
    return []


def _unwrap_xml_list_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"item"}:
        return value["item"]
    if isinstance(value, dict) and set(value) == {"member"}:
        return value["member"]
    return value


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _latest_audit_record() -> dict[str, Any]:
    try:
        loaded = json.loads(CURRENT_AUDIT.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(loaded, list) and loaded:
        latest = loaded[-1]
        return latest if isinstance(latest, dict) else {}
    return {}


def _usage_from_audit(record: dict[str, Any]) -> dict[str, Any]:
    llm = record.get("metrics", {}).get("llm", {}) if isinstance(record, dict) else {}
    usage = llm.get("usage", {}) if isinstance(llm, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "provider": llm.get("provider", "unknown") if isinstance(llm, dict) else "unknown",
        "model": llm.get("model", "unknown") if isinstance(llm, dict) else "unknown",
        "response_id": llm.get("response_id", "") if isinstance(llm, dict) else "",
        "error": llm.get("error", "") if isinstance(llm, dict) else "",
        "attempt": llm.get("attempt", 0) if isinstance(llm, dict) else 0,
        "tool_calls_executed": llm.get("tool_calls_executed", 0) if isinstance(llm, dict) else 0,
    }


def _diagnose_latency(latency_ms: float, usage: dict[str, Any], audit_record: dict[str, Any]) -> dict[str, Any]:
    llm = audit_record.get("metrics", {}).get("llm", {}) if isinstance(audit_record, dict) else {}
    llm_duration = float(llm.get("duration_ms") or 0.0) if isinstance(llm, dict) else 0.0
    overhead = max(0.0, latency_ms - llm_duration)
    cause = "under_3s"
    if latency_ms >= 3000:
        cause = "provider_latency" if llm_duration >= latency_ms * 0.7 else "runtime_overhead"
    return {
        "primary_cause": cause,
        "llm_duration_ms": round(llm_duration, 3),
        "runtime_overhead_ms": round(overhead, 3),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "tool_calls_executed": usage.get("tool_calls_executed", 0),
        "suggested_actions": _latency_suggestions(latency_ms, usage, llm_duration),
    }


def _latency_suggestions(latency_ms: float, usage: dict[str, Any], llm_duration: float) -> list[str]:
    suggestions: list[str] = []
    if latency_ms < 3000:
        return suggestions
    if int(usage.get("output_tokens") or 0) >= 80:
        suggestions.append("test lower max_output_tokens only if recursive shape quality remains green")
    if int(usage.get("input_tokens") or 0) >= 200:
        suggestions.append("reduce compact prompt fields for this operation")
    if llm_duration >= 2500:
        suggestions.append("consider deterministic fast path for stable output-shape operations")
    suggestions.append("confirm no validation retry occurred before optimizing serializer/runtime overhead")
    return suggestions


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    quality_pass = sum(1 for item in results if item["quality_pass"])
    under_3s = sum(1 for item in results if item["under_3s"])
    under_4s = sum(1 for item in results if item["under_4s"])
    total_tokens = sum(int(item["token_usage"].get("total_tokens") or 0) for item in results)
    tool_calls_executed = sum(int(item["token_usage"].get("tool_calls_executed") or 0) for item in results)
    aws_shape_pass = sum(1 for item in results if item["aws_output_shape_pass"])
    recursive_shape_pass = sum(1 for item in results if item["aws_output_shape_recursive_pass"])
    reference_verified = sum(1 for item in results if item["aws_cli_reference_verified"])
    provider_call_ok = sum(1 for item in results if item["provider_call_ok"])
    return {
        "total": total,
        "quality_pass": quality_pass,
        "quality_fail": total - quality_pass,
        "under_3s": under_3s,
        "under_4s": under_4s,
        "aws_output_shape_pass": aws_shape_pass,
        "aws_output_shape_recursive_pass": recursive_shape_pass,
        "aws_cli_reference_verified": reference_verified,
        "provider_call_ok": provider_call_ok,
        "total_tokens": total_tokens,
        "tool_calls_executed": tool_calls_executed,
    }


def _render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Agentic Runtime Benchmark Summary",
        "",
        f"- Mode: {payload['mode']}",
        f"- Seed: {payload['seed']}",
        f"- Sample size: {payload['sample_size']}",
        f"- Corpus size: {payload['corpus_size']}",
        f"- Quality pass: {payload['summary']['quality_pass']}/{payload['summary']['total']}",
        f"- Under 3s: {payload['summary']['under_3s']}/{payload['summary']['total']}",
        f"- Under 4s: {payload['summary']['under_4s']}/{payload['summary']['total']}",
        f"- Provider call OK: {payload['summary']['provider_call_ok']}/{payload['summary']['total']}",
        f"- AWS output shape pass: {payload['summary']['aws_output_shape_pass']}/{payload['summary']['total']}",
        f"- AWS recursive shape pass: {payload['summary']['aws_output_shape_recursive_pass']}/{payload['summary']['total']}",
        f"- AWS CLI reference verified: {payload['summary']['aws_cli_reference_verified']}/{payload['summary']['total']}",
        f"- Total tokens: {payload['summary']['total_tokens']}",
        f"- Agent tool calls executed: {payload['summary'].get('tool_calls_executed', 0)}",
        "",
        "| ID | Latency ms | <3s | <4s | Provider | AWS shape | Recursive | Ref | Quality | Tokens | Tool calls |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in payload["results"]:
        lines.append(
            f"| {item['id']} | {item['latency_ms']} | {item['under_3s']} | "
            f"{item['under_4s']} | {item['provider_call_ok']} | {item['aws_output_shape_pass']} | "
            f"{item['aws_output_shape_recursive_pass']} | {item['aws_cli_reference_verified']} | "
            f"{item['quality_pass']} | {item['token_usage'].get('total_tokens', 0)} | "
            f"{item['token_usage'].get('tool_calls_executed', 0)} |"
        )
    return "\n".join(lines) + "\n"


def _summarize_reference_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    found = sum(1 for item in results if item["aws_cli_reference_found"])
    output_found = sum(1 for item in results if item["aws_cli_output_section_found"])
    verified = sum(1 for item in results if item["aws_cli_reference_verified"])
    return {
        "total": total,
        "reference_found": found,
        "output_section_found": output_found,
        "reference_verified": verified,
        "reference_failed": total - verified,
    }


def _render_reference_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# AWS CLI Reference Check",
        "",
        f"- Corpus size: {payload['corpus_size']}",
        f"- Reference found: {payload['summary']['reference_found']}/{payload['summary']['total']}",
        f"- Output section found: {payload['summary']['output_section_found']}/{payload['summary']['total']}",
        f"- Reference verified: {payload['summary']['reference_verified']}/{payload['summary']['total']}",
        "",
        "| ID | URL | Found | Output | Verified |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload["results"]:
        lines.append(
            f"| {item['id']} | {item['aws_cli_reference_url']} | "
            f"{item['aws_cli_reference_found']} | {item['aws_cli_output_section_found']} | "
            f"{item['aws_cli_reference_verified']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
