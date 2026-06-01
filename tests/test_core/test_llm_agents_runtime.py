from __future__ import annotations

import json
import re

from moto.core.llm_agents import honeypot_profile
from moto.core.llm_agents.agent import handle_aws_request
from moto.core.llm_agents.runtime import (
    DEFAULT_OUTPUT,
    build_agent_prompt,
    call_gpt_api_with_meta,
    parse_agent_output,
)
from moto.core.llm_agents.runtime.runner import run_agent_loop
from moto.core.llm_agents.runtime.tool_executor import execute_agent_tool_requests
from moto.core.llm_agents.runtime.tool_registry import get_available_tool_names
from moto.core.llm_agents.shape_adapter import adapt_response_plan
from moto.core.llm_agents.tools.planning_tools import build_response_plan_tool
from moto.core.llm_agents.tools.report_tools import _build_report_prompt
from moto.core.llm_agents.tools.render_tools import serialize_response_tool
from moto.core.llm_agents.tools.request_tools import normalize_request_tool
from moto.core.llm_agents.tools.state_tools import (
    get_world_state_tool,
    has_cached_agent_response_tool,
    update_world_state_tool,
)
from moto.core.llm_agents.tools.validation_tools import (
    build_comparison_points_tool,
    validate_rendered_response_tool,
)


def test_normalizer_canonicalizes_prefixed_action() -> None:
    req = normalize_request_tool(
        service="ssm",
        action="ssm:DescribeInstanceInformation",
        url="https://ssm.ap-northeast-2.amazonaws.com/",
        headers={},
        body="",
    )
    assert req.service == "ssm"
    assert req.operation == "DescribeInstanceInformation"
    assert req.probe_style == "enumeration"
    assert req.body_format == "text"


def test_normalizer_extracts_json_request_params_and_identifiers() -> None:
    req = normalize_request_tool(
        service=None,
        action=None,
        url="https://api.ecr.ap-northeast-2.amazonaws.com/",
        headers={
            "X-Amz-Target": "AmazonEC2ContainerRegistry_V20150921.CompleteLayerUpload"
        },
        body='{"repositoryName":"demo","layerDigest":"sha256:abc","uploadId":"test"}',
    )
    assert req.service == "ecr"
    assert req.operation == "CompleteLayerUpload"
    assert req.body_format == "json"
    assert req.request_params["repositoryName"] == "demo"
    assert req.target_identifiers["repositoryName"] == "demo"
    assert req.target_identifiers["layerDigest"] == "sha256:abc"


def test_normalizer_recovers_service_from_sigv4_authorization() -> None:
    req = normalize_request_tool(
        service=None,
        action="ListAnalyzers",
        url="http://127.0.0.1:5000/analyzer",
        headers={
            "Authorization": (
                "AWS4-HMAC-SHA256 "
                "Credential=AKIAEXAMPLE/20260515/us-east-1/access-analyzer/aws4_request"
            )
        },
        body="",
    )

    assert req.service == "accessanalyzer"
    assert req.operation == "ListAnalyzers"


def test_parse_agent_output_falls_back_on_invalid_text() -> None:
    assert parse_agent_output("not-a-json") == DEFAULT_OUTPUT


def test_default_handle_aws_request_uses_agentic_runtime_without_mode_env(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("MOTO_LLM_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")
    audit_file = tmp_path / "audit.json"
    monkeypatch.setenv("MOTO_LLM_AUDIT_FILE", str(audit_file))

    response_body = handle_aws_request(
        service="ssm",
        action="DescribeInstanceInformation",
        url="https://ssm.us-east-1.amazonaws.com/",
        headers={"Authorization": "Credential=AKIAEXAMPLE/secret"},
        body="Action=DescribeInstanceInformation&Version=2014-11-06",
        reason="test",
        source="unit_test",
    )

    parsed = json.loads(response_body)
    assert parsed["InstanceInformationList"][0]["InstanceId"].startswith("i-")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    assert audit[-1]["metrics"]["llm"]["provider"] == "offline_stub"
    assert audit[-1]["request"]["headers"]["Authorization"] == "<redacted>"


def test_handle_aws_request_replans_after_validation_failure(monkeypatch) -> None:
    calls: list[str] = []
    responses = iter(
        [
            json.dumps(
                {
                    "intent_phase": "recon",
                    "response_posture": "normal",
                    "error_mode": "none",
                    "decoy_bundle_id": "first_pass",
                    "risk_delta": 0.1,
                    "reason_tags": ["enum_pattern"],
                    "response_plan": {
                        "mode": "success",
                        "posture": "normal",
                        "field_hints": {
                            "InstanceInformationList": [
                                {
                                    "InstanceId": "i-1234567890abcdef0",
                                    "PlatformType": "Linux",
                                    "ComputerName": "https://evil.example",
                                }
                            ]
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "intent_phase": "recon",
                    "response_posture": "sparse",
                    "error_mode": "none",
                    "decoy_bundle_id": "second_pass",
                    "risk_delta": 0.1,
                    "reason_tags": ["enum_pattern"],
                }
            ),
        ]
    )

    def fake_call(prompt: str) -> tuple[str, dict[str, object]]:
        calls.append(prompt)
        return next(responses), {"provider": "openai", "duration_ms": 1.0}

    monkeypatch.delenv("MOTO_LLM_OFFLINE_STUB", raising=False)
    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.runner.call_gpt_api_with_meta", fake_call
    )
    monkeypatch.setenv("MOTO_LLM_AGENT_MAX_ATTEMPTS", "2")

    response_body = handle_aws_request(
        service="ssm",
        action="DescribeInstanceInformation",
        url="https://ssm.ap-northeast-2.amazonaws.com/",
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
        body="Action=DescribeInstanceInformation&Version=2014-11-06",
        reason="test",
        source="unit_test",
    )

    parsed = json.loads(response_body)
    assert parsed["InstanceInformationList"][0]["InstanceId"].startswith("i-")
    assert len(calls) == 2
    assert "LATEST_OBSERVATION" in calls[1]
    assert "validation_failed" in calls[1]


def test_handle_aws_request_executes_agent_tool_request(monkeypatch) -> None:
    calls: list[str] = []
    responses = iter(
        [
            json.dumps(
                {
                    "intent_phase": "recon",
                    "response_posture": "sparse",
                    "error_mode": "none",
                    "decoy_bundle_id": "tool_pass",
                    "risk_delta": 0.1,
                    "reason_tags": ["enum_pattern"],
                    "tool_requests": [
                        {
                            "tool": "skills.load_skill_document",
                            "args": {"skill": "recon_skill"},
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "intent_phase": "recon",
                    "response_posture": "sparse",
                    "error_mode": "none",
                    "decoy_bundle_id": "final_pass",
                    "risk_delta": 0.1,
                    "reason_tags": ["enum_pattern"],
                    "response_plan": {
                        "mode": "success",
                        "posture": "sparse",
                        "entity_hints": {"count": 1},
                    },
                }
            ),
        ]
    )

    def fake_call(prompt: str) -> tuple[str, dict[str, object]]:
        calls.append(prompt)
        return next(responses), {"provider": "openai", "duration_ms": 1.0}

    monkeypatch.delenv("MOTO_LLM_OFFLINE_STUB", raising=False)
    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.runner.call_gpt_api_with_meta", fake_call
    )
    monkeypatch.setenv("MOTO_LLM_AGENT_MAX_ATTEMPTS", "2")

    response_body = handle_aws_request(
        service="eks",
        action="ListAddons",
        url="https://eks.us-east-1.amazonaws.com/",
        headers={},
        body='{"clusterName":"prod-main"}',
        reason="test",
        source="unit_test",
    )

    parsed = json.loads(response_body)
    assert parsed["addons"]
    assert len(calls) == 2
    assert "TOOL_OBSERVATIONS" in calls[1]
    assert "recon_skill" in calls[1]


def test_agent_tool_executor_exposes_honeypot_tools() -> None:
    canonical = normalize_request_tool(
        service="ecr",
        action="BatchCheckLayerAvailability",
        url="https://api.ecr.us-east-1.amazonaws.com/",
        headers={
            "X-Amz-Target": "AmazonEC2ContainerRegistry_V20150921.BatchCheckLayerAvailability"
        },
        body='{"repositoryName":"demo","layerDigests":["sha256:abc"]}',
    )
    world_state = {
        "region": "us-east-1",
        "phase": "recon",
        "risk_score": 0.2,
        "last_actions": ["ecr:GetDownloadUrlForLayer"],
        "consistency_locks": {"account_id": "123456789012"},
    }

    observation = execute_agent_tool_requests(
        [
            {"tool": "schema.inspect_output_shape", "args": {}},
            {"tool": "state.inspect_consistency", "args": {}},
            {"tool": "mock_data.get_mock_template", "args": {"category": "iam_policy"}},
        ],
        canonical=canonical,
        world_state=world_state,
        history_context="previous ecr response",
    )

    assert observation.startswith("TOOL_OBSERVATIONS=")
    assert "BatchCheckLayerAvailability" in observation
    assert "layers" in observation
    assert "123456789012" in observation
    assert "2012-10-17" in observation


def test_tool_registry_lists_only_agent_callable_tools() -> None:
    tools = get_available_tool_names()

    assert "skills.load_skill_document" in tools
    assert "schema.inspect_output_shape" in tools
    assert "state.inspect_consistency" in tools
    assert "mock_data.get_mock_template" in tools
    assert "shape_adapter.adapt_response_plan" not in tools
    assert "render_tools.serialize_response_tool" not in tools


def test_report_prompt_uses_first_action_timestamp_when_state_start_missing() -> None:
    prompt = _build_report_prompt(
        session_id="AKIAREPORTTEST",
        state={},
        action_log=[
            {
                "timestamp": "2026-05-27T18:43:56Z",
                "service": "sts",
                "operation": "GetCallerIdentity",
                "phase": "recon",
                "source": "unit_test",
            }
        ],
        timing={},
        iocs={},
        ttp_map={},
    )

    assert "- 세션 시작: 2026-05-27T18:43:56Z" in prompt
    assert "- 세션 시작: unknown" not in prompt


def test_validator_blocks_public_url() -> None:
    canonical = normalize_request_tool(
        service="ssm",
        action="DescribeInstanceInformation",
        url="https://ssm.ap-northeast-2.amazonaws.com/",
        headers={},
        body="",
    )
    valid, reason = validate_rendered_response_tool(
        canonical,
        '{"message": "visit https://example.com"}',
        {"consistency_locks": {"account_id": "123456789012"}},
    )
    assert valid is False
    assert "Safety pattern denied" in reason


def test_validator_allows_aws_xml_namespace() -> None:
    canonical = normalize_request_tool(
        service="ec2",
        action="DescribeInstances",
        url="https://ec2.ap-northeast-2.amazonaws.com/",
        headers={},
        body="",
    )
    body = (
        '<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">'
        "<reservationSet><item><instancesSet><item><instanceId>i-12345678</instanceId>"
        "</item></instancesSet></item></reservationSet></DescribeInstancesResponse>"
    )
    valid, reason = validate_rendered_response_tool(
        canonical,
        body,
        {"consistency_locks": {"account_id": "123456789012"}},
    )
    assert valid is True
    assert reason == "ok"


def test_comparison_points_capture_xml_parseability() -> None:
    canonical = normalize_request_tool(
        service="ec2",
        action="DescribeInstances",
        url="https://ec2.ap-northeast-2.amazonaws.com/",
        headers={},
        body="",
    )
    body = (
        '<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">'
        "<reservationSet><item><instancesSet><item><instanceId>i-12345678</instanceId>"
        "</item></instancesSet></item></reservationSet></DescribeInstancesResponse>"
    )
    points = build_comparison_points_tool(canonical, body, True, "ok")
    assert points["protocol_family_expected"] == "xml"
    assert points["response_format_detected"] == "xml"
    assert points["format_match"] is True
    assert points["response_parseable"] is True
    assert points["xml_namespace_present"] is True


def test_shape_adapter_and_serializer_for_iam_query() -> None:
    canonical = normalize_request_tool(
        service="iam",
        action="GetContextKeysForPrincipalPolicy",
        url="https://iam.amazonaws.com/",
        headers={},
        body="Action=GetContextKeysForPrincipalPolicy",
    )
    world_state = {
        "consistency_locks": {"account_id": "123456789012"},
        "exposed_assets": [],
    }
    plan = build_response_plan_tool(canonical, DEFAULT_OUTPUT, world_state, "")
    payload, _ = adapt_response_plan(canonical, plan, world_state)
    body, render_meta = serialize_response_tool(canonical, payload)

    assert "GetContextKeysForPrincipalPolicyResponse" in body
    assert "<member>aws:RequestedRegion</member>" in body
    assert render_meta["protocol"] == "query"


def test_shape_adapter_and_serializer_for_ecr_json() -> None:
    canonical = normalize_request_tool(
        service="ecr",
        action="InitiateLayerUpload",
        url="https://api.ecr.us-east-1.amazonaws.com/",
        headers={},
        body="{}",
    )
    world_state = {
        "consistency_locks": {"account_id": "123456789012"},
        "exposed_assets": [],
    }
    plan = build_response_plan_tool(canonical, DEFAULT_OUTPUT, world_state, "")
    payload, _ = adapt_response_plan(canonical, plan, world_state)
    body, render_meta = serialize_response_tool(canonical, payload)
    parsed = json.loads(body)

    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        parsed["uploadId"],
    )
    assert parsed["partSize"] > 0
    assert render_meta["protocol"] == "json"


def test_shape_adapter_echoes_request_identifiers_when_available() -> None:
    canonical = normalize_request_tool(
        service=None,
        action=None,
        url="https://api.ecr.us-east-1.amazonaws.com/",
        headers={
            "X-Amz-Target": "AmazonEC2ContainerRegistry_V20150921.CompleteLayerUpload"
        },
        body=(
            '{"repositoryName":"demo",'
            '"uploadId":"12345678-1234-1234-1234-123456789abc",'
            '"layerDigest":"sha256:abc"}'
        ),
    )
    world_state = {
        "consistency_locks": {"account_id": "123456789012"},
        "exposed_assets": [],
    }
    plan = build_response_plan_tool(canonical, DEFAULT_OUTPUT, world_state, "")
    payload, _ = adapt_response_plan(canonical, plan, world_state)

    assert payload["repositoryName"] == "demo"
    assert payload["uploadId"] == "12345678-1234-1234-1234-123456789abc"
    assert payload["layerDigest"] == "sha256:abc"


def test_shape_adapter_generates_single_member_for_union_shapes(monkeypatch) -> None:
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    response_body = handle_aws_request(
        service="accessanalyzer",
        action="ListAnalyzers",
        url="http://127.0.0.1:5000/analyzer",
        headers={
            "Authorization": (
                "AWS4-HMAC-SHA256 "
                "Credential=AKIAEXAMPLE/20260515/us-east-1/access-analyzer/aws4_request"
            )
        },
        body=b"",
        reason="unit test",
        source="unit_test",
    )

    parsed = json.loads(response_body)
    configuration = parsed["analyzers"][0]["configuration"]
    assert list(configuration) == ["unusedAccess"]


def test_shape_adapter_preserves_aws_like_instance_ids(monkeypatch) -> None:
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    response_body = handle_aws_request(
        service="ec2",
        action="MonitorInstances",
        url="http://127.0.0.1:5000/",
        headers={
            "Authorization": (
                "AWS4-HMAC-SHA256 "
                "Credential=AKIAEXAMPLE/20260515/us-east-1/ec2/aws4_request"
            )
        },
        body="Action=MonitorInstances&InstanceId.1=i-1234567890abcdef0",
        reason="unit test",
        source="unit_test",
    )

    assert "i-1234567890abcdef0" in response_body
    assert "instanceid-12345abcde" not in response_body


def test_agent_write_invalidates_cache_and_routes_only_affected_reads() -> None:
    session_id = "AKIAWRITEAFFECTEDREAD"
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/ec2/aws4_request"
        )
    }
    current = get_world_state_tool(session_id, headers)
    current["response_cache"] = {"ec2:DescribeInstances": {"Reservations": []}}
    canonical = normalize_request_tool(
        service="ec2",
        action="MonitorInstances",
        url="https://ec2.us-east-1.amazonaws.com/",
        headers=headers,
        body="Action=MonitorInstances&InstanceId.1=i-1234567890abcdef0",
    )

    update_world_state_tool(
        session_id,
        current,
        canonical,
        DEFAULT_OUTPUT,
        {"assets": []},
        field_values=None,
    )

    updated = get_world_state_tool(session_id, headers)
    assert updated["response_cache"] == {}
    assert has_cached_agent_response_tool(session_id, "ec2", "DescribeInstances")
    assert not has_cached_agent_response_tool(session_id, "ec2", "DescribeImages")


def test_honeypot_profile_state_is_seeded_for_profile_access_key() -> None:
    session_id = honeypot_profile.DEFAULT_ACCESS_KEY_ID
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/sts/aws4_request"
        )
    }

    state = get_world_state_tool(session_id, headers)

    assert state["persona"] == "nexora-prod-devops-bastion"
    assert state["region"] == honeypot_profile.REGION
    assert state["consistency_locks"]["account_id"] == honeypot_profile.ACCOUNT_ID
    assert state["honeypot_profile"]["iam_user"] == honeypot_profile.IAM_USER


def test_agent_prompt_includes_honeypot_profile_context() -> None:
    session_id = honeypot_profile.PROD_ACCESS_KEY_ID
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/backup-gateway/aws4_request"
        )
    }
    canonical = normalize_request_tool(
        service="backup-gateway",
        action="ListGateways",
        url="https://backup-gateway.us-east-1.amazonaws.com/",
        headers=headers,
        body="{}",
    )
    state = get_world_state_tool(session_id, headers)

    prompt = build_agent_prompt(canonical, state, "", "unit test", "unit_test")

    assert honeypot_profile.ACCOUNT_ID in prompt
    assert honeypot_profile.IAM_USER in prompt
    assert honeypot_profile.EKS_CLUSTER in prompt


def test_fallback_profile_overrides_backup_gateway(monkeypatch) -> None:
    session_id = honeypot_profile.DEFAULT_ACCESS_KEY_ID
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/backup-gateway/aws4_request"
        )
    }
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    response_body = handle_aws_request(
        service="backup-gateway",
        action="ListGateways",
        url="https://backup-gateway.us-east-1.amazonaws.com/",
        headers=headers,
        body="{}",
        reason="unit test",
        source="unit_test",
    )

    parsed = json.loads(response_body)
    gateway = parsed["Gateways"][0]
    assert gateway["GatewayDisplayName"] == "nexora-prod-backup-gateway"
    assert honeypot_profile.ACCOUNT_ID in gateway["GatewayArn"]
    assert "123456789012" not in response_body


def test_fallback_profile_overrides_secret_policy_validation(monkeypatch) -> None:
    session_id = honeypot_profile.DEFAULT_ACCESS_KEY_ID
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/secretsmanager/aws4_request"
        ),
        "Content-Type": "application/x-amz-json-1.1",
    }
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    response_body = handle_aws_request(
        service="secretsmanager",
        action="ValidateResourcePolicy",
        url="https://secretsmanager.us-east-1.amazonaws.com/",
        headers=headers,
        body=json.dumps(
            {
                "SecretId": "prod/db/password",
                "ResourcePolicy": (
                    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
                    '"Principal":"*","Action":"secretsmanager:GetSecretValue",'
                    '"Resource":"*"}]}'
                ),
            }
        ),
        reason="unit test",
        source="unit_test",
    )

    parsed = json.loads(response_body)
    assert parsed["PolicyValidationPassed"] is False
    assert parsed["ValidationErrors"][0]["CheckName"] == "SECURITY_WARNING"


def test_fallback_profile_overrides_iam_context_keys(monkeypatch) -> None:
    session_id = honeypot_profile.DEFAULT_ACCESS_KEY_ID
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/iam/aws4_request"
        )
    }
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    response_body = handle_aws_request(
        service="iam",
        action="GetContextKeysForPrincipalPolicy",
        url="https://iam.amazonaws.com/",
        headers=headers,
        body=(
            "Action=GetContextKeysForPrincipalPolicy&"
            "PolicySourceArn=arn:aws:iam::123456789012:user/victim-admin"
        ),
        reason="unit test",
        source="unit_test",
    )

    assert response_body.count("aws:RequestedRegion") == 1
    assert "aws:PrincipalArn" in response_body


def test_report_prompt_includes_honeypot_profile_context() -> None:
    session_id = honeypot_profile.DEFAULT_ACCESS_KEY_ID
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/sts/aws4_request"
        )
    }
    state = get_world_state_tool(session_id, headers)
    prompt = _build_report_prompt(
        session_id,
        state,
        [
            {
                "timestamp": "2026-05-22T00:00:00Z",
                "phase": "recon",
                "service": "sts",
                "operation": "GetCallerIdentity",
                "source": "moto_native",
                "new_assets": [],
            }
        ],
        {},
        {},
        {},
    )

    assert "허니팟 가상 회사 프로필" in prompt
    assert honeypot_profile.COMPANY_PREFIX in prompt
    assert honeypot_profile.IAM_USER in prompt
    assert honeypot_profile.EKS_CLUSTER in prompt


def test_moto_native_baseline_applies_resource_state_patches() -> None:
    instance_id = "i-1234567890abcdef0"
    native_body = (
        '<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">'
        "<reservationSet><item><reservationId>r-12345678</reservationId>"
        "<ownerId>123456789012</ownerId><instancesSet><item>"
        f"<instanceId>{instance_id}</instanceId>"
        "<imageId>ami-12345678</imageId><instanceType>t2.micro</instanceType>"
        "<monitoring><state>disabled</state></monitoring>"
        "</item></instancesSet></item></reservationSet>"
        "</DescribeInstancesResponse>"
    )
    canonical = normalize_request_tool(
        service="ec2",
        action="DescribeInstances",
        url="https://ec2.us-east-1.amazonaws.com/",
        headers={},
        body="Action=DescribeInstances",
    )
    world_state = {
        "consistency_locks": {"account_id": "123456789012"},
        "resource_state_patches": {
            "ec2": {instance_id: {"Monitoring": {"State": "enabled"}}}
        },
    }

    result = run_agent_loop(
        canonical=canonical,
        world_state=world_state,
        history_context="",
        reason="unit test",
        source="unit_test",
        moto_native_body=native_body,
    )

    assert result.planner_meta["provider"] == "moto_native_patches"
    assert "<instanceId>i-1234567890abcdef0</instanceId>" in result.response_body
    assert "<instanceType>t2.micro</instanceType>" in result.response_body
    assert "<state>enabled</state>" in result.response_body
    assert "<state>disabled</state>" not in result.response_body


def test_create_delete_reuses_session_resource_identifiers(monkeypatch) -> None:
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIARESOURCEFLOW/20260517/us-east-1/secretsmanager/aws4_request"
        )
    }

    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")
    created_body = handle_aws_request(
        service="secretsmanager",
        action="CreateSecret",
        url="https://secretsmanager.us-east-1.amazonaws.com/",
        headers=headers,
        body='{"Name":"prod/db"}',
        reason="unit test",
        source="unit_test",
    )
    created = json.loads(created_body)

    def fake_call(prompt: str) -> tuple[str, dict[str, object]]:
        return (
            json.dumps(
                {
                    "intent_phase": "impact_probe",
                    "response_posture": "normal",
                    "error_mode": "none",
                    "decoy_bundle_id": "delete_pass",
                    "risk_delta": 0.1,
                    "reason_tags": ["write_action"],
                    "response_plan": {
                        "mode": "success",
                        "posture": "normal",
                        "field_hints": {
                            "ARN": (
                                "arn:aws:secretsmanager:us-east-1:"
                                "999999999999:secret:wrong"
                            )
                        },
                    },
                }
            ),
            {"provider": "openai", "duration_ms": 1.0},
        )

    monkeypatch.delenv("MOTO_LLM_OFFLINE_STUB", raising=False)
    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.runner.call_gpt_api_with_meta", fake_call
    )

    deleted_body = handle_aws_request(
        service="secretsmanager",
        action="DeleteSecret",
        url="https://secretsmanager.us-east-1.amazonaws.com/",
        headers=headers,
        body='{"SecretId":"prod/db"}',
        reason="unit test",
        source="unit_test",
    )
    deleted = json.loads(deleted_body)

    assert deleted["ARN"] == created["ARN"]
    assert deleted["Name"] == created["Name"]


def test_read_create_read_delete_read_cycles_keep_resource_identity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    secret_headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIASECRETCYCLE/20260517/us-east-1/secretsmanager/aws4_request"
        )
    }
    handle_aws_request(
        service="secretsmanager",
        action="DescribeSecret",
        url="https://secretsmanager.us-east-1.amazonaws.com/",
        headers=secret_headers,
        body='{"SecretId":"prod/db/cycle"}',
        reason="unit test",
        source="unit_test",
    )
    secret_created = json.loads(
        handle_aws_request(
            service="secretsmanager",
            action="CreateSecret",
            url="https://secretsmanager.us-east-1.amazonaws.com/",
            headers=secret_headers,
            body='{"Name":"prod/db/cycle"}',
            reason="unit test",
            source="unit_test",
        )
    )
    secret_mid = json.loads(
        handle_aws_request(
            service="secretsmanager",
            action="DescribeSecret",
            url="https://secretsmanager.us-east-1.amazonaws.com/",
            headers=secret_headers,
            body='{"SecretId":"prod/db/cycle"}',
            reason="unit test",
            source="unit_test",
        )
    )
    secret_deleted = json.loads(
        handle_aws_request(
            service="secretsmanager",
            action="DeleteSecret",
            url="https://secretsmanager.us-east-1.amazonaws.com/",
            headers=secret_headers,
            body='{"SecretId":"prod/db/cycle"}',
            reason="unit test",
            source="unit_test",
        )
    )
    secret_post = json.loads(
        handle_aws_request(
            service="secretsmanager",
            action="DescribeSecret",
            url="https://secretsmanager.us-east-1.amazonaws.com/",
            headers=secret_headers,
            body='{"SecretId":"prod/db/cycle"}',
            reason="unit test",
            source="unit_test",
        )
    )

    # create/mid/delete 세 단계는 같은 ARN을 공유해야 함
    assert secret_created["ARN"] == secret_mid["ARN"] == secret_deleted["ARN"]
    assert (
        secret_created["Name"]
        == secret_mid["Name"]
        == secret_deleted["Name"]
        == "prod/db/cycle"
    )
    # 삭제 후 describe는 에러 응답이어야 함 (ARN 없음)
    assert "ARN" not in secret_post, (
        f"describe after delete should not contain ARN, got: {secret_post}"
    )

    ecr_headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIAECRCYCLE/20260517/us-east-1/ecr/aws4_request"
        )
    }
    ecr_url = "https://api.ecr.us-east-1.amazonaws.com/"
    handle_aws_request(
        service="ecr",
        action="DescribeRepositories",
        url=ecr_url,
        headers=ecr_headers,
        body='{"repositoryNames":["cycle-demo"]}',
        reason="unit test",
        source="unit_test",
    )
    ecr_created = json.loads(
        handle_aws_request(
            service="ecr",
            action="CreateRepository",
            url=ecr_url,
            headers=ecr_headers,
            body='{"repositoryName":"cycle-demo"}',
            reason="unit test",
            source="unit_test",
        )
    )["repository"]
    ecr_mid = json.loads(
        handle_aws_request(
            service="ecr",
            action="DescribeRepositories",
            url=ecr_url,
            headers=ecr_headers,
            body='{"repositoryNames":["cycle-demo"]}',
            reason="unit test",
            source="unit_test",
        )
    )["repositories"][0]
    ecr_deleted = json.loads(
        handle_aws_request(
            service="ecr",
            action="DeleteRepository",
            url=ecr_url,
            headers=ecr_headers,
            body='{"repositoryName":"cycle-demo","force":true}',
            reason="unit test",
            source="unit_test",
        )
    )["repository"]
    ecr_post_raw = json.loads(
        handle_aws_request(
            service="ecr",
            action="DescribeRepositories",
            url=ecr_url,
            headers=ecr_headers,
            body='{"repositoryNames":["cycle-demo"]}',
            reason="unit test",
            source="unit_test",
        )
    )

    # create/mid/delete 세 단계는 같은 ARN/name을 공유해야 함
    assert (
        ecr_created["repositoryArn"]
        == ecr_mid["repositoryArn"]
        == ecr_deleted["repositoryArn"]
    )
    assert (
        ecr_created["repositoryName"]
        == ecr_mid["repositoryName"]
        == ecr_deleted["repositoryName"]
        == "cycle-demo"
    )
    # 삭제 후 describe는 에러이거나 빈 repositories 목록이어야 함
    assert "__type" in ecr_post_raw or ecr_post_raw.get("repositories") == [], (
        f"describe after delete should be error or empty, got: {ecr_post_raw}"
    )


def test_live_not_found_is_stabilized_for_registered_resource(monkeypatch) -> None:
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIAECRLIVEFIX/20260517/us-east-1/ecr/aws4_request"
        )
    }
    url = "https://api.ecr.us-east-1.amazonaws.com/"

    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")
    created = json.loads(
        handle_aws_request(
            service="ecr",
            action="CreateRepository",
            url=url,
            headers=headers,
            body='{"repositoryName":"cycle-demo"}',
            reason="unit test",
            source="unit_test",
        )
    )["repository"]

    def fake_call(prompt: str) -> tuple[str, dict[str, object]]:
        return (
            json.dumps(
                {
                    "intent_phase": "impact_probe",
                    "response_posture": "sparse",
                    "error_mode": "not_found",
                    "decoy_bundle_id": "delete_not_found",
                    "risk_delta": 0.1,
                    "reason_tags": ["write_action"],
                    "response_plan": {
                        "mode": "error",
                        "posture": "sparse",
                        "field_hints": {"repository": {}},
                    },
                }
            ),
            {"provider": "openai", "duration_ms": 1.0},
        )

    monkeypatch.delenv("MOTO_LLM_OFFLINE_STUB", raising=False)
    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.runner.call_gpt_api_with_meta", fake_call
    )

    deleted = json.loads(
        handle_aws_request(
            service="ecr",
            action="DeleteRepository",
            url=url,
            headers=headers,
            body='{"repositoryName":"cycle-demo","force":true}',
            reason="unit test",
            source="unit_test",
        )
    )["repository"]

    assert deleted["repositoryArn"] == created["repositoryArn"]
    assert deleted["repositoryName"] == "cycle-demo"


def test_corpus_write_sequences_keep_request_and_generated_ids(monkeypatch) -> None:
    monkeypatch.setenv("MOTO_LLM_OFFLINE_STUB", "1")

    ec2_headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIAEC2WRITESEQ/20260517/us-east-1/ec2/aws4_request"
        )
    }
    monitored = handle_aws_request(
        service="ec2",
        action="MonitorInstances",
        url="https://ec2.us-east-1.amazonaws.com/",
        headers=ec2_headers,
        body="Action=MonitorInstances&InstanceId.1=i-1234567890abcdef0",
        reason="unit test",
        source="unit_test",
    )
    unmonitored = handle_aws_request(
        service="ec2",
        action="UnmonitorInstances",
        url="https://ec2.us-east-1.amazonaws.com/",
        headers=ec2_headers,
        body="Action=UnmonitorInstances&InstanceId.1=i-1234567890abcdef0",
        reason="unit test",
        source="unit_test",
    )

    assert "i-1234567890abcdef0" in monitored
    assert "i-1234567890abcdef0" in unmonitored

    ecr_headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIAECRWRITESEQ/20260517/us-east-1/ecr/aws4_request"
        )
    }
    ecr_url = "https://api.ecr.us-east-1.amazonaws.com/"
    upload = json.loads(
        handle_aws_request(
            service="ecr",
            action="InitiateLayerUpload",
            url=ecr_url,
            headers=ecr_headers,
            body='{"repositoryName":"demo"}',
            reason="unit test",
            source="unit_test",
        )
    )
    completed = json.loads(
        handle_aws_request(
            service="ecr",
            action="CompleteLayerUpload",
            url=ecr_url,
            headers=ecr_headers,
            body=json.dumps(
                {
                    "repositoryName": "demo",
                    "uploadId": upload["uploadId"],
                    "layerDigests": ["sha256:abc"],
                }
            ),
            reason="unit test",
            source="unit_test",
        )
    )

    assert completed["uploadId"] == upload["uploadId"]
    assert completed["repositoryName"] == "demo"
    assert completed["layerDigest"] == "sha256:abc"


def test_call_gpt_api_uses_direct_openai_by_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOTO_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS", raising=False)

    captured: dict[str, object] = {}

    def fake_post_json(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout: float,
        session: object,
    ) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout"] = timeout
        captured["session"] = session
        return {
            "id": "resp_test",
            "usage": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
            "output": [{"content": [{"type": "output_text", "text": '{"ok":true}'}]}],
        }

    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.provider._post_json", fake_post_json
    )

    text, meta = call_gpt_api_with_meta("test-prompt", timeout=7.5)

    assert text == '{"ok":true}'
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["timeout"] == 7.5
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input"][0]["content"] == "test-prompt"
    assert payload["max_output_tokens"] == 60
    assert meta["provider"] == "openai"
    assert meta["usage"]["total_tokens"] == 12


def test_call_gpt_api_uses_anthropic_when_only_anthropic_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MOTO_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("MOTO_LLM_ANTHROPIC_MAX_OUTPUT_TOKENS", "42")

    captured: dict[str, object] = {}

    def fake_post_json(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout: float,
        session: object,
    ) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout"] = timeout
        captured["session"] = session
        return {
            "id": "msg_test",
            "model": "claude-test",
            "usage": {"input_tokens": 9, "output_tokens": 4},
            "content": [{"type": "text", "text": '{"ok":true}'}],
        }

    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.provider._post_json", fake_post_json
    )

    text, meta = call_gpt_api_with_meta("test-prompt", model="claude-test", timeout=8.0)

    assert text == '{"ok":true}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["timeout"] == 8.0
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["x-api-key"] == "anthropic-test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "claude-test"
    assert payload["max_tokens"] == 42
    assert payload["messages"][0]["content"] == "test-prompt"
    assert meta["provider"] == "anthropic"
    assert meta["model"] == "claude-test"
    assert meta["usage"]["total_tokens"] == 13


def test_call_gpt_api_respects_explicit_anthropic_provider(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("MOTO_LLM_PROVIDER", "anthropic")

    monkeypatch.setattr(
        "moto.core.llm_agents.runtime.provider._post_json",
        lambda **_: {
            "id": "msg_test",
            "model": "claude-test",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "content": [{"type": "text", "text": '{"route":"anthropic"}'}],
        },
    )

    text, meta = call_gpt_api_with_meta("test-prompt", model="claude-test")

    assert text == '{"route":"anthropic"}'
    assert meta["provider"] == "anthropic"
    assert meta["usage"]["total_tokens"] == 3
