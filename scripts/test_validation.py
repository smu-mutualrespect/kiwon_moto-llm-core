#!/usr/bin/env python3
"""
validation_tools.py 및 Moto native 검증 로직 통합 테스트
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moto.core.llm_agents.tools.validation_tools import (
    _validate_against_shape,
    _validate_safety,
    validate_moto_native_response,
    validate_rendered_response_tool,
)
from moto.core.llm_agents.tools.request_tools import CanonicalRequest

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(label, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {label}" + (f" → {detail}" if detail else ""))
    results.append((label, cond))


def canonical(svc, op):
    return CanonicalRequest(
        service=svc, operation=op,
        principal_type="unknown", probe_style="enumeration",
        raw_action=op, request_params={}, target_identifiers={},
        body_format="unknown",
    )

# ─────────────────────────────────────────────
print("\n=== 1. 범용 botocore 스키마 검증 (_validate_against_shape) ===\n")

# 1-1. bedrock: JSON 프로토콜, FoundationModels 리스트 필요
body_ok = json.dumps({"modelSummaries": [{"modelId": "anthropic.claude-v2"}]})
body_bad_format = "<xml>wrong</xml>"
body_empty = "{}"

ok, reason = _validate_against_shape("bedrock", "ListFoundationModels", body_ok, check_empty=True)
check("bedrock:ListFoundationModels - 올바른 JSON", ok, reason)

ok, reason = _validate_against_shape("bedrock", "ListFoundationModels", body_bad_format, check_empty=True)
check("bedrock:ListFoundationModels - XML 응답 (거부해야 함)", not ok, reason)

ok, reason = _validate_against_shape("bedrock", "ListFoundationModels", body_empty, check_empty=True)
check("bedrock:ListFoundationModels - 빈 객체 (check_empty=True 거부해야 함)", not ok, reason)

ok, reason = _validate_against_shape("bedrock", "ListFoundationModels", body_empty, check_empty=False)
check("bedrock:ListFoundationModels - 빈 객체 (honeypot core member 없으면 check_empty=False여도 거부)", not ok, reason)

# 1-2. ec2: query 프로토콜, XML 응답이어야 함
ec2_xml_ok = """<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">
  <reservationSet><item><reservationId>r-abc</reservationId>
  <instancesSet><item><instanceId>i-1234567890abcdef0</instanceId></item></instancesSet>
  </item></reservationSet></DescribeInstancesResponse>"""
ec2_json_bad = '{"Reservations": []}'

ok, reason = _validate_against_shape("ec2", "DescribeInstances", ec2_xml_ok, check_empty=True)
check("ec2:DescribeInstances - 올바른 XML", ok, reason)

ok, reason = _validate_against_shape("ec2", "DescribeInstances", ec2_json_bad, check_empty=True)
check("ec2:DescribeInstances - JSON 응답 (거부해야 함)", not ok, reason)

# 1-3. ssm: JSON 프로토콜
ssm_ok = json.dumps({"InstanceInformationList": [{"InstanceId": "i-abc123", "PlatformType": "Linux"}]})
ssm_missing = json.dumps({"SomethingElse": []})

ok, reason = _validate_against_shape("ssm", "DescribeInstanceInformation", ssm_ok, check_empty=True)
check("ssm:DescribeInstanceInformation - 올바른 JSON", ok, reason)

ok, reason = _validate_against_shape("ssm", "DescribeInstanceInformation", ssm_missing, check_empty=True)
check("ssm:DescribeInstanceInformation - 필수 필드 누락 (거부해야 함)", not ok, reason)

# 1-4. secretsmanager: JSON 프로토콜
sm_ok = json.dumps({"PolicyValidationPassed": True})
sm_bad = json.dumps({"SomethingWrong": True})

ok, reason = _validate_against_shape("secretsmanager", "ValidateResourcePolicy", sm_ok, check_empty=True)
check("secretsmanager:ValidateResourcePolicy - 올바른 JSON", ok, reason)

ok, reason = _validate_against_shape("secretsmanager", "ValidateResourcePolicy", sm_bad, check_empty=True)
check("secretsmanager:ValidateResourcePolicy - 필수 필드 누락 (거부해야 함)", not ok, reason)

# 1-6. ecr: InitiateLayerUpload (honeypot core: uploadId, partSize)
ecr_ok = json.dumps({"uploadId": "upload-abc123", "partSize": 20971520})
ecr_missing = json.dumps({"uploadId": "upload-abc123"})

ok, reason = _validate_against_shape("ecr", "InitiateLayerUpload", ecr_ok, check_empty=True)
check("ecr:InitiateLayerUpload - 올바른 JSON", ok, reason)

ok, reason = _validate_against_shape("ecr", "InitiateLayerUpload", ecr_missing, check_empty=True)
check("ecr:InitiateLayerUpload - partSize 누락 (거부해야 함)", not ok, reason)

# 1-7. sts: DecodeAuthorizationMessage — query 프로토콜 = XML (JSON은 틀린 형식)
sts_xml_ok = '<DecodeAuthorizationMessageResponse><DecodedMessage>{"allowed":false}</DecodedMessage></DecodeAuthorizationMessageResponse>'
sts_xml_missing = '<DecodeAuthorizationMessageResponse><SomethingElse/></DecodeAuthorizationMessageResponse>'
sts_json_wrong = json.dumps({"DecodedMessage": '{"allowed": false}'})

ok, reason = _validate_against_shape("sts", "DecodeAuthorizationMessage", sts_xml_ok, check_empty=True)
check("sts:DecodeAuthorizationMessage - 올바른 XML (통과해야 함)", ok, reason)

ok, reason = _validate_against_shape("sts", "DecodeAuthorizationMessage", sts_xml_missing, check_empty=True)
check("sts:DecodeAuthorizationMessage - DecodedMessage 누락 XML (거부해야 함)", not ok, reason)

ok, reason = _validate_against_shape("sts", "DecodeAuthorizationMessage", sts_json_wrong, check_empty=True)
check("sts:DecodeAuthorizationMessage - JSON 형식 (STS는 XML이므로 거부해야 함)", not ok, reason)

# ─────────────────────────────────────────────
print("\n=== 2. 안전 패턴 검증 (_validate_safety) ===\n")

ok, reason = _validate_safety('{"result": "ok"}')
check("일반 JSON - 통과해야 함", ok, reason)

ok, reason = _validate_safety('{"url": "https://evil.com/steal"}')
check("외부 URL 포함 - 거부해야 함", not ok, reason)

ok, reason = _validate_safety('{"key": "AKIAIOSFODNN7EXAMPLE12"}')
check("AWS Access Key 패턴 포함 - 거부해야 함", not ok, reason)

ok, reason = _validate_safety('<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/"></DescribeInstancesResponse>')
check("EC2 XML namespace URL - 허용해야 함 (amazonaws.com)", ok, reason)

# ─────────────────────────────────────────────
print("\n=== 3. Moto native 응답 검증 (validate_moto_native_response) ===\n")

# 3-1. 정상 응답 - Agent 불필요
body_normal = json.dumps({"PolicyValidationPassed": True})
needs_agent = validate_moto_native_response("secretsmanager", "ValidateResourcePolicy", body_normal, 200)
check("secretsmanager 정상 응답 - Agent 불필요", not needs_agent)

# 3-2. 4xx 에러 - Agent 불필요 (에러는 정상)
needs_agent = validate_moto_native_response("secretsmanager", "ValidateResourcePolicy", '{"Error": "NotFound"}', 404)
check("404 에러 응답 - Agent 불필요", not needs_agent)

# 3-3. EC2 XML이어야 하는데 JSON 반환 - Agent 필요
needs_agent = validate_moto_native_response("ec2", "DescribeInstances", '{"Reservations": []}', 200)
check("ec2 JSON 응답 (XML이어야 함) - Agent 필요", needs_agent)

# 3-4. 필수 필드 누락 - Agent 필요
needs_agent = validate_moto_native_response("sts", "DecodeAuthorizationMessage", '{"SomeField": "val"}', 200)
check("sts 필수 필드(DecodedMessage) 누락 - Agent 필요", needs_agent)

# 3-5. 빈 body - Agent 불필요 (빈 응답은 허용)
needs_agent = validate_moto_native_response("ec2", "MonitorInstances", "", 200)
check("빈 body - Agent 불필요", not needs_agent)

# ─────────────────────────────────────────────
print("\n=== 4. validate_rendered_response_tool (LLM 응답 전체 검증) ===\n")

world_state = {"consistency_locks": {"account_id": "123456789012"}, "region": "us-east-1"}

# 4-1. account_id 일치 - 통과
body = json.dumps({"PolicyValidationPassed": True, "arn": "arn:aws:iam::123456789012:user/test"})
ok, reason = validate_rendered_response_tool(canonical("secretsmanager", "ValidateResourcePolicy"), body, world_state)
check("account_id 일치 ARN - 통과", ok, reason)

# 4-2. account_id 불일치 ARN - 거부
body = json.dumps({"PolicyValidationPassed": True, "arn": "arn:aws:iam::999999999999:user/test"})
ok, reason = validate_rendered_response_tool(canonical("secretsmanager", "ValidateResourcePolicy"), body, world_state)
check("account_id 불일치 ARN - 거부해야 함", not ok, reason)

# 4-3. 실제 외부 URL - 거부
body = json.dumps({"downloadUrl": "https://real-aws.s3.amazonaws.com/bucket/key"})
ok, reason = validate_rendered_response_tool(canonical("ecr", "GetDownloadUrlForLayer"), body, world_state)
check("실제 외부 URL 포함 - 거부해야 함", not ok, reason)

# ─────────────────────────────────────────────
print("\n=== 5. High-interaction 명령 검증 ===\n")
print("  (이 항목은 현재 코드에 구현되어 있지 않음 - 아래서 확인 예정)\n")

HIGH_INTERACTION = [
    ("ssm", "StartSession"),
    ("ssm", "ResumeSession"),
    ("ssm", "TerminateSession"),
    ("ecs", "ExecuteCommand"),
]
for svc, op in HIGH_INTERACTION:
    body = json.dumps({"SessionId": "session-abc", "TokenValue": "tok"})
    needs_agent = validate_moto_native_response(svc, op, body, 200)
    check(f"{svc}:{op} - high-interaction → 항상 Agent 필요", needs_agent)

# ─────────────────────────────────────────────
print("\n=== 결과 요약 ===\n")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  통과: {passed}/{total}")
failed = [(label, ok) for label, ok in results if not ok]
if failed:
    print("\n  실패 항목:")
    for label, _ in failed:
        print(f"    - {label}")
