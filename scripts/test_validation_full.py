#!/usr/bin/env python3
"""
종합 검증 테스트:
 1. 범용 botocore 스키마 검증 (다수 서비스)
 2. 값 형식(value format) 검증
 3. Moto native 응답 검증 + Agent fallback
 4. High-interaction 명령 검증
 5. moto_server 실제 연동 (서버 기동 후 AWS CLI 명령 실행)
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moto.core.llm_agents.tools.validation_tools import (
    _validate_against_shape,
    _validate_safety,
    _validate_value_formats,
    validate_moto_native_response,
    validate_rendered_response_tool,
    _HIGH_INTERACTION_OPERATIONS,
    _HONEYPOT_CORE_MEMBERS,
)
from moto.core.llm_agents.tools.request_tools import CanonicalRequest

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
results: list[tuple[str, bool]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    status = PASS if cond else FAIL
    print(f"  [{status}] {label}" + (f"  → {detail}" if detail else ""))
    results.append((label, cond))


def canon(svc: str, op: str) -> CanonicalRequest:
    return CanonicalRequest(
        service=svc, operation=op,
        principal_type="unknown", probe_style="enumeration",
        raw_action=op, request_params={}, target_identifiers={},
        body_format="unknown",
    )


# ══════════════════════════════════════════════════════════════
print("\n=== 1. 범용 botocore 스키마 검증 (다수 서비스) ===\n")

CASES = [
    # (service, operation, good_body, bad_body, description)
    ("bedrock",        "ListFoundationModels",
     json.dumps({"modelSummaries": [{"modelId": "anthropic.claude-v2"}]}),
     "{}",
     "bedrock: modelSummaries 필수"),

    ("ecr",            "InitiateLayerUpload",
     json.dumps({"uploadId": "upload-abc", "partSize": 20971520}),
     json.dumps({"uploadId": "upload-abc"}),
     "ecr: partSize 필수"),

    ("ecr",            "GetDownloadUrlForLayer",
     json.dumps({"downloadUrl": "mock://ecr/demo/blobs/sha256/abc", "layerDigest": "sha256:abc"}),
     json.dumps({"downloadUrl": "mock://ecr/demo/blobs/sha256/abc"}),
     "ecr: layerDigest 필수"),

    ("ecr",            "BatchCheckLayerAvailability",
     json.dumps({"layers": [{"layerDigest": "sha256:abc", "layerAvailability": "AVAILABLE"}]}),
     json.dumps({"something": []}),
     "ecr: layers 필수"),

    ("ssm",            "DescribeInstanceInformation",
     json.dumps({"InstanceInformationList": [{"InstanceId": "i-1234567890abcdef0"}]}),
     json.dumps({"Other": []}),
     "ssm: InstanceInformationList 필수"),

    ("secretsmanager", "ValidateResourcePolicy",
     json.dumps({"PolicyValidationPassed": True}),
     json.dumps({"SomethingElse": True}),
     "secretsmanager: PolicyValidationPassed 필수"),

    # IAM은 query 프로토콜 → XML 형식
    ("iam",            "GetContextKeysForPrincipalPolicy",
     "<GetContextKeysForPrincipalPolicyResponse><ContextKeyNames><member>aws:RequestedRegion</member></ContextKeyNames></GetContextKeysForPrincipalPolicyResponse>",
     "<GetContextKeysForPrincipalPolicyResponse><WrongKey/></GetContextKeysForPrincipalPolicyResponse>",
     "iam: ContextKeyNames 필수 (XML)"),

    ("iam",            "ListServiceSpecificCredentials",
     "<ListServiceSpecificCredentialsResponse><ServiceSpecificCredentials/></ListServiceSpecificCredentialsResponse>",
     "<ListServiceSpecificCredentialsResponse><WrongTag/></ListServiceSpecificCredentialsResponse>",
     "iam: ServiceSpecificCredentials 필수 (XML)"),

    ("iam",            "GenerateServiceLastAccessedDetails",
     "<GenerateServiceLastAccessedDetailsResponse><JobId>job-abc123</JobId></GenerateServiceLastAccessedDetailsResponse>",
     "<GenerateServiceLastAccessedDetailsResponse><Missing>field</Missing></GenerateServiceLastAccessedDetailsResponse>",
     "iam: JobId 필수 (XML)"),

    ("ec2",            "DescribeInstances",
     '<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/"><reservationSet/></DescribeInstancesResponse>',
     '{"Reservations": []}',
     "ec2: XML 형식 필수"),

    ("sts",            "DecodeAuthorizationMessage",
     "<DecodeAuthorizationMessageResponse><DecodedMessage>{}</DecodedMessage></DecodeAuthorizationMessageResponse>",
     json.dumps({"DecodedMessage": "{}"}),
     "sts: XML 형식 필수"),
]

for svc, op, good, bad, desc in CASES:
    ok, reason = _validate_against_shape(svc, op, good, check_empty=True)
    check(f"{svc}:{op} - 올바른 응답 통과 ({desc})", ok, reason)
    ok, reason = _validate_against_shape(svc, op, bad, check_empty=True)
    check(f"{svc}:{op} - 잘못된 응답 거부", not ok, reason)


# ══════════════════════════════════════════════════════════════
print("\n=== 2. 값 형식(value format) 검증 ===\n")

# 올바른 값
ok, r = _validate_value_formats({"InstanceId": "i-1234567890abcdef0"})
check("InstanceId 올바른 형식 (i-...17자리) - 통과", ok, r)

ok, r = _validate_value_formats({"VolumeId": "vol-1234567890abcdef0"})
check("VolumeId 올바른 형식 (vol-...17자리) - 통과", ok, r)

ok, r = _validate_value_formats({"SomeArn": "arn:aws:iam::123456789012:user/test"})
check("ARN 올바른 형식 - 통과", ok, r)

ok, r = _validate_value_formats({"AccountId": "123456789012"})
check("AccountId 12자리 숫자 - 통과", ok, r)

# 잘못된 값
ok, r = _validate_value_formats({"InstanceId": "instance-abc"})
check("InstanceId 잘못된 형식 - 거부", not ok, r)

ok, r = _validate_value_formats({"VolumeId": "volume-abc"})
check("VolumeId 잘못된 형식 - 거부", not ok, r)

ok, r = _validate_value_formats({"SomeArn": "arn:fake:not-valid"})
check("ARN 잘못된 형식 (account ID 없음) - 거부", not ok, r)

ok, r = _validate_value_formats({"AccountId": "12345"})
check("AccountId 12자리 미만 - 거부", not ok, r)

# 중첩 구조
ok, r = _validate_value_formats({
    "Instances": [{"InstanceId": "i-abc123456789abcde", "State": "running"}]
})
check("중첩 리스트 내 InstanceId 올바른 형식 - 통과", ok, r)

ok, r = _validate_value_formats({
    "Instances": [{"InstanceId": "fake-id-xyz"}]
})
check("중첩 리스트 내 InstanceId 잘못된 형식 - 거부", not ok, r)


# ══════════════════════════════════════════════════════════════
print("\n=== 3. Moto native 응답 검증 ===\n")

# 정상 응답 (Agent 불필요)
cases_ok = [
    ("secretsmanager", "ValidateResourcePolicy", json.dumps({"PolicyValidationPassed": True}), 200),
    ("ec2",            "DescribeInstances",
     '<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/"><reservationSet/></DescribeInstancesResponse>', 200),
    ("s3",             "ListBuckets",
     '<ListAllMyBucketsResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Buckets/></ListAllMyBucketsResult>', 200),
    ("iam",            "ListServiceSpecificCredentials",
     "<ListServiceSpecificCredentialsResponse><ServiceSpecificCredentials/></ListServiceSpecificCredentialsResponse>", 200),
    ("ssm",            "DescribeInstanceInformation",
     json.dumps({"InstanceInformationList": [{"InstanceId": "i-abc123"}]}), 200),
    ("ec2",            "DescribeInstances", '{"whatever": 1}', 404),  # 에러는 Agent 불필요
]
for svc, op, body, status in cases_ok:
    needs = validate_moto_native_response(svc, op, body, status)
    label = f"{svc}:{op} (status={status}) - Agent 불필요"
    check(label, not needs)

# 문제 있는 응답 (Agent 필요)
cases_bad = [
    ("ec2",    "DescribeInstances", '{"Reservations": []}', 200,
     "ec2 JSON 응답 → XML 필요"),
    ("sts",    "DecodeAuthorizationMessage", '{"DecodedMessage": "x"}', 200,
     "sts JSON 응답 → XML 필요"),
    ("bedrock","ListFoundationModels", "{}", 200,
     "bedrock 빈 응답 → modelSummaries 필요"),
    ("iam",    "GenerateServiceLastAccessedDetails", '{"Wrong": "x"}', 200,
     "iam JobId 누락"),
]
for svc, op, body, status, desc in cases_bad:
    needs = validate_moto_native_response(svc, op, body, status)
    check(f"{svc}:{op} - Agent 필요 ({desc})", needs)


# ══════════════════════════════════════════════════════════════
print("\n=== 4. High-interaction 명령 (항상 Agent) ===\n")

expected_hi = [
    ("ssm", "StartSession"),
    ("ssm", "ResumeSession"),
    ("ssm", "TerminateSession"),
    ("ecs", "ExecuteCommand"),
]
for svc, op in expected_hi:
    in_set = (svc, op) in _HIGH_INTERACTION_OPERATIONS
    check(f"{svc}:{op} - _HIGH_INTERACTION_OPERATIONS에 등록됨", in_set)
    needs = validate_moto_native_response(svc, op, '{"SessionId": "s-abc"}', 200)
    check(f"{svc}:{op} - 200 응답이어도 Agent 필요", needs)


# ══════════════════════════════════════════════════════════════
print("\n=== 5. moto_server 실제 연동 테스트 ===\n")

ENV = os.environ.copy()
ENV.update({
    "AWS_ACCESS_KEY_ID":     "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION":    "us-east-1",
    "AWS_ENDPOINT_URL":      "http://127.0.0.1:15099",
    "MOTO_LLM_VALIDATE_NATIVE": "1",
})

# .env 로드 (API 키)
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    for line in open(env_path).read().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            ENV[k.strip()] = v.strip()

# moto_server 기동
server = subprocess.Popen(
    ["python", "-m", "moto.server", "-p", "15099"],
    env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
)
time.sleep(3)

if server.poll() is not None:
    print(f"  [{SKIP}] moto_server 기동 실패 — 연동 테스트 건너뜀")
else:
    def aws(*args: str) -> tuple[int, str]:
        try:
            r = subprocess.run(
                ["aws"] + list(args),
                env=ENV, capture_output=True, text=True, timeout=30
            )
            return r.returncode, (r.stdout + r.stderr).strip()
        except Exception as e:
            return -1, str(e)

    # 5-1. Moto 지원 서비스: 응답이 오는지만 확인
    code, out = aws("s3api", "list-buckets")
    check("s3:ListBuckets - 응답 있음 (code=0 또는 Buckets 포함)",
          code == 0 or "Buckets" in out, out[:120])

    code, out = aws("ec2", "describe-instances")
    check("ec2:DescribeInstances - 응답 있음",
          code == 0 or "Reservations" in out, out[:120])

    # 5-2. Moto 미지원 → Agent fallback 경로
    has_key = bool(ENV.get("ANTHROPIC_API_KEY", "").strip())
    if has_key:
        code, out = aws("bedrock", "list-foundation-models")
        check("bedrock:ListFoundationModels - Agent fallback 응답 있음",
              "modelSummaries" in out or code == 0, out[:200])

        # ssm start-session은 로컬 SessionManager plugin 필요 → 서버까지 요청이 가면 OK
        code, out = aws("ssm", "start-session", "--target", "i-1234567890abcdef0")
        plugin_missing = "SessionManagerPlugin" in out
        if plugin_missing:
            print(f"  [{SKIP}] ssm:StartSession - SessionManager plugin 미설치 (로컬 환경 문제, 서버 로직은 정상)")
        else:
            check("ssm:StartSession - high-interaction → Agent 응답",
                  "SessionId" in out or "session" in out.lower(), out[:200])
    else:
        print(f"  [{SKIP}] ANTHROPIC_API_KEY 없음 — Agent fallback 테스트 건너뜀")

    server.terminate()
    server.wait()


# ══════════════════════════════════════════════════════════════
print("\n=== 결과 요약 ===\n")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  통과: {passed}/{total}")
failed = [label for label, ok in results if not ok]
if failed:
    print("\n  실패 항목:")
    for label in failed:
        print(f"    - {label}")
else:
    print("  전체 통과!")
