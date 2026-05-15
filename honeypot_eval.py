#!/usr/bin/env python3
"""
AWS Honeypot Response Quality Evaluator
허니팟 응답을 AWS 공식 스키마와 비교해 유사도 점수를 측정합니다.

점수 구성 (가중치):
  Schema   35% — botocore output shape 기준 필드 존재·키 유효성·타입
  Format   35% — ARN 구조·리소스 ID 패턴·계정 ID·리전 형식
  Semantic 30% — botocore enum 값·실존 인스턴스 타입·실존 모델 ID 등

사용:
  python honeypot_eval.py            # 표준 출력
  python honeypot_eval.py --json     # + honeypot_eval_result.json 저장
  python honeypot_eval.py --detail   # 각 명령어 응답 원문 포함 출력
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ── botocore ──────────────────────────────────────────────────────────────────
try:
    import botocore.session as _bsession
    _SESSION = _bsession.get_session()
    HAS_BOTOCORE = True
except ImportError:
    _SESSION = None
    HAS_BOTOCORE = False

# ── 실행 환경 ─────────────────────────────────────────────────────────────────
ENDPOINT = "http://127.0.0.1:5000"
AWS_BIN  = "/home/moto/honeypot-env/bin/aws"
BASE_ENV = {
    "AWS_ACCESS_KEY_ID":     "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION":    "us-east-1",
}

# ── 유효 값 집합 ──────────────────────────────────────────────────────────────
VALID_REGIONS: set[str] = {
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1",
    "eu-north-1", "eu-south-1",
    "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
    "ap-southeast-1", "ap-southeast-2", "ap-south-1", "ap-east-1",
    "sa-east-1", "ca-central-1", "me-south-1", "af-south-1",
}

VALID_INSTANCE_TYPES: set[str] = {
    "t2.micro", "t2.small", "t2.medium", "t2.large", "t2.xlarge", "t2.2xlarge",
    "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge", "t3.2xlarge",
    "t3a.micro", "t3a.small", "t3a.medium", "t3a.large",
    "m4.large", "m4.xlarge", "m4.2xlarge", "m4.4xlarge", "m4.10xlarge",
    "m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge", "m5.8xlarge", "m5.12xlarge",
    "m5a.large", "m5a.xlarge", "m5a.2xlarge",
    "m6i.large", "m6i.xlarge", "m6i.2xlarge", "m6i.4xlarge",
    "m6a.large", "m6a.xlarge", "m6a.2xlarge",
    "c4.large", "c4.xlarge", "c4.2xlarge", "c4.4xlarge",
    "c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge",
    "c5a.large", "c5a.xlarge", "c5a.2xlarge",
    "c6i.large", "c6i.xlarge", "c6i.2xlarge",
    "r5.large", "r5.xlarge", "r5.2xlarge", "r5.4xlarge",
    "r6i.large", "r6i.xlarge", "r6i.2xlarge",
    "i3.large", "i3.xlarge", "i3.2xlarge", "i3.4xlarge",
    "p2.xlarge", "p2.8xlarge", "p3.2xlarge", "p3.8xlarge",
    "g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge",
}

VALID_BEDROCK_MODEL_IDS: set[str] = {
    "amazon.titan-text-express-v1", "amazon.titan-text-lite-v1",
    "amazon.titan-text-premier-v1:0", "amazon.titan-embed-text-v1",
    "amazon.titan-embed-image-v1", "amazon.titan-image-generator-v1",
    "anthropic.claude-instant-v1", "anthropic.claude-v2", "anthropic.claude-v2:1",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "ai21.j2-mid-v1", "ai21.j2-ultra-v1",
    "cohere.command-text-v14", "cohere.command-light-text-v14",
    "cohere.command-r-v1:0", "cohere.command-r-plus-v1:0",
    "cohere.embed-english-v3", "cohere.embed-multilingual-v3",
    "meta.llama2-13b-chat-v1", "meta.llama2-70b-chat-v1",
    "meta.llama3-8b-instruct-v1:0", "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2", "mistral.mixtral-8x7b-instruct-v0:1",
    "stability.stable-diffusion-xl-v1",
}

VALID_PLATFORM_TYPES: set[str] = {"Windows", "Linux"}
VALID_PING_STATUS:    set[str] = {"Online", "ConnectionLost", "Inactive"}

# ── ARN 형식 규칙 ─────────────────────────────────────────────────────────────
# AWS 서비스별로 ARN 내 region/account 존재 여부가 다름
_ARN_RE = re.compile(r'^arn:(aws[a-z-]*):([\w-]+):([a-z0-9-]*):(\d*):(.+)$')

_ARN_SERVICE_RULES: dict[str, dict[str, bool]] = {
    "iam":            {"has_region": False, "has_account": True},
    "sts":            {"has_region": False, "has_account": True},
    "s3":             {"has_region": False, "has_account": False},
    "bedrock":        {"has_region": True,  "has_account": False},  # foundation-model
    "ec2":            {"has_region": True,  "has_account": True},
    "ssm":            {"has_region": True,  "has_account": True},
    "ecr":            {"has_region": True,  "has_account": True},
    "secretsmanager": {"has_region": True,  "has_account": True},
    "lambda":         {"has_region": True,  "has_account": True},
    "kms":            {"has_region": True,  "has_account": True},
}

# ── 리소스 ID 패턴 ────────────────────────────────────────────────────────────
_RESOURCE_ID_PATS: dict[str, re.Pattern] = {
    "instanceid":         re.compile(r'^i-[0-9a-f]{8,17}$'),
    "volumeid":           re.compile(r'^vol-[0-9a-f]{8,17}$'),
    "groupid":            re.compile(r'^sg-[0-9a-f]{8,17}$'),
    "securitygroupid":    re.compile(r'^sg-[0-9a-f]{8,17}$'),
    "subnetid":           re.compile(r'^subnet-[0-9a-f]{8,17}$'),
    "vpcid":              re.compile(r'^vpc-[0-9a-f]{8,17}$'),
    "snapshotid":         re.compile(r'^snap-[0-9a-f]{8,17}$'),
    "imageid":            re.compile(r'^ami-[0-9a-f]{8,17}$'),
    "uploadid":           re.compile(r'^[0-9a-f\-]{8,64}$'),
    "layerdigest":        re.compile(r'^sha256:[0-9a-f]{64}$'),
    "reservedinstancesid": re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    ),
}

_ACCOUNT_ID_FIELDS: set[str] = {"accountid", "ownerid", "awsaccountid", "customerid"}
_REGION_FIELDS:     set[str] = {"region", "availabilityzone", "availabilityzoneid"}

# ── 테스트 케이스 정의 ────────────────────────────────────────────────────────
TEST_CASES: list[tuple[str, str, list[str]]] = [
    ("bedrock", "ListFoundationModels",
     ["bedrock", "list-foundation-models"]),
    ("ec2", "MonitorInstances",
     ["ec2", "monitor-instances", "--instance-ids", "i-1234567890abcdef0"]),
    ("ec2", "UnmonitorInstances",
     ["ec2", "unmonitor-instances", "--instance-ids", "i-1234567890abcdef0"]),
    ("ec2", "DescribeReservedInstances",
     ["ec2", "describe-reserved-instances"]),
    ("ec2", "DescribeReservedInstancesListings",
     ["ec2", "describe-reserved-instances-listings"]),
    ("ec2", "PurchaseReservedInstancesOffering",
     ["ec2", "purchase-reserved-instances-offering",
      "--reserved-instances-offering-id", "aaaaaa11-bbbb-cccc-ddd-example1",
      "--instance-count", "1"]),
    ("ec2", "DescribeVolumeStatus",
     ["ec2", "describe-volume-status", "--volume-ids", "vol-1234567890abcdef0"]),
    ("ec2", "ModifyVolumeAttribute",
     ["ec2", "modify-volume-attribute",
      "--volume-id", "vol-1234567890abcdef0", "--auto-enable-io"]),
    ("ec2", "CreateSpotDatafeedSubscription",
     ["ec2", "create-spot-datafeed-subscription", "--bucket", "honeypot-ki"]),
    ("ec2", "DescribeBundleTasks",
     ["ec2", "describe-bundle-tasks"]),
    ("ssm", "DescribeInstanceInformation",
     ["ssm", "describe-instance-information"]),
    ("ecr", "BatchCheckLayerAvailability",
     ["ecr", "batch-check-layer-availability",
      "--repository-name", "demo",
      "--layer-digests",
      "sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"]),
    ("ecr", "GetDownloadUrlForLayer",
     ["ecr", "get-download-url-for-layer",
      "--repository-name", "demo",
      "--layer-digest",
      "sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"]),
    ("ecr", "InitiateLayerUpload",
     ["ecr", "initiate-layer-upload", "--repository-name", "demo"]),
    ("ecr", "CompleteLayerUpload",
     ["ecr", "complete-layer-upload",
      "--repository-name", "demo",
      "--upload-id", "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "--layer-digests",
      "sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"]),
    ("iam", "GetContextKeysForPrincipalPolicy",
     ["iam", "get-context-keys-for-principal-policy",
      "--policy-source-arn", "arn:aws:iam::123456789012:user/victim-admin"]),
    ("iam", "ListServiceSpecificCredentials",
     ["iam", "list-service-specific-credentials", "--user-name", "victim-admin"]),
    ("iam", "GenerateServiceLastAccessedDetails",
     ["iam", "generate-service-last-accessed-details",
      "--arn", "arn:aws:iam::123456789012:user/victim-admin"]),
    ("secretsmanager", "ValidateResourcePolicy",
     ["secretsmanager", "validate-resource-policy",
      "--secret-id", "prod/db/password",
      "--resource-policy",
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*",'
      '"Action":"secretsmanager:GetSecretValue","Resource":"*"}]}']),
    ("sts", "DecodeAuthorizationMessage",
     ["sts", "decode-authorization-message",
      "--encoded-message", "ZmFrZS1hdXRob3JpemF0aW9uLW1lc3NhZ2U="]),
]


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass
class DimScore:
    score:  float = 1.0
    checks: int   = 0
    passes: int   = 0
    issues: list[str] = field(default_factory=list)

    def record(self, passed: bool, issue: str = "") -> None:
        self.checks += 1
        if passed:
            self.passes += 1
        elif issue:
            self.issues.append(issue)

    def finalize(self) -> None:
        self.score = self.passes / self.checks if self.checks else 1.0


@dataclass
class EvalResult:
    service:          str
    operation:        str
    schema:           DimScore
    fmt:              DimScore
    semantic:         DimScore
    total_score:      float
    response_time_ms: int
    raw_response:     Any
    error:            Optional[str] = None


# ── CLI 실행 ──────────────────────────────────────────────────────────────────
def _run_command(
    cli_args: list[str],
) -> tuple[Optional[Any], int, Optional[str]]:
    cmd = [AWS_BIN, "--endpoint-url", ENDPOINT] + cli_args
    env = {**os.environ, **BASE_ENV}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=90
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        if proc.returncode != 0:
            return None, elapsed, proc.stderr.strip()
        stdout = proc.stdout.strip()
        # 일부 AWS 작업(ModifyVolumeAttribute 등)은 성공해도 빈 응답을 반환함
        if not stdout:
            return {}, elapsed, None
        return json.loads(stdout), elapsed, None
    except subprocess.TimeoutExpired:
        return None, 90_000, "timeout (90s)"
    except Exception as exc:
        return None, 0, str(exc)


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _iter_leaves(obj: Any, depth: int = 0) -> list[tuple[str, Any]]:
    """dict/list를 재귀 순회하며 (key, 리프값) 쌍 반환 (최대 깊이 6)."""
    if depth > 6:
        return []
    results: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)):
                results.append((k, v))
            else:
                results.extend(_iter_leaves(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_iter_leaves(item, depth + 1))
    return results


# ── Schema 점수 ───────────────────────────────────────────────────────────────
def _score_schema(service: str, operation: str, data: Any) -> DimScore:
    dim = DimScore()
    if not HAS_BOTOCORE or not isinstance(data, dict):
        dim.score = 1.0
        return dim

    try:
        svc   = _SESSION.get_service_model(service)
        op    = svc.operation_model(operation)
        shape = op.output_shape
    except Exception:
        dim.score = 1.0
        return dim

    if shape is None:
        dim.score = 1.0
        return dim

    # 필수 필드 존재 여부
    required = list(getattr(shape, "required_members", None) or [])
    for f in required:
        dim.record(
            f in data,
            issue=f"필수 필드 누락: {f}",
        )

    # 응답 키가 스키마에 있는지
    valid_keys = set(shape.members.keys())
    for k in data:
        dim.record(
            k in valid_keys,
            issue=f"스키마에 없는 키: {k}",
        )

    # 빈 응답 체크 (output_shape에 members가 있는데 {} 이면 감점)
    if shape.members:
        dim.record(bool(data), issue="응답이 빈 객체 {}")

    dim.finalize()
    return dim


# ── Format 점수 ───────────────────────────────────────────────────────────────
def _check_arn(value: str, field_name: str) -> Optional[str]:
    """ARN 형식 검사. 오류 메시지 반환, 정상이면 None."""
    m = _ARN_RE.match(value)
    if not m:
        return f"{field_name}: ARN 구문 불일치 → '{value[:60]}'"
    arn_service = m.group(2)
    arn_region  = m.group(3)
    arn_account = m.group(4)
    rules = _ARN_SERVICE_RULES.get(arn_service)
    if rules is None:
        return None  # 알 수 없는 서비스는 통과
    if not rules["has_region"] and arn_region:
        return f"{field_name}: {arn_service} ARN에 region 있으면 안 됨 ('{arn_region}')"
    if rules["has_region"] and not arn_region:
        return f"{field_name}: {arn_service} ARN에 region 없음"
    if not rules["has_account"] and arn_account:
        return f"{field_name}: {arn_service} ARN에 account 있으면 안 됨 ('{arn_account}')"
    if rules["has_account"] and not arn_account:
        return f"{field_name}: {arn_service} ARN에 account 없음"
    return None


def _score_format(data: Any) -> DimScore:
    dim = DimScore()
    if not isinstance(data, dict):
        dim.score = 1.0
        return dim

    for key, value in _iter_leaves(data):
        if not isinstance(value, str) or not value:
            continue
        low = key.lower()

        # ARN 검사
        if value.startswith("arn:aws"):
            err = _check_arn(value, key)
            dim.record(err is None, issue=err or "")
            continue

        # 리소스 ID 패턴 검사
        pat = _RESOURCE_ID_PATS.get(low)
        if pat:
            dim.record(
                bool(pat.match(value)),
                issue=f"{key}: 리소스 ID 형식 불일치 → '{value}'",
            )
            continue

        # 계정 ID (12자리 숫자)
        if low in _ACCOUNT_ID_FIELDS:
            dim.record(
                bool(re.match(r'^\d{12}$', value)),
                issue=f"{key}: 계정 ID가 12자리 숫자가 아님 → '{value}'",
            )
            continue

        # 리전 이름 (AZ 구분: us-east-1a 같은 건 제외)
        if low in _REGION_FIELDS or low.endswith("region"):
            if re.match(r'^[a-z]+-[a-z]+-\d[a-z]$', value):
                continue  # AZ (us-east-1a)
            dim.record(
                value in VALID_REGIONS,
                issue=f"{key}: 유효하지 않은 리전 → '{value}'",
            )
            continue

    dim.finalize()
    return dim


# ── Semantic 점수 ─────────────────────────────────────────────────────────────
def _check_enums(shape: Any, data: Any, dim: DimScore, depth: int = 0) -> None:
    """botocore output shape를 기준으로 enum 제약 위반을 재귀 검사."""
    if depth > 4 or not hasattr(shape, "type_name"):
        return
    if shape.type_name == "structure" and isinstance(data, dict):
        for name, member_shape in shape.members.items():
            if name in data:
                _check_enums(member_shape, data[name], dim, depth + 1)
    elif shape.type_name == "list" and isinstance(data, list):
        for item in data[:5]:
            _check_enums(shape.member, item, dim, depth + 1)
    elif shape.type_name == "string":
        enums = shape.metadata.get("enum")
        if enums and isinstance(data, str) and data:
            dim.record(
                data in enums,
                issue=f"enum 범위 초과: '{data}' (허용값: {enums[:3]}...)",
            )


def _score_semantic(service: str, operation: str, data: Any) -> DimScore:
    dim = DimScore()
    if not isinstance(data, dict):
        dim.score = 1.0
        return dim

    # botocore enum 검사
    if HAS_BOTOCORE:
        try:
            svc   = _SESSION.get_service_model(service)
            op    = svc.operation_model(operation)
            shape = op.output_shape
            if shape:
                _check_enums(shape, data, dim)
        except Exception:
            pass

    # 서비스 특화 의미 검사
    for key, value in _iter_leaves(data):
        if not isinstance(value, str) or not value:
            continue
        low = key.lower()

        if low == "instancetype":
            dim.record(
                value in VALID_INSTANCE_TYPES,
                issue=f"InstanceType: 알 수 없는 타입 → '{value}'",
            )

        elif low == "modelid" and service == "bedrock":
            dim.record(
                value in VALID_BEDROCK_MODEL_IDS,
                issue=f"modelId: 알 수 없는 Bedrock 모델 → '{value}'",
            )

        elif low == "platformtype":
            dim.record(
                value in VALID_PLATFORM_TYPES,
                issue=f"PlatformType: 유효하지 않음 → '{value}'",
            )

        elif low == "pingstatus":
            dim.record(
                value in VALID_PING_STATUS,
                issue=f"PingStatus: 유효하지 않음 → '{value}'",
            )

        # 너무 짧거나 hex가 그대로 노출된 값 (key:key 유형) 감지
        elif len(value) < 3 and low not in {"state", "type", "os"}:
            pass  # 짧아도 enum이면 이미 위에서 처리
        elif value.lower() == low:
            # key 이름과 완전히 동일한 값 (key:key 품질 버그)
            dim.record(
                False,
                issue=f"key=value 품질 버그: '{key}': '{value}'",
            )

    dim.finalize()
    return dim


# ── 통합 평가 ─────────────────────────────────────────────────────────────────
def evaluate(service: str, operation: str, cli_args: list[str]) -> EvalResult:
    data, elapsed, error = _run_command(cli_args)

    if error or data is None:
        return EvalResult(
            service=service, operation=operation,
            schema=DimScore(0.0), fmt=DimScore(0.0), semantic=DimScore(0.0),
            total_score=0.0, response_time_ms=elapsed,
            raw_response=None, error=error,
        )

    schema   = _score_schema(service, operation, data)
    fmt      = _score_format(data)
    semantic = _score_semantic(service, operation, data)
    total    = 0.35 * schema.score + 0.35 * fmt.score + 0.30 * semantic.score

    return EvalResult(
        service=service, operation=operation,
        schema=schema, fmt=fmt, semantic=semantic,
        total_score=total, response_time_ms=elapsed,
        raw_response=data,
    )


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────────
def _bar(score: float, width: int = 18) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    if score >= 0.90: return "A"
    if score >= 0.80: return "B"
    if score >= 0.70: return "C"
    if score >= 0.60: return "D"
    return "F"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    show_detail = "--detail" in sys.argv
    save_json   = "--json"   in sys.argv

    print("=" * 78)
    print("  AWS Honeypot Response Quality Evaluator")
    print("=" * 78)
    print(
        f"\n{'#':>3}  {'Service:Operation':<42}  "
        f"{'Schema':>7}  {'Format':>7}  {'Semantic':>8}  {'Total':>7}  Grade"
    )
    print("-" * 84)

    results: list[EvalResult] = []
    for idx, (service, operation, cli_args) in enumerate(TEST_CASES, 1):
        label = f"{service}:{operation}"
        print(f"  {idx:>2}  {label:<42}  ", end="", flush=True)
        r = evaluate(service, operation, cli_args)
        results.append(r)
        if r.error:
            print(f"  ✗ ERROR: {r.error[:45]}")
        else:
            print(
                f"{_pct(r.schema.score):>7}  "
                f"{_pct(r.fmt.score):>7}  "
                f"{_pct(r.semantic.score):>8}  "
                f"{_pct(r.total_score):>7}  "
                f"  {_grade(r.total_score)}"
            )

    # ── 요약 ─────────────────────────────────────────────────────────────────
    ok = [r for r in results if not r.error]
    print("\n" + "=" * 78)
    if ok:
        avg_schema   = sum(r.schema.score   for r in ok) / len(ok)
        avg_fmt      = sum(r.fmt.score      for r in ok) / len(ok)
        avg_semantic = sum(r.semantic.score for r in ok) / len(ok)
        avg_total    = sum(r.total_score    for r in ok) / len(ok)
        avg_time     = sum(r.response_time_ms for r in ok) / len(ok)

        print(f"\n  전체 평균  ({len(ok)}/{len(results)} 성공)")
        print(f"  Schema   [{_bar(avg_schema)}] {_pct(avg_schema)}")
        print(f"  Format   [{_bar(avg_fmt)}] {_pct(avg_fmt)}")
        print(f"  Semantic [{_bar(avg_semantic)}] {_pct(avg_semantic)}")
        print(f"  {'─'*50}")
        print(f"  Total    [{_bar(avg_total)}] {_pct(avg_total)}  [{_grade(avg_total)}]")
        print(f"\n  평균 응답 시간: {avg_time:,.0f} ms")

    # ── 이슈 상세 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  이슈 상세 리포트")
    print("=" * 78)
    has_any_issue = False
    for r in results:
        if r.error:
            print(f"\n  ▸ {r.service}:{r.operation}")
            print(f"    [ERROR] {r.error}")
            has_any_issue = True
            continue
        all_issues = r.schema.issues + r.fmt.issues + r.semantic.issues
        if all_issues:
            has_any_issue = True
            print(f"\n  ▸ {r.service}:{r.operation}  (총점 {_pct(r.total_score)})")
            schema_issues = r.schema.issues
            fmt_issues    = r.fmt.issues
            sem_issues    = r.semantic.issues
            if schema_issues:
                print(f"    [Schema  {_pct(r.schema.score)}]")
                for i in schema_issues[:4]:
                    print(f"      · {i}")
            if fmt_issues:
                print(f"    [Format  {_pct(r.fmt.score)}]")
                for i in fmt_issues[:4]:
                    print(f"      · {i}")
            if sem_issues:
                print(f"    [Semantic {_pct(r.semantic.score)}]")
                for i in sem_issues[:4]:
                    print(f"      · {i}")
    if not has_any_issue:
        print("\n  이슈 없음 — 모든 항목 통과")

    # ── 응답 원문 (--detail) ──────────────────────────────────────────────────
    if show_detail:
        print("\n" + "=" * 78)
        print("  응답 원문")
        print("=" * 78)
        for r in results:
            print(f"\n  [{r.service}:{r.operation}]")
            if r.error:
                print(f"    ERROR: {r.error}")
            else:
                print(json.dumps(r.raw_response, ensure_ascii=False, indent=4)[:1200])

    # ── JSON 저장 (--json) ────────────────────────────────────────────────────
    if save_json:
        out_path = "honeypot_eval_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "service":        r.service,
                        "operation":      r.operation,
                        "schema_score":   round(r.schema.score,   4),
                        "format_score":   round(r.fmt.score,      4),
                        "semantic_score": round(r.semantic.score, 4),
                        "total_score":    round(r.total_score,    4),
                        "response_time_ms": r.response_time_ms,
                        "issues": {
                            "schema":   r.schema.issues,
                            "format":   r.fmt.issues,
                            "semantic": r.semantic.issues,
                        },
                        "error": r.error,
                    }
                    for r in results
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n  JSON 결과 저장 완료: {out_path}")

    print()


if __name__ == "__main__":
    main()
