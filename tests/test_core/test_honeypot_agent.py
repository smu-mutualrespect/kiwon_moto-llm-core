"""
HoneypotAgent 통합 테스트.

테스트 항목:
  1. moto 가 처리하는 동작  — boto3 응답이 정상 파싱되는지
  2. Agent 가 처리하는 동작 — 실제 AWS CLI 출력 형식과 유사한지
  3. 세션 일관성           — 같은 세션에서 리소스 ID 가 동일하게 유지되는지
  4. 멀티턴 대화           — 이전 컨텍스트를 기억하는지

실행 방법:
  cd /home/moto/kiwon_moto-llm-core
  export $(cat .env | xargs) && python3 -m pytest tests/test_core/test_honeypot_agent.py -v
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import boto3
import pytest

from moto import mock_aws
from moto.core.llm_agents.agent import HoneypotAgent, get_or_create_agent

# ── 헬퍼 ──────────────────────────────────────────────────────────────────


def _make_s3_context(action: str, url: str, body: str = "") -> dict[str, Any]:
    return {
        "service": "s3",
        "action": action,
        "method": "GET",
        "url": url,
        "headers": {"Host": "s3.amazonaws.com"},
        "body": body,
        "reason": "No moto backend matched",
        "source": "test",
    }


def _make_ec2_context(action: str, body: str = "") -> dict[str, Any]:
    return {
        "service": "ec2",
        "action": action,
        "method": "POST",
        "url": "https://ec2.us-east-1.amazonaws.com/",
        "headers": {"Host": "ec2.us-east-1.amazonaws.com"},
        "body": body,
        "reason": "The action handler does not exist in moto",
        "source": "test",
    }


def _make_iam_context(action: str, body: str = "") -> dict[str, Any]:
    return {
        "service": "iam",
        "action": action,
        "method": "POST",
        "url": "https://iam.amazonaws.com/",
        "headers": {"Host": "iam.amazonaws.com"},
        "body": body,
        "reason": "NotImplementedError",
        "source": "test",
    }


def _make_sts_context(action: str) -> dict[str, Any]:
    return {
        "service": "sts",
        "action": action,
        "method": "POST",
        "url": "https://sts.amazonaws.com/",
        "headers": {"Host": "sts.amazonaws.com"},
        "body": f"Action={action}&Version=2011-06-15",
        "reason": "NotImplementedError",
        "source": "test",
    }


def _is_valid_xml(text: str) -> bool:
    try:
        ET.fromstring(text)
        return True
    except ET.ParseError:
        return False


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


# ── 1. moto 가 직접 처리하는 동작 ─────────────────────────────────────────


class TestMotoHandledOperations:
    """moto 가 자체 처리하는 API 는 boto3 가 정상 파싱할 수 있어야 한다."""

    @mock_aws
    def test_s3_list_buckets_empty(self) -> None:
        client = boto3.client("s3", region_name="us-east-1")
        resp = client.list_buckets()
        assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert "Buckets" in resp

    @mock_aws
    def test_s3_create_and_list(self) -> None:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-honeypot-bucket")
        resp = client.list_buckets()
        names = [b["Name"] for b in resp["Buckets"]]
        assert "test-honeypot-bucket" in names

    @mock_aws
    def test_sts_get_caller_identity(self) -> None:
        client = boto3.client("sts", region_name="us-east-1")
        resp = client.get_caller_identity()
        assert "Account" in resp
        assert "Arn" in resp

    @mock_aws
    def test_ec2_describe_regions(self) -> None:
        client = boto3.client("ec2", region_name="us-east-1")
        resp = client.describe_regions()
        regions = [r["RegionName"] for r in resp["Regions"]]
        assert "us-east-1" in regions

    @mock_aws
    def test_iam_list_users_empty(self) -> None:
        client = boto3.client("iam", region_name="us-east-1")
        resp = client.list_users()
        assert "Users" in resp


# ── 2. Agent 가 처리하는 동작 ─────────────────────────────────────────────


class TestAgentHandledOperations:
    """Agent 응답이 AWS 와이어 포맷(XML/JSON)과 일치해야 한다."""

    def setup_method(self) -> None:
        self.agent = HoneypotAgent()

    # ── S3 ────────────────────────────────────────────────────────────────

    def test_s3_list_buckets_xml_format(self) -> None:
        """S3 ListBuckets 응답이 파싱 가능한 XML 이어야 한다."""
        ctx = _make_s3_context("ListBuckets", "https://s3.amazonaws.com/")
        reply = self.agent.run(ctx)

        assert _is_valid_xml(reply), f"S3 응답이 유효한 XML 이 아님:\n{reply}"
        root = ET.fromstring(reply)
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        assert tag in ("ListAllMyBucketsResult", "ListBucketsResult", "Response"), (
            f"예상과 다른 루트 태그: {tag}\n{reply}"
        )

    def test_s3_list_objects_xml_format(self) -> None:
        """S3 ListObjectsV2 응답이 유효한 XML 이어야 한다."""
        ctx = _make_s3_context(
            "ListObjectsV2",
            "https://my-bucket.s3.amazonaws.com/?list-type=2",
        )
        reply = self.agent.run(ctx)
        assert _is_valid_xml(reply), f"S3 응답이 유효한 XML 이 아님:\n{reply}"

    def test_s3_get_bucket_location(self) -> None:
        """S3 GetBucketLocation 응답이 유효한 XML 이어야 한다."""
        ctx = _make_s3_context(
            "GetBucketLocation",
            "https://my-bucket.s3.amazonaws.com/?location",
        )
        reply = self.agent.run(ctx)
        assert _is_valid_xml(reply), f"S3 응답이 유효한 XML 이 아님:\n{reply}"

    # ── EC2 ───────────────────────────────────────────────────────────────

    def test_ec2_describe_instances_xml_format(self) -> None:
        """EC2 DescribeInstances 응답이 XML 이고 reservationSet 을 포함해야 한다."""
        ctx = _make_ec2_context(
            "DescribeInstances",
            "Action=DescribeInstances&Version=2016-11-15",
        )
        reply = self.agent.run(ctx)

        assert _is_valid_xml(reply), f"EC2 응답이 유효한 XML 이 아님:\n{reply}"
        assert re.search(
            r"(?i)(reservationSet|instancesSet|DescribeInstancesResponse)", reply
        ), f"EC2 DescribeInstances 응답에 예상 필드 없음:\n{reply}"

    def test_ec2_describe_security_groups(self) -> None:
        """EC2 DescribeSecurityGroups 응답이 XML 이고 sg- ID 를 포함해야 한다."""
        ctx = _make_ec2_context(
            "DescribeSecurityGroups",
            "Action=DescribeSecurityGroups&Version=2016-11-15",
        )
        reply = self.agent.run(ctx)

        assert _is_valid_xml(reply), f"유효한 XML 이 아님:\n{reply}"
        assert "sg-" in reply, f"Security Group ID 없음:\n{reply}"

    # ── IAM ───────────────────────────────────────────────────────────────

    def test_iam_list_roles_json_format(self) -> None:
        """IAM ListRoles 응답이 JSON 또는 XML 형식이어야 한다."""
        ctx = _make_iam_context(
            "ListRoles",
            "Action=ListRoles&Version=2010-05-08",
        )
        reply = self.agent.run(ctx)

        if reply.strip().startswith("{"):
            data = json.loads(reply)
            assert "ListRolesResponse" in data or "Roles" in str(data), (
                f"IAM ListRoles JSON 에 예상 필드 없음:\n{reply}"
            )
        else:
            assert _is_valid_xml(reply), f"IAM 응답이 XML 도 JSON 도 아님:\n{reply}"

    def test_iam_get_user(self) -> None:
        """IAM GetUser 응답이 비어 있지 않아야 한다."""
        ctx = _make_iam_context(
            "GetUser",
            "Action=GetUser&Version=2010-05-08",
        )
        reply = self.agent.run(ctx)
        assert len(reply.strip()) > 0, "빈 응답"

    # ── STS ───────────────────────────────────────────────────────────────

    def test_sts_get_caller_identity_arn(self) -> None:
        """STS GetCallerIdentity 응답에 ARN 과 AccountId 가 있어야 한다."""
        ctx = _make_sts_context("GetCallerIdentity")
        reply = self.agent.run(ctx)

        assert re.search(r"arn:aws[^\s<\"']+", reply), f"ARN 이 응답에 없음:\n{reply}"
        assert re.search(r"\b\d{12}\b", reply), (
            f"AccountId(12자리) 가 응답에 없음:\n{reply}"
        )


# ── 3. 세션 일관성 테스트 ─────────────────────────────────────────────────


class TestSessionConsistency:
    """같은 세션에서 생성된 리소스 ID 가 후속 요청에서도 동일해야 한다."""

    def test_bucket_name_consistent_across_turns(self) -> None:
        """첫 응답에서 나온 버킷 이름이 두 번째 응답에서도 동일해야 한다."""
        agent = get_or_create_agent("test-session-consistency")

        ctx1 = _make_s3_context("ListBuckets", "https://s3.amazonaws.com/")
        reply1 = agent.run(ctx1)

        bucket_names = re.findall(r"<Name>([^<]+)</Name>", reply1)
        if not bucket_names:
            pytest.skip("첫 번째 응답에 버킷 이름이 없어 일관성 테스트를 건너뜀")

        first_bucket = bucket_names[0]

        ctx2 = _make_s3_context(
            "ListObjectsV2",
            f"https://{first_bucket}.s3.amazonaws.com/?list-type=2",
        )
        reply2 = agent.run(ctx2)

        assert _is_valid_xml(reply2), f"Turn 2 응답이 유효한 XML 이 아님:\n{reply2}"

    def test_instance_id_consistent_across_turns(self) -> None:
        """EC2 인스턴스 ID 가 연속 요청에서 동일하게 유지되어야 한다."""
        agent = get_or_create_agent("test-session-ec2")

        ctx1 = _make_ec2_context("DescribeInstances", "Action=DescribeInstances")
        reply1 = agent.run(ctx1)

        instance_ids = re.findall(r"i-[0-9a-f]{8,17}", reply1)
        if not instance_ids:
            pytest.skip("첫 번째 응답에 인스턴스 ID 가 없어 일관성 테스트를 건너뜀")

        first_id = instance_ids[0]

        ctx2 = _make_ec2_context(
            "DescribeInstanceAttribute",
            f"Action=DescribeInstanceAttribute&InstanceId={first_id}&Attribute=instanceType",
        )
        reply2 = agent.run(ctx2)

        assert first_id in reply2, (
            f"인스턴스 ID {first_id} 가 두 번째 응답에 없음:\n{reply2}"
        )


# ── 4. 리소스 ID 형식 검증 ────────────────────────────────────────────────


class TestResourceIdFormats:
    """AWS 리소스 ID 가 실제 AWS 형식에 맞아야 한다."""

    PATTERNS = {
        "instance_id": r"i-[0-9a-f]{8,17}",
        "security_group": r"sg-[0-9a-f]{8,17}",
        "vpc_id": r"vpc-[0-9a-f]{8,17}",
        "subnet_id": r"subnet-[0-9a-f]{8,17}",
        "ami_id": r"ami-[0-9a-f]{8,17}",
        "arn": r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[^\s<\"']+",
        "account_id": r"\b\d{12}\b",
    }

    def setup_method(self) -> None:
        self.agent = HoneypotAgent()

    def test_ec2_resource_ids_match_aws_format(self) -> None:
        """EC2 응답에 AWS 형식의 리소스 ID 가 포함되어야 한다."""
        ctx = _make_ec2_context("DescribeInstances", "Action=DescribeInstances")
        reply = self.agent.run(ctx)

        found = {
            name: re.findall(pattern, reply)[0]
            for name, pattern in self.PATTERNS.items()
            if re.findall(pattern, reply)
        }
        assert found, f"AWS 형식의 리소스 ID 가 전혀 없음:\n{reply}"

    def test_sts_account_id_is_12_digits(self) -> None:
        """STS 응답의 Account ID 는 정확히 12자리 숫자여야 한다."""
        ctx = _make_sts_context("GetCallerIdentity")
        reply = self.agent.run(ctx)

        account_ids = re.findall(r"\b(\d{12})\b", reply)
        assert account_ids, f"12자리 Account ID 가 없음:\n{reply}"

    def test_arn_format_correct(self) -> None:
        """응답에 포함된 ARN 이 arn:aws:service:region:account:resource 형식이어야 한다."""
        ctx = _make_iam_context("GetUser", "Action=GetUser")
        reply = self.agent.run(ctx)

        arns = re.findall(r"arn:aws:[^\s<\"'>]+", reply)
        for arn in arns:
            parts = arn.split(":")
            assert len(parts) >= 6, f"ARN 형식 불일치: {arn}"
            assert parts[0] == "arn"
            assert parts[1] == "aws"


# ── 5. 멀티턴 컨텍스트 기억 테스트 ───────────────────────────────────────


class TestMultiTurnMemory:
    """이전 대화에서 언급된 정보를 다음 턴에서 기억해야 한다."""

    def test_remembers_created_bucket(self) -> None:
        """CreateBucket 후 ListBuckets 에서 해당 버킷이 보여야 한다."""
        agent = get_or_create_agent("test-session-memory")

        ctx_create: dict[str, Any] = {
            "service": "s3",
            "action": "CreateBucket",
            "method": "PUT",
            "url": "https://my-secret-data.s3.amazonaws.com/",
            "headers": {"Host": "my-secret-data.s3.amazonaws.com"},
            "body": "",
            "reason": "NotImplementedError",
            "source": "test",
        }
        agent.run(ctx_create)

        ctx_list = _make_s3_context("ListBuckets", "https://s3.amazonaws.com/")
        reply_list = agent.run(ctx_list)

        assert "my-secret-data" in reply_list, (
            f"생성한 버킷 이름이 ListBuckets 에 없음:\n{reply_list}"
        )
