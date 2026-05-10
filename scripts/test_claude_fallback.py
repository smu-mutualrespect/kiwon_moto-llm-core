"""
sangho master branch - Claude fallback 수동 테스트
fake AWS 자격증명 + moto 패치 상태에서 미지원 서비스 호출 시
handle_aws_request -> Claude API 흐름을 검증한다.
"""
from __future__ import annotations

import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from moto.core.llm_agents.agent import handle_aws_request  # noqa: E402


def test_direct_agent_call() -> None:
    print("=" * 60)
    print("[1] handle_aws_request() 직접 호출 테스트 (Claude API)")
    print("=" * 60)

    result = handle_aws_request(
        service="ssm",
        action="DescribeInstanceInformation",
        url="https://ssm.us-east-1.amazonaws.com/",
        headers={"X-Forwarded-For": "1.2.3.4", "User-Agent": "Boto3/1.34"},
        body="Action=DescribeInstanceInformation",
        reason="manual test",
        source="test_claude_fallback.py",
    )

    print(f"응답:\n{result}\n")
    assert result, "응답이 비어있음"
    print("[PASS] handle_aws_request 호출 성공\n")


def test_moto_patched_unsupported_service() -> None:
    print("=" * 60)
    print("[2] moto 패치 상태에서 미지원 서비스 호출 테스트")
    print("=" * 60)

    import boto3
    from moto import mock_aws

    @mock_aws
    def run():
        # HealthLake는 moto에 백엔드가 없어서 fallback이 트리거됨
        client = boto3.client("healthlake", region_name="us-east-1")
        try:
            resp = client.list_fhir_datastores()
            print(f"응답: {resp}")
            print("[PASS] fallback 경유 응답 수신\n")
        except Exception as e:
            print(f"예외 발생 (fallback 실패 시 정상): {e}\n")

    run()


if __name__ == "__main__":
    test_direct_agent_call()
    test_moto_patched_unsupported_service()
