"""
허니팟 Agent 수동 테스트 스크립트.

실행:
  export $(cat .env | xargs) && python3 manual_test.py
"""
from __future__ import annotations

import json

import boto3

from moto import mock_aws

SEP = "-" * 60


@mock_aws
def test_moto_handled() -> None:
    """moto 가 직접 처리하는 명령 — AWS CLI 응답 형식 확인."""
    print(f"\n{'='*60}")
    print("[ moto 처리 구간 ]")
    print(f"{'='*60}")

    # S3 ListBuckets
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="honeypot-logs")
    s3.create_bucket(Bucket="app-assets-prod")
    resp = s3.list_buckets()
    print(f"\n$ aws s3 ls")
    for b in resp["Buckets"]:
        print(f"  {b['CreationDate'].strftime('%Y-%m-%d %H:%M:%S')}  {b['Name']}")

    # STS GetCallerIdentity
    sts = boto3.client("sts", region_name="us-east-1")
    resp = sts.get_caller_identity()
    print(f"\n$ aws sts get-caller-identity")
    print(f"  Account : {resp['Account']}")
    print(f"  Arn     : {resp['Arn']}")
    print(f"  UserId  : {resp['UserId']}")

    # EC2 DescribeRegions
    ec2 = boto3.client("ec2", region_name="us-east-1")
    resp = ec2.describe_regions()
    regions = [r["RegionName"] for r in resp["Regions"]]
    print(f"\n$ aws ec2 describe-regions --query 'Regions[].RegionName'")
    print(f"  {regions[:5]} ... ({len(regions)} regions total)")


@mock_aws
def test_agent_handled() -> None:
    """Agent 가 처리하는 명령 — LLM 이 AWS 서버처럼 응답하는지 확인."""
    print(f"\n{'='*60}")
    print("[ Agent 처리 구간 — LLM 호출 발생 ]")
    print(f"{'='*60}")

    from moto.core.llm_agents.agent import HoneypotAgent

    agent = HoneypotAgent()

    cases = [
        {
            "label": "$ aws ec2 describe-instances",
            "context": {
                "service": "ec2",
                "action": "DescribeInstances",
                "method": "POST",
                "url": "https://ec2.us-east-1.amazonaws.com/",
                "headers": {},
                "body": "Action=DescribeInstances&Version=2016-11-15",
                "reason": "NotImplementedError",
                "source": "manual_test",
            },
        },
        {
            "label": "$ aws iam get-user",
            "context": {
                "service": "iam",
                "action": "GetUser",
                "method": "POST",
                "url": "https://iam.amazonaws.com/",
                "headers": {},
                "body": "Action=GetUser&Version=2010-05-08",
                "reason": "NotImplementedError",
                "source": "manual_test",
            },
        },
        {
            "label": "$ aws sts get-caller-identity  (agent fallback)",
            "context": {
                "service": "sts",
                "action": "GetCallerIdentity",
                "method": "POST",
                "url": "https://sts.amazonaws.com/",
                "headers": {},
                "body": "Action=GetCallerIdentity&Version=2011-06-15",
                "reason": "NotImplementedError",
                "source": "manual_test",
            },
        },
    ]

    for case in cases:
        print(f"\n{SEP}")
        print(case["label"])
        print(SEP)
        reply = agent.run(case["context"])
        # 출력 길이 제한 (처음 800자)
        preview = reply[:800] + (" ..." if len(reply) > 800 else "")
        print(preview)


@mock_aws
def test_session_memory() -> None:
    """멀티턴 세션 — 공격자가 만든 리소스를 기억하는지 확인."""
    print(f"\n{'='*60}")
    print("[ 세션 기억 테스트 — 같은 Agent 인스턴스로 2번 호출 ]")
    print(f"{'='*60}")

    from moto.core.llm_agents.agent import HoneypotAgent

    agent = HoneypotAgent()

    # 1st: 버킷 생성
    print(f"\n{SEP}")
    print("Turn 1 — CreateBucket: attacker-exfil-data")
    print(SEP)
    reply1 = agent.run({
        "service": "s3",
        "action": "CreateBucket",
        "method": "PUT",
        "url": "https://attacker-exfil-data.s3.amazonaws.com/",
        "headers": {},
        "body": "",
        "reason": "NotImplementedError",
        "source": "manual_test",
    })
    print(reply1[:400])

    # 2nd: 버킷 목록 — 방금 만든 버킷이 보이는지
    print(f"\n{SEP}")
    print("Turn 2 — ListBuckets  (방금 만든 버킷이 보여야 함)")
    print(SEP)
    reply2 = agent.run({
        "service": "s3",
        "action": "ListBuckets",
        "method": "GET",
        "url": "https://s3.amazonaws.com/",
        "headers": {},
        "body": "",
        "reason": "NotImplementedError",
        "source": "manual_test",
    })
    print(reply2[:600])

    if "attacker-exfil-data" in reply2:
        print("\n✅ 세션 기억 성공 — 이전 턴에서 만든 버킷이 유지됨")
    else:
        print("\n⚠️  버킷 이름이 응답에 없음 — 세션 기억 불완전")


@mock_aws
def test_fallback_integration() -> None:
    """
    실제 boto3 호출 → moto fallback → Agent 전체 경로를 검증한다.

    moto가 NotImplementedError를 던지는 API를 boto3로 호출해서
    botocore_stubber → responses.py → HoneypotAgent 체인이 동작하는지 확인한다.
    """
    print(f"\n{'='*60}")
    print("[ 통합 테스트 — boto3 → moto fallback → Agent ]")
    print(f"{'='*60}")

    ec2 = boto3.client("ec2", region_name="us-east-1")

    # ① ModifyInstanceAttribute 중 moto가 NotImplementedError를 던지는 케이스
    # (예: userData 속성 변경 — moto 미구현)
    print(f"\n{SEP}")
    print("Case 1: ec2.modify_instance_attribute (moto 미구현 속성)")
    print(SEP)
    try:
        # 먼저 인스턴스 하나 생성 (moto가 처리)
        run_resp = ec2.run_instances(
            ImageId="ami-12345678",
            MinCount=1,
            MaxCount=1,
            InstanceType="t2.micro",
        )
        instance_id = run_resp["Instances"][0]["InstanceId"]
        print(f"  생성된 인스턴스: {instance_id}")

        # userData 변경 — moto에서 NotImplementedError → Agent fallback 발생
        import base64
        resp = ec2.modify_instance_attribute(
            InstanceId=instance_id,
            UserData={"Value": base64.b64encode(b"#!/bin/bash\necho hello").decode()},
        )
        print(f"  응답 HTTP 상태: {resp['ResponseMetadata']['HTTPStatusCode']}")
        print("  ✅ fallback 경로 통과 (Agent가 응답 처리)")
    except Exception as e:
        print(f"  에러: {type(e).__name__}: {e}")

    # ② 아예 moto에 없는 서비스 URL → botocore_stubber의 no_backend_match 경로
    print(f"\n{SEP}")
    print("Case 2: boto3 직접 HTTP 호출 — no_backend_match 경로")
    print(SEP)
    import botocore.awsrequest
    import botocore.httpsession
    try:
        # 존재하지 않는 서비스 엔드포인트 — botocore_stubber가 agent로 위임
        import urllib.request
        # requests 대신 boto3 세션을 통해 직접 호출
        session = boto3.session.Session()
        client = session.client("ec2", region_name="us-east-1")
        # DescribeSpotFleetInstances — moto에서 구현되지 않은 경우가 많음
        try:
            resp2 = client.describe_spot_fleet_instances(SpotFleetRequestId="sfr-fake-id")
            print(f"  응답: {str(resp2)[:200]}")
            print("  ✅ Agent가 응답 생성 (moto 미구현 API)")
        except Exception as inner:
            print(f"  boto3 에러 (Agent 응답이 파싱 불가일 수 있음): {type(inner).__name__}: {inner}")
    except Exception as e:
        print(f"  에러: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_moto_handled()
    test_agent_handled()
    test_session_memory()
    test_fallback_integration()
