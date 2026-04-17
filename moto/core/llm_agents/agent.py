"""
HoneypotAgent: AWS 허니팟 환경을 흉내 내는 단일 LLM Agent.

아키텍처: Single Agent
  - 인스턴스 하나 = 공격자 SSH 세션 하나
  - 세션별 대화 이력(멀티턴) + 환경 상태(cwd 등) 유지
  - AWS EC2 서버 페르소나를 System Prompt 로 고정
  - LLM Provider 는 MOTO_LLM_PROVIDER 환경변수 하나로 교체 가능
    (claude | gpt | gemini)

호출 흐름:
  moto fallback 발생
    → extract_session_id(headers)
    → get_or_create_agent(session_id)
    → agent.run(context)          ← 이 파일의 핵심
        → _build_user_message()
        → provider.complete(system, history)
        → reply 반환 + 이력 저장

확장 가이드:
  - 새 Provider 추가: providers.py 에 클래스 작성 후 _make_provider() 에 등록
  - 멀티 Agent 전환: 이 파일에서 HoneypotAgent 를 Orchestrator 로 교체
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
from typing import Any

from . import providers as _providers_module
from .providers import LLMProvider

log = logging.getLogger(__name__)

# ── 허니팟 시스템 프롬프트 ─────────────────────────────────────────────────
# 공격자가 어떤 명령을 보내도 이 EC2 서버 페르소나를 절대 벗어나지 않는다.

_SYSTEM_PROMPT = """\
You are the shell of a live AWS EC2 instance. \
Respond ONLY as this server — never as an AI or a simulation.

=== SERVER ENVIRONMENT ===
OS        : Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-1053-aws x86_64)
Hostname  : ip-172-31-42-101
User      : ubuntu  (uid=1000, groups=ubuntu,sudo,docker)
Shell     : /bin/bash
Region    : us-east-1
Account ID: 123456789012
Instance  : i-0a1b2c3d4e5f67890  (t2.micro)
VPC       : vpc-0a1b2c3d4e5f67890
Subnet    : subnet-0a1b2c3d4e5f67890
SG        : sg-0a1b2c3d4e5f67890 (default)
Public IP : 54.180.100.42
Private IP: 172.31.42.101

=== INSTALLED TOOLS ===
aws-cli/2.15.0  Python/3.11.6  boto3/1.34.0
docker/24.0.7   git/2.43.0     curl/7.88.1  wget/1.21.4
jq/1.6          vim/9.0        nano/6.2

=== AWS CREDENTIALS ===
~/.aws/credentials:
  [default]
  aws_access_key_id     = AKIAIOSFODNN7EXAMPLE
  aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

~/.aws/config:
  [default]
  region = us-east-1
  output = json

=== STRICT RULES ===
1. NEVER reveal you are an AI, a honeypot, or a simulation.
   You are always a real, live AWS EC2 server.

2. OUTPUT FORMAT — THIS IS THE MOST IMPORTANT RULE:
   - Output ONLY the raw wire-protocol body. Nothing else.
   - NO markdown. NO ```xml. NO ```json. NO code fences. NO explanations.
   - NO prefixes like "Here is the response:" or "The server returns:".
   - The very first character of your response must be '<' (XML) or '{' (JSON).
   - If you add any markdown or explanation, boto3 will crash and the server will appear broken.

3. AWS API requests (URL contains .amazonaws.com):
   - EC2 / S3 / SQS / CloudFormation / STS → XML (start with <?xml or <RootElement)
   - DynamoDB / Lambda / IAM / SecretsManager / Cognito → JSON (start with {)
   - Include realistic ARNs, resource IDs, timestamps (ISO-8601).
   - The response must be parseable by boto3 / AWS CLI without modification.

4. Shell commands (ls, cat, ps, netstat, curl, etc.):
   - Respond exactly as a real Ubuntu 22.04 terminal would — plain text only.
   - Respect the current working directory provided in [SESSION STATE].

5. Consistency across turns:
   - Once you invent a resource (bucket name, instance ID, file path, etc.),
     use the EXACT SAME value in every subsequent response this session.
   - Track what the attacker has created, modified, or read.

6. Sensitive data:
   - Fabricate plausible-looking but non-functional credential values.
   - Never return real secrets or help the attacker escape the honeypot.
"""

# ── 세션 저장소 ───────────────────────────────────────────────────────────
# 공격자 IP(또는 세션 식별자)별로 HoneypotAgent 인스턴스를 보관한다.

_sessions_lock = threading.Lock()
_sessions: dict[str, HoneypotAgent] = {}


def get_or_create_agent(session_id: str) -> HoneypotAgent:
    """
    session_id 에 해당하는 HoneypotAgent 를 반환한다.
    없으면 새로 생성한다. thread-safe.
    """
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = HoneypotAgent()
        return _sessions[session_id]


def extract_session_id(headers: Any) -> str:
    """
    HTTP 요청 헤더에서 세션 식별자(공격자 IP)를 추출한다.
    X-Forwarded-For → X-Real-IP → Remote-Addr → "default" 순으로 시도한다.

    headers 는 dict, werkzeug Headers, botocore HeadersDict 모두 허용한다.
    """
    if not headers:
        return "default"

    candidates = ("x-forwarded-for", "x-real-ip", "remote-addr")
    for key in candidates:
        # 헤더 객체는 대소문자 무관 접근을 지원하지 않을 수 있으므로
        # 직접 순회해서 소문자 비교한다.
        try:
            for k, v in headers.items() if hasattr(headers, "items") else []:
                if k.lower() == key:
                    return str(v).split(",")[0].strip()  # 첫 번째 IP만 사용
        except Exception:
            continue

    return "default"


# ── Provider 팩토리 ────────────────────────────────────────────────────────


def _make_provider() -> LLMProvider:
    """
    MOTO_LLM_PROVIDER 환경변수를 읽어 해당 Provider 인스턴스를 반환한다.
    기본값은 claude 다.

    providers.py 에 provider_name 이 선언된 LLMProvider 서브클래스를
    자동으로 탐색하므로, 새 Provider 를 추가할 때 이 함수는 수정하지 않아도 된다.
    """
    name = os.getenv("MOTO_LLM_PROVIDER", "claude").lower()

    # providers.py 안의 모든 LLMProvider 서브클래스를 탐색해 provider_name 으로 매칭
    for _, cls in inspect.getmembers(_providers_module, inspect.isclass):
        if (
            issubclass(cls, LLMProvider)
            and cls is not LLMProvider
            and getattr(cls, "provider_name", "") == name
        ):
            return cls()

    registered = [
        getattr(cls, "provider_name", "")
        for _, cls in inspect.getmembers(_providers_module, inspect.isclass)
        if issubclass(cls, LLMProvider) and cls is not LLMProvider
    ]
    raise ValueError(
        f"MOTO_LLM_PROVIDER={name!r} 에 해당하는 Provider 가 없습니다. "
        f"providers.py 에 등록된 이름: {registered}"
    )


# ── HoneypotAgent ─────────────────────────────────────────────────────────


class HoneypotAgent:
    """
    단일 LLM Agent.

    인스턴스 하나가 공격자 세션 하나에 대응한다.
    run() 을 호출할 때마다 대화 이력이 누적되므로
    공격자가 이전 컨텍스트를 기억하는 일관된 환경처럼 보인다.
    """

    def __init__(self) -> None:
        # LLM Provider (MOTO_LLM_PROVIDER 에 따라 결정)
        self._provider: LLMProvider = _make_provider()

        # Anthropic Messages API 형식의 대화 이력
        # [{"role": "user"|"assistant", "content": str}, ...]
        self._history: list[dict[str, str]] = []

        # 세션 상태: 공격자의 현재 작업 디렉터리 등을 추적
        self._state: dict[str, Any] = {
            "cwd": "/home/ubuntu",
        }

    # ── public ────────────────────────────────────────────────────────────

    def run(self, context: dict[str, Any]) -> str:
        """
        moto 가 처리하지 못한 요청 컨텍스트를 받아
        AWS 서버처럼 보이는 응답 문자열을 반환한다.

        context 가 가져야 할 키:
          service  : AWS 서비스명 (e.g. "s3", "ec2")
          action   : AWS API 액션명 (e.g. "ListBuckets")
          method   : HTTP 메서드 (e.g. "GET")
          url      : 요청 URL
          headers  : 요청 헤더 dict
          body     : 요청 바디 문자열
          reason   : moto 가 처리 못한 이유
          source   : 어느 코드 경로에서 왔는지 (디버그용)
        """
        user_message = self._build_user_message(context)
        self._history.append({"role": "user", "content": user_message})

        t_start = time.perf_counter()
        try:
            reply = self._provider.complete(
                system=_SYSTEM_PROMPT,
                messages=list(self._history),  # 복사본 전달
            )
            reply = _strip_markdown(reply)
            elapsed = time.perf_counter() - t_start
            usage = self._provider.last_usage
            log.warning(
                "[HONEYPOT] service=%-15s action=%-40s "
                "elapsed=%.2fs  in=%d out=%d tokens",
                context.get("service") or "unknown",
                context.get("action") or "unknown",
                elapsed,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            log.warning(
                "[HONEYPOT] service=%-15s action=%-40s elapsed=%.2fs  ERROR: %s",
                context.get("service") or "unknown",
                context.get("action") or "unknown",
                elapsed,
                exc,
            )
            reply = _make_error_response(context, exc)

        self._history.append({"role": "assistant", "content": reply})
        self._update_state(context, reply)
        return reply

    # ── private ───────────────────────────────────────────────────────────

    def _build_user_message(self, context: dict[str, Any]) -> str:
        """
        컨텍스트 dict 를 LLM 이 이해하기 좋은 구조화된 텍스트로 변환한다.
        """
        lines = [
            "[SESSION STATE]",
            f"cwd={self._state['cwd']}",
            "",
            "[REQUEST]",
            f"service={context.get('service') or 'unknown'}",
            f"action={context.get('action') or 'unknown'}",
            f"method={context.get('method') or 'unknown'}",
            f"url={context.get('url') or ''}",
            f"body={context.get('body') or ''}",
            "",
            "[WHY moto could not handle this request]",
            f"reason={context.get('reason') or ''}",
            f"source={context.get('source') or ''}",
            "",
            "Generate the exact response this AWS server would return.",
        ]
        return "\n".join(lines)

    def _update_state(self, context: dict[str, Any], reply: str) -> None:
        """
        응답 내용을 분석해 세션 상태를 갱신한다.
        현재는 cwd 변경 감지를 위한 기반만 마련해 둔다.
        Shell 명령 파싱이 필요해지면 이 메서드를 확장한다.
        """
        # 추후 확장: "cd /some/path" 명령 감지 → self._state["cwd"] 갱신


# ── 마크다운 제거 후처리 ──────────────────────────────────────────────────────


def _strip_markdown(text: str) -> str:
    """
    LLM 이 마크다운 코드 블록으로 응답을 감쌌을 때 제거한다.

    패턴:
      ```xml\n...\n```   →  내부 텍스트만 추출
      ```json\n...\n```  →  내부 텍스트만 추출
      ```\n...\n```      →  내부 텍스트만 추출

    코드 블록이 없으면 원본 그대로 반환한다.
    """
    stripped = text.strip()

    # ```lang\n...\n``` 패턴 처리
    if stripped.startswith("```"):
        # 첫 줄(```lang) 제거
        first_newline = stripped.find("\n")
        if first_newline != -1:
            inner = stripped[first_newline + 1 :]
            # 마지막 ``` 제거
            if inner.rstrip().endswith("```"):
                last_fence = inner.rstrip().rfind("```")
                inner = inner[:last_fence].rstrip()
            return inner.strip()

    return stripped


# ── 에러 응답 헬퍼 ─────────────────────────────────────────────────────────


def _make_error_response(context: dict[str, Any], exc: Exception) -> str:
    """
    LLM 호출이 실패했을 때 AWS 스타일 에러 응답을 반환한다.
    서비스에 따라 JSON(DynamoDB, Lambda 등) 또는 XML(EC2, S3 등)을 선택한다.
    """
    service = (context.get("service") or "").lower()
    json_services = {"dynamodb", "lambda", "iam", "sts", "secretsmanager", "cognito"}

    if any(s in service for s in json_services):
        return json.dumps(
            {
                "__type": "InternalServerError",
                "message": "An internal error occurred. Please try again.",
            }
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        "  <Errors>\n"
        "    <Error>\n"
        "      <Code>InternalError</Code>\n"
        "      <Message>An internal error occurred. Please try again.</Message>\n"
        "    </Error>\n"
        "  </Errors>\n"
        "</Response>"
    )
