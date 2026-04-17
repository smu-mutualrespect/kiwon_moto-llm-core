"""
LLM Provider 추상화 레이어.

새 LLM 추가 방법 (providers.py 만 수정하면 된다):
  1. LLMProvider 를 상속하는 클래스를 작성한다.
  2. 클래스 맨 위에 provider_name = "이름" 을 선언한다.
  3. complete() 를 구현한다.
  → MOTO_LLM_PROVIDER=이름 으로 즉시 사용 가능. agent.py 수정 불필요.

예시:
  class MyLLMProvider(LLMProvider):
      provider_name = "myllm"
      def complete(self, *, system, messages, timeout=30.0): ...

각 Provider 는 LLMProvider 를 구현한다:
  complete(*, system, messages, timeout) -> str

하위 호환 함수:
  call_claude_api(prompt) / call_gpt_api(prompt) 은 그대로 유지한다.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests as _requests

# ── Provider Protocol (인터페이스) ──────────────────────────────────────────


class LLMProvider:
    """
    모든 LLM 제공자가 상속해야 하는 기본 클래스.

    새 Provider 추가 규칙:
      - provider_name 클래스 변수를 선언한다. (MOTO_LLM_PROVIDER 값과 일치)
      - complete() 를 구현한다.
      - 이 파일에 클래스를 추가하기만 하면 자동으로 등록된다.
    """

    provider_name: str = ""  # 서브클래스에서 반드시 선언

    def __init__(self) -> None:
        # complete() 호출 후 토큰 사용량을 저장한다.
        # {"input_tokens": int, "output_tokens": int}
        self.last_usage: dict[str, int] = {}

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        timeout: float = 30.0,
    ) -> str:
        """
        system   : 시스템 프롬프트 (모델에게 역할·규칙을 지시)
        messages : [{"role": "user"|"assistant", "content": str}, ...]
                   멀티턴 대화 이력을 포함한다.
        반환값   : 모델이 생성한 텍스트
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.complete() 를 구현해야 합니다."
        )


# ── Claude (Anthropic) ──────────────────────────────────────────────────────


class ClaudeProvider(LLMProvider):
    """Anthropic Claude Messages API 제공자."""

    provider_name = "claude"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self._model = model or os.getenv(
            "MOTO_LLM_ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        timeout: float = 30.0,
    ) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 2000,
            "system": system,
            "messages": messages,
        }

        response = _post_json(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
            timeout=timeout,
        )

        usage = response.get("usage", {})
        self.last_usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        parts = [
            item["text"]
            for item in response.get("content", [])
            if item.get("type") == "text" and item.get("text")
        ]
        return "\n".join(parts).strip()


# ── GPT (OpenAI) ──────────────────────────────────────────────────────────


class GPTProvider(LLMProvider):
    """OpenAI Chat Completions API 제공자."""

    provider_name = "gpt"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self._model = model or os.getenv("MOTO_LLM_OPENAI_MODEL", "gpt-4o")

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        timeout: float = 30.0,
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        # OpenAI Chat Completions: system 메시지를 맨 앞에 추가한다.
        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": system}
        ] + list(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
        }

        response = _post_json(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout=timeout,
        )

        usage = response.get("usage", {})
        self.last_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""


# ── Gemini (Google) ──────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """
    Google Gemini API 제공자.

    환경변수:
      GEMINI_API_KEY        : Google AI Studio 에서 발급한 키
      MOTO_LLM_GEMINI_MODEL : 사용할 모델 (기본: gemini-1.5-pro)
    """

    provider_name = "gemini"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self._model = model or os.getenv("MOTO_LLM_GEMINI_MODEL", "gemini-1.5-pro")

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        timeout: float = 30.0,
    ) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

        # Gemini API: user/assistant → user/model 으로 role 매핑
        gemini_contents = [
            {
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}],
            }
            for msg in messages
        ]

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": gemini_contents,
            "generationConfig": {"maxOutputTokens": 2000},
        }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{self._model}:generateContent?key={api_key}"
        )

        response = _post_json(
            url=url,
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout=timeout,
        )

        usage = response.get("usageMetadata", {})
        self.last_usage = {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
        }

        candidates = response.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if "text" in p]
            return "\n".join(texts).strip()
        return ""


# ── 하위 호환 함수 ─────────────────────────────────────────────────────────
# 기존 코드(botocore_stubber, custom_responses_mock, responses)가
# call_claude_api / call_gpt_api 를 직접 import 하므로 그대로 유지한다.
# 새 코드는 Provider 클래스를 직접 사용할 것.


def call_claude_api(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    """단일 턴 Claude 호출 (하위 호환용). 새 코드에서는 ClaudeProvider 를 사용하세요."""
    return ClaudeProvider(model=model).complete(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )


def call_gpt_api(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    """단일 턴 GPT 호출 (하위 호환용). 새 코드에서는 GPTProvider 를 사용하세요."""
    return GPTProvider(model=model).complete(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )


# ── 공통 HTTP 헬퍼 ─────────────────────────────────────────────────────────


def _post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """JSON POST 요청을 보내고 파싱된 JSON 객체를 반환하는 공통 헬퍼."""
    response = _requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError("응답이 JSON 객체가 아닙니다.")
    return parsed
