from __future__ import annotations

import json


def build_llm_fallback_json(
    message: str = "llm_fallback!!",
) -> tuple[dict[str, str], str]:
    # fallback 표식을 JSON 응답 body와 헤더로 만들어 돌려준다.

    headers = {"Content-Type": "application/json"}
    # 응답 body가 JSON이라는 것을 명시한다.

    body = json.dumps({"message": message})
    # 사람이 보기 쉬운 단일 message 필드 JSON을 만든다.

    return headers, body
    # 호출부가 바로 HTTP 응답에 쓸 수 있도록 헤더와 body를 함께 반환한다.
