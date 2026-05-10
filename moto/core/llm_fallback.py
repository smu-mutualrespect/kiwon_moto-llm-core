from __future__ import annotations

# 미래형 타입 힌트를 문자열 평가 없이 사용할 수 있게 한다.
import json

# fallback 확인용 JSON body를 만들 때 사용한다.


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
