from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

_DOTENV_LOADED = False


def call_gpt_api(
    prompt: str, *, model: Optional[str] = None, timeout: float = 20.0
) -> str:
    text, _ = call_gpt_api_with_meta(prompt, model=model, timeout=timeout)
    return text


def call_gpt_api_with_meta(
    prompt: str, *, model: Optional[str] = None, timeout: float = 20.0
) -> tuple[str, dict[str, Any]]:
    _load_dotenv_if_present()
    provider = select_provider()
    if provider == "anthropic":
        return _call_anthropic_api_with_meta(prompt, model=model, timeout=timeout)
    if provider != "openai":
        raise ValueError(f"Unsupported MOTO_LLM_PROVIDER: {provider}")
    return _call_openai_api_with_meta(prompt, model=model, timeout=timeout)


def select_provider() -> str:
    explicit = os.getenv("MOTO_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _call_openai_api_with_meta(
    prompt: str, *, model: Optional[str] = None, timeout: float = 20.0
) -> tuple[str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    payload = {
        "model": model or os.getenv("MOTO_LLM_OPENAI_MODEL", "gpt-5-mini"),
        "max_output_tokens": int(os.getenv("MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS", "60")),
        "reasoning": {
            "effort": os.getenv("MOTO_LLM_OPENAI_REASONING_EFFORT", "minimal")
        },
        "input": [{"role": "user", "content": prompt}],
    }
    started = time.perf_counter()
    response = _post_json(
        url="https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=timeout,
    )
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if text:
                    parts.append(text)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return "\n".join(parts).strip(), {
        "provider": "openai",
        "model": payload["model"],
        "usage": response.get("usage"),
        "duration_ms": round(elapsed_ms, 3),
        "response_id": response.get("id"),
    }


def _call_anthropic_api_with_meta(
    prompt: str, *, model: Optional[str] = None, timeout: float = 20.0
) -> tuple[str, dict[str, Any]]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    payload = {
        "model": model
        or os.getenv("MOTO_LLM_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        "max_tokens": int(
            os.getenv(
                "MOTO_LLM_ANTHROPIC_MAX_OUTPUT_TOKENS",
                os.getenv("MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS", "60"),
            )
        ),
        "messages": [{"role": "user", "content": prompt}],
    }
    started = time.perf_counter()
    response = _post_json(
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": os.getenv("MOTO_LLM_ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=timeout,
    )
    parts: list[str] = []
    for content in response.get("content", []):
        if content.get("type") == "text":
            text = content.get("text")
            if text:
                parts.append(text)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    usage = _normalize_anthropic_usage(response.get("usage"))
    return "\n".join(parts).strip(), {
        "provider": "anthropic",
        "model": response.get("model", payload["model"]),
        "usage": usage,
        "duration_ms": round(elapsed_ms, 3),
        "response_id": response.get("id"),
    }


def _normalize_anthropic_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    normalized = dict(usage)
    normalized.setdefault("total_tokens", input_tokens + output_tokens)
    return normalized


def _post_json(
    *, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = Request(
        url=url,
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response")
    return parsed


def _load_dotenv_if_present() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    env_file = os.getenv("MOTO_LLM_ENV_FILE")
    if env_file:
        _load_env_file(Path(env_file))
        _DOTENV_LOADED = True
        return
    cwd = Path.cwd()
    for candidate_dir in [cwd, *cwd.parents]:
        candidate = candidate_dir / ".env"
        if candidate.exists():
            _load_env_file(candidate)
            _DOTENV_LOADED = True
            return
    _DOTENV_LOADED = True


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))
