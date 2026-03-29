# core/llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import requests


@dataclass
class LLMSettings:
    enabled: bool
    provider: str  # openai / groq / openrouter / ollama / custom
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 500


# Provider presets (OpenAI-compatible)
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


def build_base_url(provider: str, custom_url: Optional[str] = None) -> str:
    if provider == "custom" and custom_url:
        return custom_url.rstrip("/")
    base = PROVIDER_BASE_URLS.get(provider, "https://api.openai.com/v1")
    return base.rstrip("/")


class LLMError(RuntimeError):
    pass


def _clamp(x: int, lo: int, hi: int) -> int:
    try:
        x = int(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))


def _make_headers(settings: LLMSettings) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}

    # Ollama doesn't need auth
    if settings.provider != "ollama":
        headers["Authorization"] = f"Bearer {settings.api_key}"

    # OpenRouter usually wants these (not mandatory but recommended)
    if settings.provider == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:8501"
        headers["X-Title"] = "IKNI RAG MVP"

    return headers


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60) -> Tuple[int, str, Dict[str, Any]]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    text = r.text or ""
    try:
        js = r.json()
    except Exception:
        js = {}
    return r.status_code, text, js


def chat_completion(settings: LLMSettings, messages: List[Dict[str, str]]) -> str:
    """
    Minimal OpenAI-compatible Chat Completions client.
    Works with OpenAI, Groq, OpenRouter, Ollama (v1 compatible), or custom endpoints.

    ✅ Improvements:
    - Shows provider error body on failure (VERY important for debugging)
    - Clamps max_tokens
    - Groq fallback: try max_completion_tokens if max_tokens rejected
    """
    if not settings.enabled:
        raise LLMError("LLM is disabled in settings.")
    if not settings.model.strip():
        raise LLMError("LLM model is empty.")
    if settings.provider != "ollama" and not settings.api_key.strip():
        raise LLMError("API key is missing.")

    base_url = build_base_url(settings.provider, settings.base_url)
    url = f"{base_url}/chat/completions"

    headers = _make_headers(settings)

    # Keep safe bounds (many providers reject huge values)
    max_tokens = _clamp(settings.max_tokens, 16, 2048)

    payload: Dict[str, Any] = {
        "model": settings.model.strip(),
        "messages": messages,
        "temperature": float(settings.temperature),
        "max_tokens": max_tokens,
    }

    status, text, js = _post_json(url, headers, payload, timeout=60)

    # ✅ If success
    if 200 <= status < 300:
        try:
            return js["choices"][0]["message"]["content"]
        except Exception:
            raise LLMError(f"LLM response parse error. Raw JSON: {js}")

    # ✅ GROQ fallback: some deployments prefer max_completion_tokens
    if settings.provider == "groq" and status == 400:
        payload2 = dict(payload)
        payload2.pop("max_tokens", None)
        payload2["max_completion_tokens"] = max_tokens

        status2, text2, js2 = _post_json(url, headers, payload2, timeout=60)

        if 200 <= status2 < 300:
            try:
                return js2["choices"][0]["message"]["content"]
            except Exception:
                raise LLMError(f"LLM response parse error (fallback). Raw JSON: {js2}")

        # If fallback also failed -> show fallback error
        raise LLMError(
            f"[{settings.provider}] HTTP {status2} Bad Request.\n"
            f"URL: {url}\n"
            f"Model: {settings.model}\n"
            f"Response: {text2}\n"
        )

    # Default error message with body
    raise LLMError(
        f"[{settings.provider}] HTTP {status} Error.\n"
        f"URL: {url}\n"
        f"Model: {settings.model}\n"
        f"Response: {text}\n"
    )
