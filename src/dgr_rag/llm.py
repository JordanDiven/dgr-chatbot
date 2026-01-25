from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def call_chat(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 700,
    timeout_seconds: float = 600.0,
) -> str:
    provider = (provider or "").strip().lower()
    if provider == "openai":
        return _call_openai(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            num_predict=num_predict,
            timeout_seconds=timeout_seconds,
        )
    if provider == "ollama":
        return _call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            num_predict=num_predict,
            timeout_seconds=timeout_seconds,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def _call_ollama(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    timeout_seconds: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DEFAULT_OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama request failed. Is the server running on localhost:11434?") from exc

    obj = json.loads(body)
    message = obj.get("message", {})
    return str(message.get("content", "")).strip()


def _call_openai(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    timeout_seconds: float,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    payload: Dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_CHAT_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("OpenAI request failed. Check your network and API key.") from exc

    obj = json.loads(body)
    choices = obj.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()
