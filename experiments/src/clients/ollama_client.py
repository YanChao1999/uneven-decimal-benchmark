"""Ollama local OpenAI-compatible chat client (tools disabled)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from experiments.src.clients.base import ModelResponse


class OllamaClient:
    """Talks to a local Ollama server (default http://127.0.0.1:11434/v1)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.name = model
        self.provider = "ollama"
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        ).rstrip("/")
        # Ollama ignores the key but OpenAI-compatible clients often send one.
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [],
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw=data)
