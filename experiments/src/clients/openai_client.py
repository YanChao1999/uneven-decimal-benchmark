"""OpenAI-compatible Chat Completions client (tools disabled)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from experiments.src.clients.base import ModelResponse


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.name = model
        self.provider = "openai"
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Explicitly disable tools / function calling.
            "tools": [],
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw=data)
