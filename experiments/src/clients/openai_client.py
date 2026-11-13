"""OpenAI-compatible Chat Completions client (tools / function-calling disabled)."""

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
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        default_base_url: str = "https://api.openai.com/v1",
        provider: str = "openai",
        timeout: float = 120.0,
        disable_tools: bool = True,
    ) -> None:
        self.name = model
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.api_key_env = api_key_env
        env_base = os.environ.get(base_url_env)
        self.base_url = (base_url or env_base or default_base_url).rstrip("/")
        self.timeout = timeout
        self.disable_tools = disable_tools

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Disable tools without sending an empty tools=[] (some providers 400 on that).
        if self.disable_tools:
            payload["tool_choice"] = "none"
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
        message = data["choices"][0]["message"]
        text = message.get("content") or ""
        # If a provider still returns tool_calls, refuse to treat them as answers.
        if self.disable_tools and message.get("tool_calls"):
            raise RuntimeError(
                f"Provider {self.provider} returned tool_calls despite tool_choice=none; "
                "aborting so arithmetic is not delegated to tools."
            )
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw=data)
