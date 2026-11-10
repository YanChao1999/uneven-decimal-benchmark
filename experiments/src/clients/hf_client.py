"""Hugging Face OpenAI-compatible Inference client (e.g. Qwen). Tools disabled."""

from __future__ import annotations

import os
from typing import Any

import httpx

from experiments.src.clients.base import ModelResponse


class HuggingFaceClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.name = model
        self.provider = "hf"
        self.model = model
        self.api_key = api_key or os.environ.get("HF_TOKEN", "")
        self.base_url = (
            base_url or os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1")
        ).rstrip("/")
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
            raise RuntimeError("HF_TOKEN is not set")
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
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw=data)
