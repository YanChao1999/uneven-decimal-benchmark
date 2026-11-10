"""Ollama local chat client.

Uses the native ``/api/chat`` endpoint (works across Ollama versions).
Falls back to OpenAI-compatible ``/v1/chat/completions`` when available.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from experiments.src.clients.base import ModelResponse


class OllamaClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        prefer_native: bool = True,
    ) -> None:
        self.name = model
        self.provider = "ollama"
        self.model = model
        # Accept either http://host:11434 or .../v1 — normalize to host root.
        raw = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        if raw.endswith("/v1"):
            raw = raw[: -len("/v1")]
        self.host = raw
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
        self.timeout = timeout
        self.prefer_native = prefer_native

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ModelResponse:
        errors: list[str] = []
        if self.prefer_native:
            try:
                return self._complete_native(messages, temperature, max_tokens, seed)
            except Exception as exc:  # noqa: BLE001 — fall through to OpenAI path
                errors.append(f"native /api/chat: {exc}")
            try:
                return self._complete_openai(messages, temperature, max_tokens, seed)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"openai /v1/chat/completions: {exc}")
        else:
            try:
                return self._complete_openai(messages, temperature, max_tokens, seed)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"openai /v1/chat/completions: {exc}")
            try:
                return self._complete_native(messages, temperature, max_tokens, seed)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"native /api/chat: {exc}")

        hint = (
            f"Ollama call failed for model {self.model!r} at {self.host}. "
            "Check: (1) `ollama serve` is running; "
            f"(2) `ollama pull {self.model}`; "
            "(3) `curl -s http://127.0.0.1:11434/api/tags`. "
            f"Details: {' | '.join(errors)}"
        )
        raise RuntimeError(hint)

    def _complete_native(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> ModelResponse:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            if resp.status_code == 404:
                # Often means the model tag is missing.
                raise RuntimeError(
                    f"404 from /api/chat — is model {self.model!r} pulled? "
                    f"Try: ollama pull {self.model}"
                )
            resp.raise_for_status()
            data = resp.json()
        message = data.get("message") or {}
        text = message.get("content") or ""
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw=data)

    def _complete_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.host}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw=data)
