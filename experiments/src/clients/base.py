"""Unified model client interface. Tools / function calling are always disabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ModelResponse:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] | None = None


class ModelClient(Protocol):
    name: str
    provider: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ModelResponse: ...


def build_client(spec: dict[str, Any]) -> ModelClient:
    provider = spec["provider"]
    model = spec["model"]
    if provider == "openai":
        from experiments.src.clients.openai_client import OpenAIClient

        return OpenAIClient(model=model)
    if provider == "hf":
        from experiments.src.clients.hf_client import HuggingFaceClient

        return HuggingFaceClient(model=model)
    if provider == "ollama":
        from experiments.src.clients.ollama_client import OllamaClient

        return OllamaClient(model=model, base_url=spec.get("base_url"))
    if provider == "local_stub":
        from experiments.src.clients.local_stub import LocalStubClient

        return LocalStubClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")
