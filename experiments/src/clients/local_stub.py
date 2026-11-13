"""Deterministic local stub for dry-run evaluation (no network)."""

from __future__ import annotations

import hashlib
import re

from experiments.src.clients.base import ModelResponse
from experiments.src.prompts import ANSWER_TAG, PARTIAL_TAG


class LocalStubClient:
    """Returns a structured but intentionally imperfect answer for pipeline tests."""

    def __init__(self, model: str = "local-stub") -> None:
        self.name = model
        self.provider = "local_stub"
        self.model = model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> ModelResponse:
        del max_tokens  # unused in stub
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        # Extract addend-looking decimals from the expression line.
        addends = _extract_addends(user)
        wants_partials = PARTIAL_TAG.lower() in user.lower() or "step by step" in user.lower()
        digest = hashlib.sha256(
            f"{self.model}|{temperature}|{seed}|{user}".encode()
        ).hexdigest()
        # With high probability emit a wrong final to exercise taxonomy; sometimes echo first addend.
        mode = int(digest[:2], 16) % 5
        if not addends:
            text = f"{ANSWER_TAG}: 0"
        elif mode == 0:
            # Pretend to be correct-ish by summing with a buggy float (still a stub).
            text = _stub_sum_text(addends, wants_partials, mutate="none")
        elif mode == 1:
            text = _stub_sum_text(addends, wants_partials, mutate="shift")
        elif mode == 2:
            text = _stub_sum_text(addends, wants_partials, mutate="trunc")
        elif mode == 3:
            text = "I cannot compute this."
        else:
            text = _stub_sum_text(addends, wants_partials, mutate="carry")
        return ModelResponse(text=text, model=self.model, provider=self.provider, raw={"stub": True})


def _extract_addends(user: str) -> list[str]:
    # Prefer the first line that looks like an addition expression.
    for line in user.splitlines():
        if "+" in line and re.search(r"\d", line):
            parts = [p.strip() for p in line.split("+")]
            cleaned: list[str] = []
            for p in parts:
                m = re.search(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)", p)
                if m:
                    cleaned.append(m.group(0))
            if cleaned:
                return cleaned
    return []


def _stub_sum_text(addends: list[str], wants_partials: bool, mutate: str) -> str:
    from decimal import Decimal

    from experiments.src.decimal_utils import normalize_decimal_str

    lines: list[str] = []
    running = Decimal(0)
    for i, a in enumerate(addends, start=1):
        running += Decimal(a)
        if wants_partials:
            lines.append(f"{PARTIAL_TAG} {i}: {normalize_decimal_str(running)}")
    final = running
    if mutate == "shift":
        final = running * Decimal(10)
    elif mutate == "trunc":
        final = running.quantize(Decimal("1"))
    elif mutate == "carry":
        final = running + Decimal(1)
    lines.append(f"{ANSWER_TAG}: {normalize_decimal_str(final)}")
    return "\n".join(lines)
