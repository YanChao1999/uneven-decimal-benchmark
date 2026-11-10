"""English prompt templates for the four evaluation conditions."""

from __future__ import annotations

from typing import Any, Literal

PromptCondition = Literal["direct", "scratch_partial", "align_hint", "scratch_align"]

ANSWER_TAG = "FINAL_ANSWER"
PARTIAL_TAG = "PARTIAL"

_ALIGNMENT_RULES = (
    "When adding decimals, align numbers by the decimal point (not by the leftmost digit). "
    "Pad shorter fractional parts with zeros on the right if helpful. "
    "Add digit-by-digit from right to left and propagate carries explicitly whenever a column "
    "sum is 10 or greater."
)


def format_addends(addends: list[str]) -> str:
    return " + ".join(addends)


def build_messages(
    condition: PromptCondition,
    addends: list[str],
    *,
    answer_tag: str = ANSWER_TAG,
    partial_tag: str = PARTIAL_TAG,
) -> list[dict[str, str]]:
    expression = format_addends(addends)
    system = (
        "You are a careful arithmetic assistant. "
        "Do not use tools, calculators, or external functions. "
        "Work only from the text of the problem."
    )

    if condition == "direct":
        user = (
            f"Compute the sum of the following decimals:\n{expression}\n\n"
            "Do not show intermediate work. "
            f"Reply with exactly one line of the form `{answer_tag}: <decimal>` "
            "and nothing else."
        )
    elif condition == "scratch_partial":
        user = (
            f"Compute the chain sum of the following decimals:\n{expression}\n\n"
            "You must accumulate step by step. After including each addend, write a line "
            f"`{partial_tag} k: <running_total>` for k = 1..n. "
            f"End with `{answer_tag}: <decimal>` equal to the final running total. "
            "Do not use tools."
        )
    elif condition == "align_hint":
        user = (
            f"Compute the sum of the following decimals:\n{expression}\n\n"
            f"{_ALIGNMENT_RULES}\n\n"
            "Do not show intermediate work beyond what is necessary. "
            f"Reply with exactly one line of the form `{answer_tag}: <decimal>`."
        )
    elif condition == "scratch_align":
        user = (
            f"Compute the chain sum of the following decimals:\n{expression}\n\n"
            f"{_ALIGNMENT_RULES}\n\n"
            "Accumulate step by step. After including each addend, write "
            f"`{partial_tag} k: <running_total>`. "
            f"End with `{answer_tag}: <decimal>`. Do not use tools."
        )
    else:
        raise ValueError(f"Unknown prompt condition: {condition}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_prompt_text(messages: list[dict[str, str]]) -> str:
    parts = [f"[{m['role']}]\n{m['content']}" for m in messages]
    return "\n\n".join(parts)


def condition_expects_partials(condition: PromptCondition) -> bool:
    return condition in {"scratch_partial", "scratch_align"}


def load_prompt_config(path_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return path_config or {
        "conditions": ["direct", "scratch_partial", "align_hint", "scratch_align"],
        "answer_tag": ANSWER_TAG,
        "partial_tag": PARTIAL_TAG,
    }
