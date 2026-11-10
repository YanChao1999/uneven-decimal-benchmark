"""Answer parsing, exact comparison, and heuristic error taxonomy."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from experiments.src.decimal_utils import frac_len, normalize_decimal_str, parse_decimal
from experiments.src.prompts import ANSWER_TAG, PARTIAL_TAG

ErrorType = Literal[
    "correct",
    "wrong_final",
    "misaligned_place",
    "carry_error",
    "dropped_digit",
    "truncated_frac",
    "format_parse_fail",
    "partial_sum_inconsistent",
]

_ANSWER_RE = re.compile(
    rf"(?im)^\s*{re.escape(ANSWER_TAG)}\s*:\s*([+-]?(?:\d+\.\d*|\d*\.\d+|\d+))\s*$"
)
_PARTIAL_RE = re.compile(
    rf"(?im)^\s*{re.escape(PARTIAL_TAG)}\s*(\d+)\s*:\s*([+-]?(?:\d+\.\d*|\d*\.\d+|\d+))\s*$"
)
_LOOSE_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)")


@dataclass
class EvalResult:
    example_id: str
    stratum: str
    condition: str
    model: str
    temperature: float
    sample_index: int
    predicted: str | None
    expected: str
    correct: bool
    error_type: ErrorType
    cluster_id: str = ""
    alignment_level: int = 0
    carry_level: int = 0
    parsed_partials: list[str] = field(default_factory=list)
    expected_partials: list[str] = field(default_factory=list)
    raw_text: str = ""
    manual_override: ErrorType | None = None
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def extract_final_answer(text: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return normalize_decimal_str(matches[-1].group(1))
    # Fallback: last decimal-looking token on the last non-empty line.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    loose = _LOOSE_NUMBER_RE.findall(lines[-1])
    if not loose:
        return None
    try:
        return normalize_decimal_str(loose[-1])
    except (InvalidOperation, ValueError):
        return None


def extract_partials(text: str) -> list[str]:
    found: dict[int, str] = {}
    for m in _PARTIAL_RE.finditer(text):
        idx = int(m.group(1))
        found[idx] = normalize_decimal_str(m.group(2))
    if not found:
        return []
    max_k = max(found)
    return [found.get(k, "") for k in range(1, max_k + 1)]


def _scale_mismatch(pred: str, expected: str) -> bool:
    """True if values differ by a power-of-ten place shift (rough misalignment signal)."""
    try:
        p = parse_decimal(pred)
        e = parse_decimal(expected)
    except ValueError:
        return False
    if p == 0 or e == 0:
        return False
    ratio = abs(p / e)
    # powers of ten within a reasonable range
    for k in range(1, 12):
        ten = Decimal(10) ** k
        if ratio == ten or ratio == (Decimal(1) / ten):
            return True
    return False


def _likely_carry_error(pred: str, expected: str) -> bool:
    try:
        p = abs(parse_decimal(pred))
        e = abs(parse_decimal(expected))
    except ValueError:
        return False
    diff = abs(p - e)
    if diff == 0:
        return False
    # Classic off-by-one-carry footprints: multiples of powers of ten, often small.
    for k in range(0, 10):
        unit = Decimal(10) ** k
        if diff == unit or diff == 9 * unit:
            return True
    return False


def classify_error(
    *,
    predicted: str | None,
    expected: str,
    parsed_partials: list[str],
    expected_partials: list[str],
    expect_partials: bool,
) -> ErrorType:
    if predicted is None:
        return "format_parse_fail"
    try:
        if normalize_decimal_str(predicted) == normalize_decimal_str(expected):
            if expect_partials and expected_partials:
                if not parsed_partials or any(
                    (i >= len(parsed_partials) or parsed_partials[i] != expected_partials[i])
                    for i in range(len(expected_partials))
                ):
                    return "partial_sum_inconsistent"
            return "correct"
    except (InvalidOperation, ValueError):
        return "format_parse_fail"

    if expect_partials and expected_partials and parsed_partials:
        if any(
            (i >= len(parsed_partials) or not parsed_partials[i]
             or parsed_partials[i] != expected_partials[i])
            for i in range(len(expected_partials))
        ):
            # Still wrong final; mark partial inconsistency preferentially when present.
            if normalize_decimal_str(predicted) != normalize_decimal_str(expected):
                return "partial_sum_inconsistent"

    if _scale_mismatch(predicted, expected):
        return "misaligned_place"

    if frac_len(predicted) < frac_len(expected) and normalize_decimal_str(
        predicted
    ) != normalize_decimal_str(expected):
        # Truncation heuristic: predicted matches expected truncated to fewer frac digits.
        try:
            e = parse_decimal(expected)
            trunc = normalize_decimal_str(
                e.quantize(Decimal(1).scaleb(-frac_len(predicted)))
                if frac_len(predicted) > 0
                else e.to_integral_value()
            )
            # Simpler string truncation check on fractional part.
            if "." in expected and frac_len(predicted) >= 0:
                whole, frac = expected.lstrip("-").split(".", 1)
                sign = "-" if expected.startswith("-") else ""
                cut = sign + whole + (("." + frac[: frac_len(predicted)]) if frac_len(predicted) else "")
                if normalize_decimal_str(cut) == normalize_decimal_str(predicted):
                    return "truncated_frac"
        except Exception:
            pass

    if len(predicted.replace(".", "").replace("-", "")) < len(
        expected.replace(".", "").replace("-", "")
    ):
        if _likely_carry_error(predicted, expected):
            return "carry_error"
        return "dropped_digit"

    if _likely_carry_error(predicted, expected):
        return "carry_error"

    return "wrong_final"


def evaluate_response(
    *,
    example: dict[str, Any],
    condition: str,
    model: str,
    temperature: float,
    sample_index: int,
    raw_text: str,
    expect_partials: bool,
    manual_override: ErrorType | None = None,
) -> EvalResult:
    predicted = extract_final_answer(raw_text)
    parsed_partials = extract_partials(raw_text) if expect_partials else []
    expected = example["expected_sum"]
    expected_partials = list(example.get("partial_sums", []))
    error_type = classify_error(
        predicted=predicted,
        expected=expected,
        parsed_partials=parsed_partials,
        expected_partials=expected_partials,
        expect_partials=expect_partials,
    )
    if manual_override is not None:
        error_type = manual_override
    correct = error_type == "correct"
    cluster_id = str(example.get("cluster_id") or example.get("stratum") or "")
    return EvalResult(
        example_id=example["id"],
        stratum=str(example.get("stratum", cluster_id)),
        cluster_id=cluster_id,
        alignment_level=int(example.get("alignment_level", 0)),
        carry_level=int(example.get("carry_level", 0)),
        condition=condition,
        model=model,
        temperature=temperature,
        sample_index=sample_index,
        predicted=predicted,
        expected=expected,
        correct=correct,
        error_type=error_type,
        parsed_partials=parsed_partials,
        expected_partials=expected_partials if expect_partials else [],
        raw_text=raw_text,
        manual_override=manual_override,
    )
