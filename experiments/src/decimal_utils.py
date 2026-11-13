"""Exact decimal helpers (never use binary float for labels)."""

from __future__ import annotations

import re
from decimal import Decimal, getcontext
from typing import Iterable

getcontext().prec = 200

_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)$")


def normalize_decimal_str(value: str | Decimal) -> str:
    """Normalize a decimal to a canonical string without scientific notation."""
    d = Decimal(str(value))
    # Normalize removes trailing zeros in the exponent form; force fixed point.
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in {"", "-", "+"}:
        return "0"
    if s.startswith("+"):
        s = s[1:]
    return s


def parse_decimal(value: str) -> Decimal:
    text = value.strip().replace(",", "")
    if not _DECIMAL_RE.match(text):
        raise ValueError(f"Invalid decimal string: {value!r}")
    return Decimal(text)


def frac_len(s: str) -> int:
    s = s.lstrip("+-")
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])


def int_len(s: str) -> int:
    s = s.lstrip("+-")
    head = s.split(".", 1)[0]
    return max(len(head), 1)


def pad_frac(s: str, width: int) -> str:
    """Right-pad fractional digits with zeros to ``width`` (sign preserved)."""
    sign = ""
    body = s
    if body.startswith(("+", "-")):
        sign, body = body[0], body[1:]
    if "." not in body:
        body = body + "."
    whole, frac = body.split(".", 1)
    if len(frac) > width:
        raise ValueError("fraction longer than pad width")
    return f"{sign}{whole}.{frac.ljust(width, '0')}"


def column_carry_count(addends: Iterable[str]) -> int:
    """Count column-wise carries after aligning by decimal point (right-pad)."""
    items = [normalize_decimal_str(a) for a in addends]
    max_frac = max((frac_len(a) for a in items), default=0)
    padded = [pad_frac(a, max_frac) for a in items]
    # Work with absolute digit columns from the rightmost fractional place.
    digit_rows: list[list[int]] = []
    max_len = 0
    for p in padded:
        sign = -1 if p.startswith("-") else 1
        body = p.lstrip("+-").replace(".", "")
        digits = [sign * int(ch) for ch in body]
        digit_rows.append(digits)
        max_len = max(max_len, len(digits))
    # Left-pad integer parts so columns align from the right.
    aligned = [[0] * (max_len - len(row)) + row for row in digit_rows]
    carries = 0
    carry = 0
    for col in range(max_len - 1, -1, -1):
        total = carry + sum(row[col] for row in aligned)
        # Carry toward the next more-significant digit.
        next_carry = total // 10
        if next_carry != 0:
            carries += abs(next_carry)
        carry = next_carry
    return carries


def chain_sum(addends: Iterable[str]) -> str:
    total = sum((parse_decimal(a) for a in addends), Decimal(0))
    return normalize_decimal_str(total)


def partial_sums(addends: Iterable[str]) -> list[str]:
    running = Decimal(0)
    out: list[str] = []
    for a in addends:
        running += parse_decimal(a)
        out.append(normalize_decimal_str(running))
    return out
