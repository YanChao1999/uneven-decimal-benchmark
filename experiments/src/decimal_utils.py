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


def _aligned_digit_rows(addends: Iterable[str]) -> tuple[list[list[int]], int, int]:
    """Return (rows aligned from the right, max_len, max_frac)."""
    items = [normalize_decimal_str(a) for a in addends]
    max_frac = max((frac_len(a) for a in items), default=0)
    padded = [pad_frac(a, max_frac) for a in items]
    digit_rows: list[list[int]] = []
    max_len = 0
    for p in padded:
        sign = -1 if p.startswith("-") else 1
        body = p.lstrip("+-").replace(".", "")
        digits = [sign * int(ch) for ch in body]
        digit_rows.append(digits)
        max_len = max(max_len, len(digits))
    aligned = [[0] * (max_len - len(row)) + row for row in digit_rows]
    return aligned, max_len, max_frac


def column_carry_count(addends: Iterable[str]) -> int:
    """Count column-wise carries after aligning by decimal point (right-pad)."""
    return int(carry_profile(addends)["carry_count"])


def carry_profile(addends: Iterable[str]) -> dict[str, int | float | list[int]]:
    """Structural carry stats for difficulty labeling (not just a noisy count).

    Fractional places are indexed from the decimal point:
    place 0 = tenths, 1 = hundredths, ... (LSB among fractional columns).
    """
    aligned, max_len, max_frac = _aligned_digit_rows(addends)
    if max_len == 0:
        return {
            "carry_count": 0,
            "n_carry_columns": 0,
            "max_carry_run": 0,
            "odd_place_carries": 0,
            "even_place_carries": 0,
            "frac_carry_columns": 0,
            "carry_mask_frac": [],
        }

    carry_count = 0
    n_carry_columns = 0
    max_run = 0
    run = 0
    # mask over fractional places only, LSB-first (place 0 = tenths)
    carry_mask_frac: list[int] = [0] * max_frac
    odd_place = 0
    even_place = 0
    frac_carry_columns = 0

    carry = 0
    # Walk columns right→left; column index from right: 0 = least significant digit.
    for dist_from_right in range(max_len):
        col = max_len - 1 - dist_from_right
        total = carry + sum(row[col] for row in aligned)
        next_carry = total // 10
        produced = abs(next_carry) != 0
        if produced:
            carry_count += abs(next_carry)
            n_carry_columns += 1
            run += 1
            max_run = max(max_run, run)
            # Fractional place index if this column is in the frac region.
            if dist_from_right < max_frac:
                place = dist_from_right  # 0=tenths
                carry_mask_frac[place] = 1
                frac_carry_columns += 1
                if place % 2 == 1:
                    odd_place += 1
                else:
                    even_place += 1
        else:
            run = 0
        carry = next_carry

    return {
        "carry_count": carry_count,
        "n_carry_columns": n_carry_columns,
        "max_carry_run": max_run,
        "odd_place_carries": odd_place,
        "even_place_carries": even_place,
        "frac_carry_columns": frac_carry_columns,
        "carry_mask_frac": carry_mask_frac,
    }


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
