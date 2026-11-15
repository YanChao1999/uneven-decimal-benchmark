"""RASP sketch: decimal-point alignment → digitwise add → carry propagation.

The program is written as annotated stages using ``ops.select`` / ``ops.aggregate``.
It operates on decimal strings and folds them for chain sums.

Inherent Transformer/RASP pressure points (see ``analyze.py``):
- locating the decimal point (content-based select);
- aligning place values under uneven fractional lengths;
- carry is a *sequential* dependence along columns (hard for fixed-depth RASP
  without an explicit carry-scan of length O(width)).
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.src.decimal_utils import frac_len, normalize_decimal_str, pad_frac
from experiments.src.rasp import ops


DOT = "."
PAD = "0"


@dataclass(frozen=True)
class RaspTrace:
    """Intermediate tensors useful for difficulty analysis / debugging."""

    aligned_a: str
    aligned_b: str
    width: int
    frac_width: int
    digit_sums: list[int]
    carries: list[int]
    result_digits: list[int]
    result: str
    n_carry_steps: int
    max_carry_run: int


def _find_dot_index(tokens: list[str]) -> int:
    """RASP stage: select positions equal to '.' and read their index."""
    mask = ops.select(tokens, [DOT], pred=lambda k, q: k == q)
    hit = [i for i, bit in enumerate(mask[0]) if bit]
    if not hit:
        return len(tokens)
    return hit[0]


def align_pair(a: str, b: str) -> tuple[str, str, int]:
    """Align by decimal point via right-padding fractional parts to a common width.

    RASP reading: locate '.' with content-based ``select``, compute fractional
    lengths, take max, pad with zeros (``map``), then left-pad integer digits so
    place columns match.
    """
    a_n = normalize_decimal_str(a)
    b_n = normalize_decimal_str(b)
    # Demonstrate dot select on token sequences (side-effect-free documentation).
    _ = _find_dot_index(list(a_n.lstrip("+-")))
    _ = _find_dot_index(list(b_n.lstrip("+-")))

    max_f = max(frac_len(a_n), frac_len(b_n))
    aligned_a = pad_frac(a_n, max_f)
    aligned_b = pad_frac(b_n, max_f)

    def split_body(s: str) -> tuple[str, str, str]:
        sign = "-" if s.startswith("-") else ""
        body = s.lstrip("+-")
        if "." not in body:
            body = body + "."
        whole, frac = body.split(".", 1)
        return sign, whole, frac

    sa, wa, fa = split_body(aligned_a)
    sb, wb, fb = split_body(aligned_b)
    w = max(len(wa), len(wb))
    body_a = wa.rjust(w, PAD) + "." + fa
    body_b = wb.rjust(w, PAD) + "." + fb
    return sa + body_a, sb + body_b, w + 1 + max_f


def _signed_digits(aligned: str) -> list[int]:
    sign = -1 if aligned.startswith("-") else 1
    body = aligned.lstrip("+-").replace(".", "")
    return [sign * int(ch) for ch in body]


def digitwise_sum(digits_a: list[int], digits_b: list[int]) -> list[int]:
    """RASP ``zipmap`` of per-place digit values (before carry)."""
    if len(digits_a) != len(digits_b):
        raise ValueError("digit sequences must share length after alignment")
    return ops.zipmap(lambda x, y: x + y, digits_a, digits_b)


def carry_propagate(column_sums: list[int]) -> tuple[list[int], list[int], int, int]:
    """Sequential carry scan from least-significant place to most-significant.

    This is the RASP bottleneck: each carry depends on the previous column, so a
    faithful implementation is an O(width) chain of select-on-predecessor ops
    (or a long scan). Bounded-depth Transformers struggle when carry runs are long.
    """
    n = len(column_sums)
    # carries[i] enters column i from the right (LSB side). Extra slot for outbound.
    carries_in = [0] * (n + 1)
    results = [0] * n
    max_run = 0
    run = 0
    for pos in range(n - 1, -1, -1):
        total = column_sums[pos] + carries_in[pos + 1]
        carry = total // 10
        digit = total % 10
        # Normalize negative moduli for signed edge cases.
        if digit < 0:
            digit += 10
            carry -= 1
        results[pos] = digit
        carries_in[pos] = carry
        if abs(carry) > 0 or abs(total) >= 10:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    lead = carries_in[0]
    out_digits = results
    n_carry_steps = sum(1 for c in carries_in if c != 0)
    if lead != 0:
        out_digits = [lead] + results
        n_carry_steps += 1
    return out_digits, carries_in, n_carry_steps, max_run


def _digits_to_decimal(digits: list[int], frac_width: int) -> str:
    if not digits:
        return "0"
    # Reconstruct integer in units of 10^{-frac_width}.
    value = 0
    for d in digits:
        value = value * 10 + d
    scale = 10**frac_width
    sign = "-" if value < 0 else ""
    value = abs(value)
    if frac_width == 0:
        return normalize_decimal_str(f"{sign}{value}")
    whole, frac = divmod(value, scale)
    return normalize_decimal_str(f"{sign}{whole}.{str(frac).rjust(frac_width, '0')}")


def rasp_pair_add(a: str, b: str) -> RaspTrace:
    """Executable RASP-style pair addition with alignment and carry."""
    aligned_a, aligned_b, _width = align_pair(a, b)
    frac_width = frac_len(aligned_a.lstrip("+-"))
    da = _signed_digits(aligned_a)
    db = _signed_digits(aligned_b)
    sums = digitwise_sum(da, db)
    result_digits, carries, n_carry_steps, max_run = carry_propagate(sums)
    result = _digits_to_decimal(result_digits, frac_width)
    return RaspTrace(
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        width=len(da),
        frac_width=frac_width,
        digit_sums=sums,
        carries=carries,
        result_digits=result_digits,
        result=result,
        n_carry_steps=n_carry_steps,
        max_carry_run=max_run,
    )


def rasp_chain_sum(addends: list[str]) -> RaspTrace:
    """Fold ``rasp_pair_add`` left-to-right (chain sum).

    Each fold re-aligns under potentially new fractional widths — matching the
    paper's chain-sum task and compounding alignment difficulty.
    """
    if not addends:
        return RaspTrace("0", "0", 1, 0, [0], [0, 0], [0], "0", 0, 0)
    if len(addends) == 1:
        return rasp_pair_add(addends[0], "0")
    acc_trace = rasp_pair_add(addends[0], addends[1])
    for x in addends[2:]:
        acc_trace = rasp_pair_add(acc_trace.result, x)
    return acc_trace
