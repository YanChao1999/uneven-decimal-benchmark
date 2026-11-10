"""Seeded generators for aligned / uneven low-carry / uneven high-carry chain sums."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from experiments.src.decimal_utils import (
    chain_sum,
    column_carry_count,
    frac_len,
    int_len,
    normalize_decimal_str,
    partial_sums,
)

CarryBias = Literal["low", "moderate", "high"]
StratumId = Literal["aligned", "uneven_low_carry", "uneven_high_carry"]


@dataclass(frozen=True)
class Example:
    id: str
    stratum: StratumId
    addends: list[str]
    expected_sum: str
    partial_sums: list[str]
    n_addends: int
    int_lens: list[int]
    frac_lens: list[int]
    carry_count: int
    aligned: bool
    seed: int
    meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _digit_from_bias(rng: random.Random, bias: CarryBias, partner_hint: int | None = None) -> int:
    """Sample a digit under a carry-oriented bias."""
    if bias == "moderate":
        return rng.randint(0, 9)
    if bias == "low":
        # Prefer small digits so column sums rarely reach 10.
        return rng.choices(range(10), weights=[5, 5, 4, 3, 2, 1, 1, 1, 1, 1], k=1)[0]
    # high: prefer digits that induce carries when paired with another high digit
    if partner_hint is not None and partner_hint >= 5:
        return rng.randint(5, 9)
    return rng.choices(range(10), weights=[1, 1, 1, 1, 1, 3, 4, 5, 5, 5], k=1)[0]


def _random_decimal_str(
    rng: random.Random,
    *,
    int_digits: int,
    frac_digits: int,
    bias: CarryBias,
    allow_negative: bool,
) -> str:
    if int_digits < 1:
        raise ValueError("int_digits must be >= 1")
    # Leading integer digit non-zero when int_digits > 1, allow 0 when length 1.
    if int_digits == 1:
        int_parts = [str(_digit_from_bias(rng, bias))]
    else:
        first = rng.randint(1, 9)
        rest = [_digit_from_bias(rng, bias) for _ in range(int_digits - 1)]
        int_parts = [str(first)] + [str(d) for d in rest]
    frac_parts = [str(_digit_from_bias(rng, bias)) for _ in range(frac_digits)]
    body = "".join(int_parts)
    if frac_digits > 0:
        # Keep trailing zeros so aligned strata stay visually equal-length.
        body = f"{body}.{''.join(frac_parts)}"
    if allow_negative and rng.random() < 0.2:
        body = f"-{body}"
    return body


def _choose_frac_lengths(
    rng: random.Random,
    *,
    n: int,
    aligned: bool,
    frac_digits: int | None,
    frac_min: int,
    frac_max: int,
) -> list[int]:
    if aligned:
        width = frac_digits if frac_digits is not None else rng.randint(frac_min, frac_max)
        return [width] * n
    if frac_min == frac_max or n == 1:
        return [rng.randint(frac_min, frac_max) for _ in range(n)]
    lengths = [rng.randint(frac_min, frac_max) for _ in range(n)]
    if len(set(lengths)) == 1:
        # Guarantee uneven fractional lengths for uneven strata.
        alt = frac_min if lengths[0] != frac_min else frac_max
        lengths[rng.randrange(n)] = alt
    return lengths


def generate_example(
    rng: random.Random,
    *,
    stratum: StratumId,
    example_index: int,
    n_addends: int,
    int_digits_min: int,
    int_digits_max: int,
    frac_digits_min: int,
    frac_digits_max: int,
    aligned: bool,
    carry_bias: CarryBias,
    allow_negative: bool = False,
    frac_digits: int | None = None,
    seed: int,
) -> Example:
    frac_lens = _choose_frac_lengths(
        rng,
        n=n_addends,
        aligned=aligned,
        frac_digits=frac_digits,
        frac_min=frac_digits_min,
        frac_max=frac_digits_max,
    )
    int_lens = [rng.randint(int_digits_min, int_digits_max) for _ in range(n_addends)]
    addends = [
        _random_decimal_str(
            rng,
            int_digits=int_lens[i],
            frac_digits=frac_lens[i],
            bias=carry_bias,
            allow_negative=allow_negative,
        )
        for i in range(n_addends)
    ]

    # For high/low carry strata, optionally reshape digits once using column bias.
    if carry_bias in {"low", "high"} and n_addends >= 2:
        addends = _reshape_for_carry_bias(rng, addends, carry_bias)

    expected = chain_sum(addends)
    parts = partial_sums(addends)
    carries = column_carry_count(addends)
    return Example(
        id=f"{stratum}_{example_index:05d}",
        stratum=stratum,
        addends=addends,
        expected_sum=expected,
        partial_sums=parts,
        n_addends=len(addends),
        int_lens=[int_len(a) for a in addends],
        frac_lens=[frac_len(a) for a in addends],
        carry_count=carries,
        aligned=aligned and len(set(frac_lens)) == 1,
        seed=seed,
        meta={"carry_bias": carry_bias},
    )


def _trim_frac(s: str, width: int) -> str:
    """Keep exactly ``width`` fractional digits (no normalize / no strip)."""
    sign = "-" if s.startswith("-") else ""
    body = s.lstrip("+-")
    if width <= 0:
        return sign + body.split(".", 1)[0]
    if "." not in body:
        body = body + "."
    whole, frac = body.split(".", 1)
    return f"{sign}{whole}.{frac[:width].ljust(width, '0')}"


def _reshape_for_carry_bias(
    rng: random.Random, addends: list[str], bias: CarryBias
) -> list[str]:
    """Nudge fractional digits of the first two addends toward / away from carries."""
    from experiments.src.decimal_utils import pad_frac

    orig_frac = [frac_len(a) for a in addends]
    max_f = max(orig_frac) if orig_frac else 0
    if max_f == 0:
        return addends
    padded = [pad_frac(a, max_f) for a in addends]
    a0, a1 = padded[0], padded[1]
    sign0, body0 = (a0[0], a0[1:]) if a0[0] in "+-" else ("", a0)
    sign1, body1 = (a1[0], a1[1:]) if a1[0] in "+-" else ("", a1)
    w0, f0 = body0.split(".", 1)
    w1, f1 = body1.split(".", 1)
    f0_list = list(f0)
    f1_list = list(f1)
    for i in range(len(f0_list)):
        if bias == "high":
            f0_list[i] = str(rng.randint(5, 9))
            f1_list[i] = str(rng.randint(5, 9))
        else:
            d0 = rng.randint(0, 4)
            f0_list[i] = str(d0)
            f1_list[i] = str(rng.randint(0, 9 - d0))
    new0 = _trim_frac(f"{sign0}{w0}.{''.join(f0_list)}", orig_frac[0])
    new1 = _trim_frac(f"{sign1}{w1}.{''.join(f1_list)}", orig_frac[1])
    return [new0, new1] + addends[2:]


def iter_dataset(config: dict[str, Any]) -> Iterator[Example]:
    seed = int(config.get("seed", 0))
    defaults = dict(config.get("defaults", {}))
    strata = config["strata"]
    for stratum_name, block in strata.items():
        stratum: StratumId = stratum_name  # type: ignore[assignment]
        count = int(block["count"])
        merged = {**defaults, **block}
        carry_bias: CarryBias = merged.get("carry_bias", "moderate")
        for i in range(count):
            # Independent sub-seed per example (stable across PYTHONHASHSEED).
            digest = hashlib.sha256(f"{seed}:{stratum}:{i}".encode()).hexdigest()
            example_seed = int(digest[:16], 16) % 1_000_000_003
            rng = random.Random(example_seed)
            yield generate_example(
                rng,
                stratum=stratum,
                example_index=i,
                n_addends=int(merged["n_addends"]),
                int_digits_min=int(merged["int_digits_min"]),
                int_digits_max=int(merged["int_digits_max"]),
                frac_digits_min=int(merged["frac_digits_min"]),
                frac_digits_max=int(merged["frac_digits_max"]),
                aligned=bool(merged.get("aligned", False)),
                carry_bias=carry_bias,
                allow_negative=bool(merged.get("allow_negative", False)),
                frac_digits=merged.get("frac_digits"),
                seed=example_seed,
            )


def write_jsonl(path: Path, examples: Iterator[Example] | list[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_json(), ensure_ascii=True) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data
