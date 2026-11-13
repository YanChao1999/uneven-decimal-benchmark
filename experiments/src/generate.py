"""Seeded generators with orthogonal alignment × carry difficulty factors."""

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
    partial_sums,
)

CarryBias = Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class Example:
    id: str
    # cluster_id is the primary sample-family key (e.g. A0C1).
    # stratum is kept as an alias of cluster_id for older eval code paths.
    stratum: str
    cluster_id: str
    alignment_level: int  # 0 = aligned, 1 = uneven
    carry_level: int  # 0 = low carry, 1 = high carry
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


def _digit_from_bias(rng: random.Random, bias: CarryBias) -> int:
    if bias == "moderate":
        return rng.randint(0, 9)
    if bias == "low":
        return rng.choices(range(10), weights=[5, 5, 4, 3, 2, 1, 1, 1, 1, 1], k=1)[0]
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
    if int_digits == 1:
        int_parts = [str(_digit_from_bias(rng, bias))]
    else:
        first = rng.randint(1, 9)
        rest = [_digit_from_bias(rng, bias) for _ in range(int_digits - 1)]
        int_parts = [str(first)] + [str(d) for d in rest]
    frac_parts = [str(_digit_from_bias(rng, bias)) for _ in range(frac_digits)]
    body = "".join(int_parts)
    if frac_digits > 0:
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
        alt = frac_min if lengths[0] != frac_min else frac_max
        lengths[rng.randrange(n)] = alt
    return lengths


def _trim_frac(s: str, width: int) -> str:
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
    """Force factor C in place-value columns without changing each addend's frac length."""
    orig_frac = [frac_len(a) for a in addends]
    max_f = max(orig_frac) if orig_frac else 0
    if max_f == 0 or len(addends) < 2:
        return addends

    parsed: list[tuple[str, str, list[str]]] = []
    for a in addends:
        sign = "-" if a.startswith("-") else ""
        body = a.lstrip("+-")
        if "." not in body:
            body = body + "."
        whole, frac = body.split(".", 1)
        parsed.append((sign, whole, list(frac)))

    # Column i = fractional place 10^{-(i+1)} (0 = tenths).
    for i in range(max_f):
        present = [k for k, (_s, _w, frac) in enumerate(parsed) if i < len(frac)]
        if len(present) < 2:
            # Single addend owns this finer place: keep small for low, mid for high.
            for k in present:
                parsed[k][2][i] = str(rng.randint(5, 9) if bias == "high" else rng.randint(0, 2))
            continue
        if bias == "high":
            for k in present:
                parsed[k][2][i] = str(rng.randint(5, 9))
        else:
            # Keep sum of digits in this column < 10.
            budget = 8
            for j, k in enumerate(present):
                remaining = len(present) - j - 1
                hi = min(4, budget - remaining * 0)
                hi = max(0, min(hi, budget))
                d = rng.randint(0, hi) if hi > 0 else 0
                parsed[k][2][i] = str(d)
                budget -= d

    out: list[str] = []
    for idx, (sign, whole, frac) in enumerate(parsed):
        # Preserve exact original fractional width.
        width = orig_frac[idx]
        out.append(f"{sign}{whole}.{''.join(frac[:width])}" if width else f"{sign}{whole}")
    return out


def generate_example(
    rng: random.Random,
    *,
    cluster_id: str,
    alignment_level: int,
    carry_level: int,
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
    if carry_bias in {"low", "high"} and n_addends >= 2:
        addends = _reshape_for_carry_bias(rng, addends, carry_bias)

    expected = chain_sum(addends)
    parts = partial_sums(addends)
    carries = column_carry_count(addends)
    measured_frac = [frac_len(a) for a in addends]
    return Example(
        id=f"{cluster_id}_{example_index:05d}",
        stratum=cluster_id,
        cluster_id=cluster_id,
        alignment_level=alignment_level,
        carry_level=carry_level,
        addends=addends,
        expected_sum=expected,
        partial_sums=parts,
        n_addends=len(addends),
        int_lens=[int_len(a) for a in addends],
        frac_lens=measured_frac,
        carry_count=carries,
        aligned=aligned and len(set(measured_frac)) <= 1,
        seed=seed,
        meta={
            "carry_bias": carry_bias,
            "alignment_factor": f"A{alignment_level}",
            "carry_factor": f"C{carry_level}",
        },
    )


def _legacy_strata_to_clusters(config: dict[str, Any]) -> dict[str, Any]:
    """Map old three-stratum YAML into factorial clusters for compatibility."""
    mapping = {
        "aligned": {"alignment_level": 0, "carry_level": 0, "aligned": True, "carry_bias": "moderate"},
        "uneven_low_carry": {
            "alignment_level": 1,
            "carry_level": 0,
            "aligned": False,
            "carry_bias": "low",
        },
        "uneven_high_carry": {
            "alignment_level": 1,
            "carry_level": 1,
            "aligned": False,
            "carry_bias": "high",
        },
    }
    clusters: dict[str, Any] = {}
    for name, block in config["strata"].items():
        base = mapping.get(name, {
            "alignment_level": int(block.get("alignment_level", 0)),
            "carry_level": int(block.get("carry_level", 0)),
            "aligned": bool(block.get("aligned", False)),
            "carry_bias": block.get("carry_bias", "moderate"),
        })
        cluster_id = f"A{base['alignment_level']}C{base['carry_level']}"
        # Avoid collisions if multiple legacy names map to same cell: keep name as id.
        if name in mapping:
            # Prefer explicit A/C ids when using the new factorial view of legacy sets.
            # Keep separate if both would collide — append suffix.
            if cluster_id in clusters:
                cluster_id = f"{cluster_id}_{name}"
            else:
                # For aligned+moderate, treat as A0C0; if A0C1 also needed later, fine.
                pass
            # Use stable names matching factor design for the three classic strata:
            if name == "aligned":
                cluster_id = "A0C0"
            elif name == "uneven_low_carry":
                cluster_id = "A1C0"
            elif name == "uneven_high_carry":
                cluster_id = "A1C1"
        clusters[cluster_id] = {**base, **block, "cluster_id": cluster_id}
    return clusters


def _resolve_clusters(config: dict[str, Any]) -> dict[str, Any]:
    if "clusters" in config:
        factors = config.get("factors", {})
        align_defs = factors.get("alignment", {})
        carry_defs = factors.get("carry", {})
        out: dict[str, Any] = {}
        for cluster_id, block in config["clusters"].items():
            a_key = block.get("alignment", block.get("alignment_factor"))
            c_key = block.get("carry", block.get("carry_factor"))
            a_def = dict(align_defs.get(a_key, {})) if a_key else {}
            c_def = dict(carry_defs.get(c_key, {})) if c_key else {}
            # Parse A0 / C1 style if levels not explicit.
            alignment_level = int(block.get("alignment_level", a_def.get("level", 0)))
            carry_level = int(block.get("carry_level", c_def.get("level", 0)))
            if isinstance(a_key, str) and a_key.startswith("A") and a_key[1:].isdigit():
                alignment_level = int(a_key[1:])
            if isinstance(c_key, str) and c_key.startswith("C") and c_key[1:].isdigit():
                carry_level = int(c_key[1:])
            merged = {
                **a_def,
                **c_def,
                **block,
                "alignment_level": alignment_level,
                "carry_level": carry_level,
                "aligned": bool(block.get("aligned", a_def.get("aligned", alignment_level == 0))),
                "carry_bias": block.get(
                    "carry_bias",
                    c_def.get("carry_bias", "high" if carry_level >= 1 else "low"),
                ),
                "cluster_id": cluster_id,
            }
            out[cluster_id] = merged
        return out
    if "strata" in config:
        return _legacy_strata_to_clusters(config)
    raise ValueError("Config must define `clusters` or legacy `strata`")


def iter_dataset(config: dict[str, Any]) -> Iterator[Example]:
    seed = int(config.get("seed", 0))
    defaults = dict(config.get("defaults", {}))
    clusters = _resolve_clusters(config)
    for cluster_id, block in clusters.items():
        count = int(block["count"])
        merged = {**defaults, **block}
        for i in range(count):
            digest = hashlib.sha256(f"{seed}:{cluster_id}:{i}".encode()).hexdigest()
            example_seed = int(digest[:16], 16) % 1_000_000_003
            rng = random.Random(example_seed)
            yield generate_example(
                rng,
                cluster_id=cluster_id,
                alignment_level=int(merged["alignment_level"]),
                carry_level=int(merged["carry_level"]),
                example_index=i,
                n_addends=int(merged["n_addends"]),
                int_digits_min=int(merged["int_digits_min"]),
                int_digits_max=int(merged["int_digits_max"]),
                frac_digits_min=int(merged["frac_digits_min"]),
                frac_digits_max=int(merged["frac_digits_max"]),
                aligned=bool(merged.get("aligned", False)),
                carry_bias=merged.get("carry_bias", "moderate"),  # type: ignore[arg-type]
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
