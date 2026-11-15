"""Seeded generators with orthogonal alignment × carry difficulty factors."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from experiments.src.decimal_utils import (
    carry_profile,
    chain_sum,
    column_carry_count,
    frac_len,
    int_len,
    partial_sums,
)

CarryBias = Literal["low", "moderate", "high"]
# Structural C modes (preferred over digit-bias alone — less N-confounded noise).
CarryStructure = Literal[
    "none",        # no fractional carries
    "isolated",    # sparse non-adjacent carries
    "odd_places",  # carry on odd frac places only (1,3,5…)
    "even_places", # carry on even frac places only (0,2,4…)
    "chain",       # consecutive multi-place carry run
    "density",     # target carry-column density ρ = k / F (length-normalized)
]
DensityLayout = Literal["chain", "spread"]  # how to place the k forced carries


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


def _parse_addends_for_reshape(
    addends: list[str],
) -> tuple[list[tuple[str, str, list[str]]], list[int], int]:
    orig_frac = [frac_len(a) for a in addends]
    max_f = max(orig_frac) if orig_frac else 0
    parsed: list[tuple[str, str, list[str]]] = []
    for a in addends:
        sign = "-" if a.startswith("-") else ""
        body = a.lstrip("+-")
        if "." not in body:
            body = body + "."
        whole, frac = body.split(".", 1)
        parsed.append((sign, whole, list(frac)))
    return parsed, orig_frac, max_f


def _format_reshaped(
    parsed: list[tuple[str, str, list[str]]], orig_frac: list[int]
) -> list[str]:
    out: list[str] = []
    for idx, (sign, whole, frac) in enumerate(parsed):
        width = orig_frac[idx]
        out.append(f"{sign}{whole}.{''.join(frac[:width])}" if width else f"{sign}{whole}")
    return out


def _frac_str_index(lsb_place: int, frac_len_k: int, max_f: int) -> int | None:
    """Map LSB-first place (0=finest digit) → left-to-right index in this addend's frac.

    Example: max_f=4, lsb_place=0 → string index 3 (rightmost).
    """
    str_i = max_f - 1 - lsb_place
    if str_i < 0 or str_i >= frac_len_k:
        return None
    return str_i


def _set_column_no_carry(
    rng: random.Random,
    parsed: list[tuple[str, str, list[str]]],
    lsb_place: int,
    max_f: int,
    *,
    absorb_incoming: bool = False,
) -> None:
    """Force digit-sum low at LSB-first fractional place (0=finest).

    If ``absorb_incoming``, use budget ≤ 8 so an incoming carry of 1 still
    yields total ≤ 9 (stops propagation into the next coarser place).
    """
    present_idx: list[tuple[int, int]] = []  # (addend_k, str_index)
    for k, (_s, _w, frac) in enumerate(parsed):
        si = _frac_str_index(lsb_place, len(frac), max_f)
        if si is not None:
            present_idx.append((k, si))
    if not present_idx:
        return
    budget = 8 if absorb_incoming else 9
    if len(present_idx) == 1:
        k, si = present_idx[0]
        parsed[k][2][si] = str(rng.randint(0, min(4, budget)))
        return
    for j, (k, si) in enumerate(present_idx):
        hi = max(0, budget)
        d = rng.randint(0, hi) if hi > 0 else 0
        parsed[k][2][si] = str(d)
        budget -= d


def _set_column_force_carry(
    rng: random.Random,
    parsed: list[tuple[str, str, list[str]]],
    lsb_place: int,
    max_f: int,
) -> None:
    """Force digit-sum ≥ 10 at LSB-first fractional place (independent of incoming carry)."""
    present_idx: list[tuple[int, int]] = []
    for k, (_s, _w, frac) in enumerate(parsed):
        si = _frac_str_index(lsb_place, len(frac), max_f)
        if si is not None:
            present_idx.append((k, si))
    if not present_idx:
        return
    if len(present_idx) == 1:
        k, si = present_idx[0]
        parsed[k][2][si] = "9"
        return
    for k, si in present_idx:
        parsed[k][2][si] = "0"
    n = len(present_idx)
    # Cap at 15 so outgoing carry is exactly 1 — keeps absorb columns effective
    # (incoming≤1 ⇒ digit-sum≤8 still stops propagation).
    target = rng.randint(10, min(9 * n, 15))
    (k0, s0), (k1, s1) = present_idx[0], present_idx[1]
    d0 = min(9, target - 1)
    d1 = min(9, target - d0)
    parsed[k0][2][s0] = str(d0)
    parsed[k1][2][s1] = str(d1)
    rem = target - d0 - d1
    for k, si in present_idx[2:]:
        if rem <= 0:
            parsed[k][2][si] = str(rng.randint(0, 2))
        else:
            d = min(9, rem)
            parsed[k][2][si] = str(d)
            rem -= d

def _zero_integer_carries(
    rng: random.Random, parsed: list[tuple[str, str, list[str]]]
) -> None:
    """Keep integer-place digit sums < 10 so C0 is not polluted by int carries."""
    max_int = max((len(whole) for _s, whole, _f in parsed), default=0)
    if max_int == 0:
        return
    # Work right-aligned on integer digits.
    int_digits: list[list[str]] = []
    for _s, whole, _f in parsed:
        digits = list(whole) if whole else ["0"]
        int_digits.append(["0"] * (max_int - len(digits)) + digits)
    for col in range(max_int):
        budget = 9
        for k in range(len(parsed)):
            hi = max(0, budget)
            # MSD (col==0) of a multi-digit int should stay non-zero when possible.
            if col == 0 and max_int > 1 and k == 0 and hi >= 1:
                d = rng.randint(1, min(3, hi))
            else:
                d = rng.randint(0, min(3, hi)) if hi > 0 else 0
            int_digits[k][col] = str(d)
            budget -= d
    for k, (sign, _whole, frac) in enumerate(parsed):
        body = "".join(int_digits[k]).lstrip("0") or "0"
        parsed[k] = (sign, body, frac)

def _reshape_for_carry_bias(
    rng: random.Random, addends: list[str], bias: CarryBias
) -> list[str]:
    """Legacy digit-bias reshape (noisy; prefer ``carry_structure``)."""
    parsed, orig_frac, max_f = _parse_addends_for_reshape(addends)
    if max_f == 0 or len(addends) < 2:
        return addends

    for i in range(max_f):
        present = [k for k, (_s, _w, frac) in enumerate(parsed) if i < len(frac)]
        if len(present) < 2:
            for k in present:
                parsed[k][2][i] = str(rng.randint(5, 9) if bias == "high" else rng.randint(0, 2))
            continue
        if bias == "high":
            for k in present:
                parsed[k][2][i] = str(rng.randint(5, 9))
        else:
            # i is left-to-right; convert to LSB-first for the helper.
            _set_column_no_carry(rng, parsed, max_f - 1 - i, max_f)

    return _format_reshaped(parsed, orig_frac)


def _force_places_for_density(
    max_f: int,
    target_density: float,
    layout: DensityLayout,
) -> set[int]:
    """Choose k≈ρ·F LSB-first places to force-carry (length-normalized C)."""
    if max_f <= 0:
        return set()
    rho = max(0.0, min(1.0, float(target_density)))
    k = int(round(rho * max_f))
    k = max(0, min(max_f, k))
    if k == 0:
        return set()
    if layout == "chain":
        return set(range(0, k))
    if layout == "spread":
        if k == 1:
            return {0}
        # Evenly spaced across the frac length (LSB-first indices).
        return {int(round(i * (max_f - 1) / (k - 1))) for i in range(k)}
    raise ValueError(f"Unknown density_layout: {layout!r}")


def _reshape_for_carry_structure(
    rng: random.Random,
    addends: list[str],
    structure: CarryStructure,
    *,
    min_carry_run: int = 3,
    target_carry_density: float | None = None,
    density_layout: DensityLayout = "spread",
) -> list[str]:
    """Force structural carry patterns on fractional places (lengths preserved).

    Modes are designed to be less confounded with N than raw digit-bias:
    each column is explicitly forced to carry or not-carry.

    ``density`` (or any mode with ``target_carry_density``) sets
    ρ = (# forced carry places) / F, so C is length-normalized against F.
    """
    parsed, orig_frac, max_f = _parse_addends_for_reshape(addends)
    if max_f == 0 or len(addends) < 2:
        return addends

    # Force sets use LSB-first place ids (0 = finest / rightmost frac digit),
    # matching carry propagation direction and ``carry_profile`` masks.
    force: set[int] = set()
    if target_carry_density is not None or structure == "density":
        rho = 0.0 if target_carry_density is None else float(target_carry_density)
        force = _force_places_for_density(max_f, rho, density_layout)
    elif structure == "none":
        force = set()
    elif structure == "isolated":
        start = rng.randint(0, 1) if max_f > 1 else 0
        force = set(range(start, max_f, 2))
        if max_f >= 1 and not force:
            force = {0}
    elif structure == "odd_places":
        force = {i for i in range(max_f) if i % 2 == 1}
        if max_f >= 2 and not force:
            force = {1}
    elif structure == "even_places":
        force = {i for i in range(max_f) if i % 2 == 0}
    elif structure == "chain":
        run = min(max(1, min_carry_run), max_f)
        # Classic pencil-and-paper: chain starts at the finest places.
        force = set(range(0, run))
    else:
        raise ValueError(f"Unknown carry_structure: {structure!r}")

    # Absorb on non-forced places whenever we care about a clean pattern/density.
    # (Also for chain/density so run does not bleed into unused F length.)
    absorb = structure in {
        "isolated",
        "odd_places",
        "even_places",
        "chain",
        "density",
    } or target_carry_density is not None
    for lsb in range(max_f):
        if lsb in force:
            _set_column_force_carry(rng, parsed, lsb, max_f)
        else:
            _set_column_no_carry(
                rng, parsed, lsb, max_f, absorb_incoming=absorb
            )

    # Density / none: kill integer-place carries so ρ is frac-only and N-stable.
    if structure in {"none", "density"} or target_carry_density is not None:
        _zero_integer_carries(rng, parsed)

    return _format_reshaped(parsed, orig_frac)


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
    frac_level: int = 0,
    seed: int,
    carry_structure: CarryStructure | None = None,
    min_carry_run: int = 3,
    target_carry_density: float | None = None,
    density_layout: DensityLayout = "spread",
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
    # When using structural C, sample digits neutrally then reshape columns.
    sample_bias: CarryBias = "moderate" if carry_structure else carry_bias
    addends = [
        _random_decimal_str(
            rng,
            int_digits=int_lens[i],
            frac_digits=frac_lens[i],
            bias=sample_bias,
            allow_negative=allow_negative,
        )
        for i in range(n_addends)
    ]
    if carry_structure is not None and n_addends >= 2:
        addends = _reshape_for_carry_structure(
            rng,
            addends,
            carry_structure,
            min_carry_run=min_carry_run,
            target_carry_density=target_carry_density,
            density_layout=density_layout,
        )
    elif carry_bias in {"low", "high"} and n_addends >= 2:
        addends = _reshape_for_carry_bias(rng, addends, carry_bias)

    expected = chain_sum(addends)
    parts = partial_sums(addends)
    profile = carry_profile(addends)
    carries = int(profile["carry_count"])
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
            "carry_structure": carry_structure,
            "min_carry_run": min_carry_run if carry_structure == "chain" else None,
            "target_carry_density": target_carry_density,
            "density_layout": density_layout if carry_structure == "density" or target_carry_density is not None else None,
            "max_carry_run": profile["max_carry_run"],
            "n_carry_columns": profile["n_carry_columns"],
            "odd_place_carries": profile["odd_place_carries"],
            "even_place_carries": profile["even_place_carries"],
            "carry_mask_frac": profile["carry_mask_frac"],
            "carry_col_density": profile["carry_col_density"],
            "carry_run_density": profile["carry_run_density"],
            "carry_amount_density": profile["carry_amount_density"],
            "alignment_factor": f"A{alignment_level}",
            "carry_factor": f"C{carry_level}",
            "frac_level": frac_level,
            "frac_factor": f"F{frac_level}",
            "n_addends_factor": f"N{n_addends}",
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


def _factor_axis_keys(factors: dict[str, Any], *names: str) -> list[str]:
    for name in names:
        block = factors.get(name)
        if isinstance(block, dict) and block:
            return list(block.keys())
    return []


def _expand_factorial_cluster_specs(config: dict[str, Any]) -> dict[str, Any]:
    """Cartesian product of factor axes → cluster specs (ids like F2C1N5).

    Enabled when ``expand_factorial: true``. Optional ``clusters`` may be a
    mapping of overrides (``count``, extra fields) keyed by cluster id, or a
    mapping with only ``count`` / defaults applied to every cell via
    ``per_cluster_count``.
    """
    factors = config.get("factors", {})
    if not isinstance(factors, dict) or not factors:
        raise ValueError("expand_factorial requires non-empty `factors`")

    f_keys = _factor_axis_keys(factors, "frac", "fractional_length")
    c_keys = _factor_axis_keys(factors, "carry")
    n_keys = _factor_axis_keys(factors, "n_addends", "chain_length")
    a_keys = _factor_axis_keys(factors, "alignment")

    if not f_keys or not c_keys or not n_keys:
        raise ValueError(
            "expand_factorial needs factors.frac, factors.carry, and factors.n_addends"
        )

    per_count = int(config.get("per_cluster_count", 10))
    overrides_raw = config.get("clusters", {})
    overrides: dict[str, Any] = overrides_raw if isinstance(overrides_raw, dict) else {}

    specs: dict[str, Any] = {}
    align_iter = a_keys if a_keys else [None]
    for a_key in align_iter:
        for f_key in f_keys:
            for c_key in c_keys:
                for n_key in n_keys:
                    parts = []
                    if a_key is not None:
                        parts.append(str(a_key))
                    parts.extend([str(f_key), str(c_key), str(n_key)])
                    cluster_id = "".join(parts)
                    block: dict[str, Any] = {
                        "count": per_count,
                        "frac": f_key,
                        "carry": c_key,
                        "n_addends_factor": n_key,
                    }
                    if a_key is not None:
                        block["alignment"] = a_key
                    if cluster_id in overrides and isinstance(overrides[cluster_id], dict):
                        block = {**block, **overrides[cluster_id]}
                    specs[cluster_id] = block
    return specs


def _merge_cluster_block(
    cluster_id: str,
    block: dict[str, Any],
    factors: dict[str, Any],
) -> dict[str, Any]:
    align_defs = factors.get("alignment", {})
    carry_defs = factors.get("carry", {})
    frac_defs = factors.get("frac", factors.get("fractional_length", {}))
    n_defs = factors.get("n_addends", factors.get("chain_length", {}))

    a_key = block.get("alignment", block.get("alignment_factor"))
    c_key = block.get("carry", block.get("carry_factor"))
    f_key = block.get("frac", block.get("frac_factor", block.get("fractional_length")))
    n_key = block.get("n_addends_factor", block.get("chain_length", block.get("n_factor")))
    a_def = dict(align_defs.get(a_key, {})) if a_key else {}
    c_def = dict(carry_defs.get(c_key, {})) if c_key else {}
    f_def = dict(frac_defs.get(f_key, {})) if f_key else {}
    n_def = dict(n_defs.get(n_key, {})) if n_key else {}
    # Parse A0 / C1 / F2 / N5 style if levels not explicit.
    alignment_level = int(block.get("alignment_level", a_def.get("level", 0)))
    carry_level = int(block.get("carry_level", c_def.get("level", 0)))
    frac_level = int(block.get("frac_level", f_def.get("level", 0)))
    if isinstance(a_key, str) and a_key.startswith("A") and a_key[1:].isdigit():
        alignment_level = int(a_key[1:])
    if isinstance(c_key, str) and c_key.startswith("C") and c_key[1:].isdigit():
        carry_level = int(c_key[1:])
    if isinstance(f_key, str) and f_key.startswith("F") and f_key[1:].isdigit():
        frac_level = int(f_key[1:])
    n_addends = block.get("n_addends", n_def.get("n_addends", n_def.get("n")))
    if isinstance(n_key, str) and n_key.startswith("N") and n_key[1:].isdigit():
        n_addends = int(n_key[1:])
    bias_default = "low"
    if carry_level >= 2:
        bias_default = "high"
    elif carry_level == 1:
        bias_default = "moderate"
    carry_structure = block.get("carry_structure", c_def.get("carry_structure"))
    min_carry_run = int(block.get("min_carry_run", c_def.get("min_carry_run", 3)))
    raw_density = block.get("target_carry_density", c_def.get("target_carry_density"))
    target_carry_density = float(raw_density) if raw_density is not None else None
    density_layout = str(
        block.get("density_layout", c_def.get("density_layout", "spread"))
    )
    merged = {
        **a_def,
        **c_def,
        **f_def,
        **n_def,
        **block,
        "alignment_level": alignment_level,
        "carry_level": carry_level,
        "frac_level": frac_level,
        "aligned": bool(block.get("aligned", a_def.get("aligned", alignment_level == 0))),
        "carry_bias": block.get(
            "carry_bias",
            c_def.get("carry_bias", bias_default),
        ),
        "carry_structure": carry_structure,
        "min_carry_run": min_carry_run,
        "target_carry_density": target_carry_density,
        "density_layout": density_layout,
        "cluster_id": cluster_id,
    }
    if n_addends is not None:
        merged["n_addends"] = int(n_addends)
    return merged


def _resolve_clusters(config: dict[str, Any]) -> dict[str, Any]:
    factors = config.get("factors", {}) if isinstance(config.get("factors"), dict) else {}
    if config.get("expand_factorial"):
        specs = _expand_factorial_cluster_specs(config)
        return {
            cid: _merge_cluster_block(cid, block, factors) for cid, block in specs.items()
        }
    if "clusters" in config:
        out: dict[str, Any] = {}
        for cluster_id, block in config["clusters"].items():
            out[cluster_id] = _merge_cluster_block(cluster_id, dict(block), factors)
        return out
    if "strata" in config:
        return _legacy_strata_to_clusters(config)
    raise ValueError("Config must define `clusters`, `expand_factorial`, or legacy `strata`")


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
            structure = merged.get("carry_structure")
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
                frac_level=int(merged.get("frac_level", 0)),
                seed=example_seed,
                carry_structure=structure,  # type: ignore[arg-type]
                min_carry_run=int(merged.get("min_carry_run", 3)),
                target_carry_density=merged.get("target_carry_density"),
                density_layout=str(merged.get("density_layout", "spread")),  # type: ignore[arg-type]
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
