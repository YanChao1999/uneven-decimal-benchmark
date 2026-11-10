"""RASP difficulty proxies and notes for scale-trend hypotheses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.src.decimal_utils import column_carry_count, frac_len, normalize_decimal_str
from experiments.src.rasp.decimal_add import rasp_chain_sum


@dataclass(frozen=True)
class InstanceDifficulty:
    example_id: str
    stratum: str
    n_addends: int
    max_frac: int
    frac_unevenness: int  # max_frac - min_frac
    width_after_align: int
    carry_count: int
    rasp_n_carry_steps: int
    rasp_max_carry_run: int
    rasp_result: str
    expected_sum: str
    rasp_matches_expected: bool
    # Heuristic "RASP hardness" score used when plotting vs model scale.
    rasp_hardness: float


def _hardness(
    *,
    n_addends: int,
    frac_unevenness: int,
    width: int,
    max_carry_run: int,
    carry_count: int,
) -> float:
    """Scalar proxy: longer carry runs and uneven alignment dominate."""
    return (
        0.35 * max_carry_run
        + 0.25 * carry_count
        + 0.20 * frac_unevenness
        + 0.10 * width
        + 0.10 * max(0, n_addends - 2)
    )


def analyze_instance(example: dict[str, Any]) -> InstanceDifficulty:
    addends = list(example["addends"])
    frac_lens = [frac_len(normalize_decimal_str(a)) for a in addends]
    max_f = max(frac_lens) if frac_lens else 0
    min_f = min(frac_lens) if frac_lens else 0
    trace = rasp_chain_sum(addends)
    expected = example["expected_sum"]
    matches = normalize_decimal_str(trace.result) == normalize_decimal_str(expected)
    return InstanceDifficulty(
        example_id=example["id"],
        stratum=example["stratum"],
        n_addends=len(addends),
        max_frac=max_f,
        frac_unevenness=max_f - min_f,
        width_after_align=trace.width,
        carry_count=column_carry_count(addends),
        rasp_n_carry_steps=trace.n_carry_steps,
        rasp_max_carry_run=trace.max_carry_run,
        rasp_result=trace.result,
        expected_sum=expected,
        rasp_matches_expected=matches,
        rasp_hardness=_hardness(
            n_addends=len(addends),
            frac_unevenness=max_f - min_f,
            width=trace.width,
            max_carry_run=trace.max_carry_run,
            carry_count=column_carry_count(addends),
        ),
    )


def difficulty_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [analyze_instance(ex) for ex in examples]
    by_stratum: dict[str, list[InstanceDifficulty]] = {}
    for row in rows:
        by_stratum.setdefault(row.stratum, []).append(row)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    stratum_summary = {
        name: {
            "n": len(items),
            "mean_hardness": avg([x.rasp_hardness for x in items]),
            "mean_max_carry_run": avg([float(x.rasp_max_carry_run) for x in items]),
            "mean_frac_unevenness": avg([float(x.frac_unevenness) for x in items]),
            "rasp_agreement": avg([1.0 if x.rasp_matches_expected else 0.0 for x in items]),
        }
        for name, items in sorted(by_stratum.items())
    }
    return {
        "n": len(rows),
        "rasp_agreement": avg([1.0 if r.rasp_matches_expected else 0.0 for r in rows]),
        "mean_hardness": avg([r.rasp_hardness for r in rows]),
        "by_stratum": stratum_summary,
        "instances": [asdict(r) for r in rows],
        "inherent_difficulties": INHERENT_DIFFICULTIES,
        "scale_hypothesis": SCALE_HYPOTHESIS,
    }


INHERENT_DIFFICULTIES: list[dict[str, str]] = [
    {
        "id": "dot_localization",
        "title": "Decimal-point localization",
        "detail": (
            "Finding '.' is a content-based select. With uneven strings, place-value "
            "indices are not positional constants; the model must bind places dynamically."
        ),
    },
    {
        "id": "alignment_padding",
        "title": "Alignment / padding under uneven fractional lengths",
        "detail": (
            "RASP must compute max fractional length and pad — a global compare + scatter. "
            "Errors here surface as misaligned_place failures in the LLM taxonomy."
        ),
    },
    {
        "id": "carry_scan",
        "title": "Sequential carry propagation",
        "detail": (
            "Carry is a scan: column i depends on column i+1. Faithful RASP depth grows with "
            "carry-run length. Fixed-depth Transformers are predicted to fail when "
            "max_carry_run exceeds what induction heads / depth can simulate."
        ),
    },
    {
        "id": "chain_fold",
        "title": "Chain-sum re-alignment",
        "detail": (
            "Folding many addends restarts alignment each time, compounding unevenness and "
            "state-tracking load beyond a single pair addition."
        ),
    },
]

SCALE_HYPOTHESIS: dict[str, str] = {
    "statement": (
        "We implement chain-sum in RASP (a Transformer-native abstraction), derive theoretical "
        "sample difficulty, then check whether models of different scales match that gradient "
        "in relative difficulty and failure-mode distributions. Matching implicates inherent "
        "Transformer computational constraints—not only missing similar training exemplars—"
        "and helps separate architecture effects from training-distribution effects. "
        "This study examines only relative difficulty trends and error-type correlations."
    ),
    "prediction_1": (
        "Relative accuracy should follow uneven_high_carry ≤ aligned ≤ uneven_low_carry "
        "(higher is easier), i.e. many carries ≻ alignment pressure ≻ few carries."
    ),
    "prediction_2": (
        "Error-type mass should correlate with the stressed RASP stage: carry_error with "
        "max_carry_run; misaligned_place with frac_unevenness under low carry."
    ),
    "prediction_3": (
        "If the relative order and error correlations hold from small local models "
        "(e.g. Ollama ~3B) through larger models, the gradient is architecture-linked; "
        "prompt/scale breaks with fixed RASP ranks point toward data or elicitation."
    ),
}


def write_difficulty_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def join_scale_trends(
    difficulty: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Join RASP hardness summaries with per-model aggregate accuracy for plotting."""
    return {
        "hypothesis": SCALE_HYPOTHESIS,
        "rasp_by_stratum": difficulty.get("by_stratum", {}),
        "model_groups": aggregate.get("by_group", []),
        "overall_accuracy": aggregate.get("accuracy"),
        "error_counts": aggregate.get("error_counts", {}),
    }
