"""RASP difficulty proxies, cluster budgets, and prediction axes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.src.decimal_utils import column_carry_count, frac_len, normalize_decimal_str
from experiments.src.rasp.budgets import budget_report
from experiments.src.rasp.decimal_add import rasp_chain_sum


@dataclass(frozen=True)
class InstanceDifficulty:
    example_id: str
    stratum: str
    cluster_id: str
    alignment_level: int
    carry_level: int
    n_addends: int
    max_frac: int
    frac_unevenness: int
    width_after_align: int
    carry_count: int
    rasp_n_carry_steps: int
    rasp_max_carry_run: int
    rasp_result: str
    expected_sum: str
    rasp_matches_expected: bool
    rasp_hardness: float


def _hardness(
    *,
    n_addends: int,
    frac_unevenness: int,
    width: int,
    max_carry_run: int,
    carry_count: int,
) -> float:
    return (
        0.35 * max_carry_run
        + 0.25 * carry_count
        + 0.20 * frac_unevenness
        + 0.10 * width
        + 0.10 * max(0, n_addends - 2)
    )


def analyze_instance(example: dict[str, Any]) -> InstanceDifficulty:
    addends = list(example["addends"])
    frac_lens = [frac_len(a) for a in addends]
    max_f = max(frac_lens) if frac_lens else 0
    min_f = min(frac_lens) if frac_lens else 0
    trace = rasp_chain_sum(addends)
    expected = example["expected_sum"]
    matches = normalize_decimal_str(trace.result) == normalize_decimal_str(expected)
    cluster_id = str(example.get("cluster_id") or example.get("stratum") or "unknown")
    return InstanceDifficulty(
        example_id=example["id"],
        stratum=str(example.get("stratum", cluster_id)),
        cluster_id=cluster_id,
        alignment_level=int(example.get("alignment_level", 0)),
        carry_level=int(example.get("carry_level", 0)),
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
    by_cluster: dict[str, list[InstanceDifficulty]] = {}
    for row in rows:
        by_cluster.setdefault(row.cluster_id, []).append(row)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    cluster_summary = {
        name: {
            "n": len(items),
            "alignment_level": items[0].alignment_level,
            "carry_level": items[0].carry_level,
            "mean_hardness": avg([x.rasp_hardness for x in items]),
            "mean_max_carry_run": avg([float(x.rasp_max_carry_run) for x in items]),
            "mean_frac_unevenness": avg([float(x.frac_unevenness) for x in items]),
            "rasp_agreement": avg([1.0 if x.rasp_matches_expected else 0.0 for x in items]),
        }
        for name, items in sorted(by_cluster.items())
    }
    budgets = budget_report(examples)
    return {
        "n": len(rows),
        "rasp_agreement": avg([1.0 if r.rasp_matches_expected else 0.0 for r in rows]),
        "mean_hardness": avg([r.rasp_hardness for r in rows]),
        "by_stratum": cluster_summary,  # backward-compatible key
        "by_cluster": cluster_summary,
        "rasp_budgets": budgets,
        "instances": [asdict(r) for r in rows],
        "inherent_difficulties": INHERENT_DIFFICULTIES,
        "scale_hypothesis": SCALE_HYPOTHESIS,
    }


INHERENT_DIFFICULTIES: list[dict[str, str]] = [
    {
        "id": "alignment_factor",
        "title": "Decimal-point alignment factor (A)",
        "detail": (
            "Independent factor: uneven fractional lengths force dynamic place binding "
            "(dot select, max-frac, pad). Orthogonal to carry density."
        ),
    },
    {
        "id": "carry_factor",
        "title": "Carry-chain degree factor (C)",
        "detail": (
            "Independent factor: long carry runs impose sequential depth in RASP. "
            "Orthogonal to whether fractions are aligned."
        ),
    },
    {
        "id": "depth_head_budgets",
        "title": "Min depth / head budgets per cluster",
        "detail": (
            "Each A×C cluster gets its own RASP sketch; we report min_depth_budget and "
            "min_head_budget as Transformer-native cost predictions."
        ),
    },
]

SCALE_HYPOTHESIS: dict[str, str] = {
    "statement": (
        "Difficulty factors split into alignment (A) and carry-chain degree (C). "
        "Independent A×C clusters each have a RASP program with predicted min depth/head "
        "budgets. We test two axes: (1) fixed model, vary clusters; (2) fixed clusters, "
        "vary model scale. Relative accuracy order and error-type correlations should "
        "track those budgets if Transformer constraints dominate missing similar training data."
    ),
    "prediction_1": (
        "Fixed model: accuracy decreases as cluster min_depth_budget / min_head_budget rise "
        "(typically A0C0 easiest, A1C1 hardest)."
    ),
    "prediction_2": (
        "Fixed clusters: larger models may lift absolute accuracy, but the relative order "
        "across A×C clusters should remain RASP-consistent."
    ),
    "prediction_3": (
        "Error types: misaligned_place mass tracks A; carry_error mass tracks C."
    ),
}


def write_difficulty_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def join_scale_trends(
    difficulty: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hypothesis": SCALE_HYPOTHESIS,
        "rasp_by_cluster": difficulty.get("by_cluster", difficulty.get("by_stratum", {})),
        "rasp_budgets": difficulty.get("rasp_budgets", {}),
        "model_groups": aggregate.get("by_group", []),
        "axes": aggregate.get("axes", {}),
        "overall_accuracy": aggregate.get("accuracy"),
        "error_counts": aggregate.get("error_counts", {}),
    }
