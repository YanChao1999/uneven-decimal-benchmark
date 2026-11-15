"""Per-cluster RASP programs → minimum depth / head budgets.

Each sample cluster (alignment_level × carry_level) has a specialized RASP sketch.
We report *lower-bound style* resource budgets for a faithful implementation:

- **min_depth_budget**: layers needed for sequential dependencies (esp. carry scan)
  plus alignment stages when fractional lengths are uneven.
- **min_head_budget**: distinct parallel select patterns required in the widest layer
  (dot localization, length/place binding, digit zipmap, predecessor carry).

These are theoretical predictions for Transformer-native cost, not fitted scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.src.rasp.decimal_add import rasp_chain_sum


@dataclass(frozen=True)
class ClusterRaspProgram:
    cluster_id: str
    alignment_level: int
    carry_level: int
    stages: list[str]
    min_depth_budget: int
    min_head_budget: int
    mean_max_carry_run: float
    mean_frac_unevenness: float
    n_examples: int
    notes: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _alignment_depth(level: int) -> int:
    # A0: places already aligned → skip dynamic padding (0).
    # A1: find '.' + compute max frac + broadcast pad (≥2 select/map layers).
    return 0 if level <= 0 else 2


def _carry_depth(level: int, mean_max_carry_run: float) -> int:
    # Faithful LSB→MSB scan: depth grows with carry-run length.
    # Low-carry clusters still need ≥1 digitwise+carry layer.
    run = max(1, int(round(mean_max_carry_run)))
    if level <= 0:
        return min(run, 2)  # short runs expected
    return max(run, 3)


def _digitwise_depth() -> int:
    return 1


def _head_budget(alignment_level: int, carry_level: int) -> int:
    # Parallel selects that cannot share one head pattern in the same layer:
    heads = 1  # digitwise zipmap / place values
    if alignment_level >= 1:
        heads += 2  # dot content-select + frac-length compare/gather
    if carry_level >= 1:
        heads += 1  # predecessor-column select for carry (long-run stress)
    else:
        heads += 1  # still need a carry/pred head, but lighter
    return heads


def cluster_rasp_program(
    cluster_id: str,
    examples: list[dict[str, Any]],
) -> ClusterRaspProgram:
    if not examples:
        raise ValueError(f"No examples for cluster {cluster_id}")
    a_lvl = int(examples[0].get("alignment_level", 0))
    c_lvl = int(examples[0].get("carry_level", 0))
    runs: list[float] = []
    uneven: list[float] = []
    for ex in examples:
        trace = rasp_chain_sum(ex["addends"])
        runs.append(float(trace.max_carry_run))
        fr = ex.get("frac_lens") or []
        if fr:
            uneven.append(float(max(fr) - min(fr)))
        else:
            uneven.append(float(trace.frac_width))  # fallback
    mean_run = sum(runs) / len(runs)
    mean_uneven = sum(uneven) / len(uneven)

    stages: list[str] = []
    if a_lvl >= 1:
        stages.extend(
            [
                "select_dot",
                "aggregate_frac_len",
                "pad_to_common_width",
            ]
        )
    else:
        stages.append("skip_align_constant_width")
    stages.append("digitwise_zipmap_add")
    stages.append("carry_scan_lsb_to_msb")
    if len(examples[0].get("addends", [])) > 2:
        stages.append("chain_fold_realign")

    depth = _alignment_depth(a_lvl) + _digitwise_depth() + _carry_depth(c_lvl, mean_run)
    # Extra fold cost: each additional addend may restart align under A1.
    n_add = int(examples[0].get("n_addends", 2))
    if n_add > 2 and a_lvl >= 1:
        depth += n_add - 2
    heads = _head_budget(a_lvl, c_lvl)

    notes = (
        f"A{a_lvl}×C{c_lvl}: alignment_depth={_alignment_depth(a_lvl)}, "
        f"carry_depth={_carry_depth(c_lvl, mean_run)}, "
        f"digitwise=1; heads={heads}. "
        "Depth lower-bounds sequential carry; heads count distinct select patterns."
    )
    return ClusterRaspProgram(
        cluster_id=cluster_id,
        alignment_level=a_lvl,
        carry_level=c_lvl,
        stages=stages,
        min_depth_budget=depth,
        min_head_budget=heads,
        mean_max_carry_run=mean_run,
        mean_frac_unevenness=mean_uneven,
        n_examples=len(examples),
        notes=notes,
    )


def budget_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        cid = ex.get("cluster_id") or ex.get("stratum") or "unknown"
        by_cluster.setdefault(str(cid), []).append(ex)

    programs = [cluster_rasp_program(cid, rows) for cid, rows in sorted(by_cluster.items())]
    # Predicted hardness order by (depth, heads, carry_run).
    ranked = sorted(
        programs,
        key=lambda p: (p.min_depth_budget, p.min_head_budget, p.mean_max_carry_run),
    )
    return {
        "factor_axes": {
            "alignment": "decimal-point alignment / fractional unevenness (A0 aligned, A1 uneven)",
            "carry": "carry-chain degree (C0 low, C1 high)",
        },
        "clusters": [p.to_json() for p in programs],
        "predicted_order_easiest_to_hardest": [p.cluster_id for p in ranked],
        "prediction": {
            "fixed_model_vary_clusters": (
                "For one model, accuracy should fall and carry/misalign errors should rise "
                "as min_depth_budget / min_head_budget increase across clusters."
            ),
            "fixed_clusters_vary_scale": (
                "For one cluster, larger models may raise absolute accuracy, but relative "
                "ordering across clusters should remain RASP-consistent if architecture "
                "constraints dominate training-distribution gaps."
            ),
        },
    }
