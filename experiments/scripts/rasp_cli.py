#!/usr/bin/env python3
"""Run RASP alignment+carry on a dataset and emit difficulty reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.src.generate import load_jsonl
from experiments.src.paths import repo_root, resolve_repo_path
from experiments.src.rasp.analyze import difficulty_report, write_difficulty_report
from experiments.src.rasp.decimal_add import rasp_chain_sum


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RASP decimal add + difficulty analysis")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/data/fixtures/smoke.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/rasp_difficulty.json"),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify RASP results match expected sums",
    )
    args = parser.parse_args(argv)
    root = repo_root()
    dataset = resolve_repo_path(args.dataset, root=root)
    examples = load_jsonl(dataset)
    if args.check_only:
        bad = []
        for ex in examples:
            got = rasp_chain_sum(ex["addends"]).result
            if got != ex["expected_sum"]:
                bad.append({"id": ex["id"], "got": got, "expected": ex["expected_sum"]})
        print(json.dumps({"n": len(examples), "mismatches": bad}, indent=2))
        if bad:
            raise SystemExit(1)
        return

    report = difficulty_report(examples)
    out = resolve_repo_path(args.output, root=root)
    write_difficulty_report(out, report)
    print(f"Wrote {out}")
    print(
        f"rasp_agreement={report['rasp_agreement']:.3f} "
        f"mean_hardness={report['mean_hardness']:.3f}"
    )
    print("Cluster means + RASP budgets:")
    budgets = {
        c["cluster_id"]: c for c in report.get("rasp_budgets", {}).get("clusters", [])
    }
    for name, block in report.get("by_cluster", report["by_stratum"]).items():
        b = budgets.get(name, {})
        print(
            f"  {name}: hardness={block['mean_hardness']:.3f} "
            f"carry_run={block['mean_max_carry_run']:.3f} "
            f"uneven={block['mean_frac_unevenness']:.3f} "
            f"depth≥{b.get('min_depth_budget', '?')} "
            f"heads≥{b.get('min_head_budget', '?')}"
        )
    order = report.get("rasp_budgets", {}).get("predicted_order_easiest_to_hardest", [])
    if order:
        print("Predicted order (easy→hard):", " < ".join(order))


if __name__ == "__main__":
    main()
