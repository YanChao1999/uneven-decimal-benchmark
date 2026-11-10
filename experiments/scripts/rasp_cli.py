#!/usr/bin/env python3
"""Run RASP alignment+carry on a dataset and emit difficulty reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.src.generate import load_jsonl
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
    examples = load_jsonl(args.dataset)
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
    write_difficulty_report(args.output, report)
    print(f"Wrote {args.output}")
    print(
        f"rasp_agreement={report['rasp_agreement']:.3f} "
        f"mean_hardness={report['mean_hardness']:.3f}"
    )
    print("Stratum means:")
    for name, block in report["by_stratum"].items():
        print(
            f"  {name}: hardness={block['mean_hardness']:.3f} "
            f"max_carry_run={block['mean_max_carry_run']:.3f}"
        )


if __name__ == "__main__":
    main()
