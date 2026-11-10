#!/usr/bin/env python3
"""Run model evaluation (dry-run by default via smoke config)."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.src.aggregate import aggregate, load_results_jsonl, write_aggregate
from experiments.src.generate import load_jsonl, load_yaml
from experiments.src.rasp.analyze import (
    difficulty_report,
    join_scale_trends,
    write_difficulty_report,
)
from experiments.src.run_eval import run_eval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on uneven-decimal chain sums")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/eval_smoke.yaml"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force local stub client (no API calls)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Allow real API clients from models.yaml",
    )
    args = parser.parse_args(argv)
    root = Path.cwd()
    cfg = load_yaml(args.config)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.no_dry_run:
        cfg["dry_run"] = False

    out = run_eval(cfg, repo_root=root)
    summary = aggregate(load_results_jsonl(out))
    agg_path = root / cfg["output_dir"] / "aggregate.json"
    write_aggregate(agg_path, summary)
    print(f"Wrote raw results to {out}")
    print(f"Wrote aggregate to {agg_path}")
    print(f"Overall accuracy: {summary['accuracy']:.3f} ({summary['n_correct']}/{summary['n_rows']})")

    # Optional RASP difficulty + scale-trend join (used by eval_scale_smoke.yaml).
    if cfg.get("rasp_report") or cfg.get("scale_join"):
        examples = load_jsonl(root / cfg["dataset"])
        rasp = difficulty_report(examples)
        if cfg.get("rasp_report"):
            rasp_path = root / cfg["rasp_report"]
            write_difficulty_report(rasp_path, rasp)
            print(f"Wrote RASP difficulty report to {rasp_path}")
        if cfg.get("scale_join"):
            joined = join_scale_trends(rasp, summary)
            join_path = root / cfg["scale_join"]
            join_path.parent.mkdir(parents=True, exist_ok=True)
            join_path.write_text(
                __import__("json").dumps(joined, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote RASP×scale join to {join_path}")


if __name__ == "__main__":
    main()
