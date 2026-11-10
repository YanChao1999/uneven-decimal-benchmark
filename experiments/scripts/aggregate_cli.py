#!/usr/bin/env python3
"""Aggregate raw eval JSONL into accuracy / error-type summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.src.aggregate import aggregate, load_results_jsonl, write_aggregate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="raw_results.jsonl path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="aggregate.json path (default: alongside input)",
    )
    args = parser.parse_args(argv)
    rows = load_results_jsonl(args.input)
    summary = aggregate(rows)
    out = args.output or args.input.with_name("aggregate.json")
    write_aggregate(out, summary)
    print(f"Wrote {out}")
    print(f"accuracy={summary['accuracy']:.3f} n={summary['n_rows']}")


if __name__ == "__main__":
    main()
