#!/usr/bin/env python3
"""Emit Lean ground-truth checks and optionally run `lake build`."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.src.paths import repo_root, resolve_repo_path
from experiments.src.verify_lean import verify_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify dataset ground truth with Lean")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/data/fixtures/smoke.jsonl"),
    )
    parser.add_argument(
        "--lean-dir",
        type=Path,
        default=Path("experiments/lean"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to emit")
    parser.add_argument(
        "--no-lake",
        action="store_true",
        help="Only emit Generated.lean; do not run lake",
    )
    args = parser.parse_args(argv)
    root = repo_root()

    run_lake = not args.no_lake
    if run_lake and shutil.which("lake") is None:
        print("lake not found on PATH; emitting Generated.lean only.")
        run_lake = False

    result = verify_dataset(
        resolve_repo_path(args.dataset, root=root),
        resolve_repo_path(args.lean_dir, root=root),
        limit=args.limit,
        run_lake=run_lake,
    )
    print(json.dumps({k: result[k] for k in result if k not in {"stdout", "stderr"}}, indent=2))
    if result.get("lake_ran"):
        if result.get("stdout"):
            print(result["stdout"])
        if result.get("stderr"):
            print(result["stderr"])
        if not result.get("lake_ok"):
            raise SystemExit(result.get("returncode", 1))


if __name__ == "__main__":
    main()
