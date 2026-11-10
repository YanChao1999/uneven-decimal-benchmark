#!/usr/bin/env python3
"""Generate JSONL datasets from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.src.generate import iter_dataset, load_yaml, write_jsonl
from experiments.src.paths import repo_root, resolve_repo_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate uneven-decimal chain-sum datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/dataset_default.yaml"),
        help="Dataset YAML config (relative to repo root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output JSONL path",
    )
    args = parser.parse_args(argv)
    root = repo_root()
    cfg_path = resolve_repo_path(args.config, root=root)
    cfg = load_yaml(cfg_path)
    out = resolve_repo_path(
        args.output or Path(cfg.get("output", "experiments/data/dataset_default.jsonl")),
        root=root,
    )
    n = write_jsonl(out, iter_dataset(cfg))
    print(f"Wrote {n} examples to {out}")


if __name__ == "__main__":
    main()
