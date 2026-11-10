"""Aggregate accuracy and error-type distributions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy by stratum × condition × model × temperature + error histograms."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["stratum"], row["condition"], row["model"], row["temperature"])
        groups[key].append(row)

    by_group: list[dict[str, Any]] = []
    for (stratum, condition, model, temperature), items in sorted(groups.items()):
        n = len(items)
        n_correct = sum(1 for r in items if r.get("correct"))
        errors = Counter(r.get("error_type", "unknown") for r in items)
        by_group.append(
            {
                "stratum": stratum,
                "condition": condition,
                "model": model,
                "temperature": temperature,
                "n": n,
                "n_correct": n_correct,
                "accuracy": (n_correct / n) if n else 0.0,
                "error_counts": dict(errors),
            }
        )

    overall_n = len(rows)
    overall_correct = sum(1 for r in rows if r.get("correct"))
    return {
        "n_rows": overall_n,
        "n_correct": overall_correct,
        "accuracy": (overall_correct / overall_n) if overall_n else 0.0,
        "error_counts": dict(Counter(r.get("error_type", "unknown") for r in rows)),
        "by_group": by_group,
    }


def write_aggregate(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
