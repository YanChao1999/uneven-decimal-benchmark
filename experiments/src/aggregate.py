"""Aggregate accuracy and the two prediction axes (fixed model / fixed cluster)."""

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


def _cluster_of(row: dict[str, Any]) -> str:
    return str(row.get("cluster_id") or row.get("stratum") or "unknown")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy by cluster/stratum × condition × model × temperature + axis views."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (_cluster_of(row), row["condition"], row["model"], row["temperature"])
        groups[key].append(row)

    by_group: list[dict[str, Any]] = []
    for (cluster, condition, model, temperature), items in sorted(groups.items()):
        n = len(items)
        n_correct = sum(1 for r in items if r.get("correct"))
        errors = Counter(r.get("error_type", "unknown") for r in items)
        by_group.append(
            {
                "stratum": cluster,
                "cluster_id": cluster,
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
        "axes": {
            "fixed_model_vary_clusters": _axis_fixed_model(by_group),
            "fixed_cluster_vary_models": _axis_fixed_cluster(by_group),
        },
    }


def _axis_fixed_model(by_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each model (and condition/temp), list cluster accuracies easiest→reported."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for g in by_group:
        buckets[(g["model"], g["condition"], g["temperature"])].append(g)
    out: list[dict[str, Any]] = []
    for (model, condition, temperature), items in sorted(buckets.items()):
        ranked = sorted(items, key=lambda x: (-x["accuracy"], x["cluster_id"]))
        out.append(
            {
                "model": model,
                "condition": condition,
                "temperature": temperature,
                "clusters_by_accuracy_desc": [
                    {
                        "cluster_id": x["cluster_id"],
                        "accuracy": x["accuracy"],
                        "error_counts": x["error_counts"],
                    }
                    for x in ranked
                ],
            }
        )
    return out


def _axis_fixed_cluster(by_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each cluster (and condition/temp), list model accuracies."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for g in by_group:
        buckets[(g["cluster_id"], g["condition"], g["temperature"])].append(g)
    out: list[dict[str, Any]] = []
    for (cluster, condition, temperature), items in sorted(buckets.items()):
        ranked = sorted(items, key=lambda x: (-x["accuracy"], x["model"]))
        out.append(
            {
                "cluster_id": cluster,
                "condition": condition,
                "temperature": temperature,
                "models_by_accuracy_desc": [
                    {
                        "model": x["model"],
                        "accuracy": x["accuracy"],
                        "error_counts": x["error_counts"],
                    }
                    for x in ranked
                ],
            }
        )
    return out


def write_aggregate(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
