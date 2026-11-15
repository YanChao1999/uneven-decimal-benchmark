"""Aggregate accuracy and the two prediction axes (fixed model / fixed cluster)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_FCN_RE = re.compile(
    r"^(?:A(?P<a>\d+))?F(?P<f>\d+)C(?P<c>\d+)N(?P<n>\d+)$"
)


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


def parse_fcn_cluster(cluster_id: str) -> dict[str, int] | None:
    m = _FCN_RE.match(str(cluster_id))
    if not m:
        return None
    out = {"F": int(m.group("f")), "C": int(m.group("c")), "N": int(m.group("n"))}
    if m.group("a") is not None:
        out["A"] = int(m.group("a"))
    return out


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
        "trend_matrices": trend_matrices(by_group),
    }


def trend_matrices(by_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """F×N accuracy grids per model × C (for reading monotonic difficulty trends)."""
    # model, condition, temp, C -> {(F,N): (correct, n)}
    buckets: dict[tuple[Any, ...], dict[tuple[int, int], list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for g in by_group:
        parsed = parse_fcn_cluster(g["cluster_id"])
        if not parsed:
            continue
        key = (g["model"], g["condition"], g["temperature"], parsed["C"])
        cell = buckets[key][(parsed["F"], parsed["N"])]
        cell[0] += int(g["n_correct"])
        cell[1] += int(g["n"])

    out: list[dict[str, Any]] = []
    for (model, condition, temperature, c_lvl), cells in sorted(buckets.items()):
        f_vals = sorted({f for f, _ in cells})
        n_vals = sorted({n for _, n in cells})
        grid: list[dict[str, Any]] = []
        for f in f_vals:
            row = {"F": f, "by_N": {}}
            for n in n_vals:
                ok, tot = cells.get((f, n), [0, 0])
                row["by_N"][str(n)] = {
                    "n": tot,
                    "n_correct": ok,
                    "accuracy": (ok / tot) if tot else None,
                }
            grid.append(row)
        # Marginal means along each axis (equal weight over available cells).
        by_f: dict[str, float] = {}
        for f in f_vals:
            accs = [
                cells[(f, n)][0] / cells[(f, n)][1]
                for n in n_vals
                if cells.get((f, n), [0, 0])[1]
            ]
            if accs:
                by_f[str(f)] = sum(accs) / len(accs)
        by_n: dict[str, float] = {}
        for n in n_vals:
            accs = [
                cells[(f, n)][0] / cells[(f, n)][1]
                for f in f_vals
                if cells.get((f, n), [0, 0])[1]
            ]
            if accs:
                by_n[str(n)] = sum(accs) / len(accs)
        out.append(
            {
                "model": model,
                "condition": condition,
                "temperature": temperature,
                "C": c_lvl,
                "F_levels": f_vals,
                "N_levels": n_vals,
                "grid_F_by_N": grid,
                "marginal_accuracy_by_F": by_f,
                "marginal_accuracy_by_N": by_n,
            }
        )
    return out


def format_trend_tables(summary: dict[str, Any]) -> str:
    """Human-readable F×N accuracy tables from aggregate trend_matrices."""
    lines: list[str] = []
    for block in summary.get("trend_matrices") or []:
        lines.append(
            f"## {block['model']}  C={block['C']}  "
            f"(condition={block['condition']}, T={block['temperature']})"
        )
        n_levels = block["N_levels"]
        header = "F\\N | " + " ".join(f"{n:>6}" for n in n_levels)
        lines.append(header)
        lines.append("-" * len(header))
        for row in block["grid_F_by_N"]:
            cells = []
            for n in n_levels:
                cell = row["by_N"].get(str(n), {})
                acc = cell.get("accuracy")
                cells.append(f"{acc:6.2f}" if acc is not None else f"{'—':>6}")
            lines.append(f"F{row['F']:<2} | " + " ".join(cells))
        marg_f = block.get("marginal_accuracy_by_F") or {}
        marg_n = block.get("marginal_accuracy_by_N") or {}
        if marg_f:
            lines.append(
                "marg F: "
                + " ".join(f"F{k}={v:.2f}" for k, v in sorted(marg_f.items(), key=lambda x: int(x[0])))
            )
        if marg_n:
            lines.append(
                "marg N: "
                + " ".join(f"N{k}={v:.2f}" for k, v in sorted(marg_n.items(), key=lambda x: int(x[0])))
            )
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


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
