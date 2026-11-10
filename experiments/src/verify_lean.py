"""Emit Lean check statements from JSONL and optionally invoke Lake."""

from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any


def decimal_to_rat_parts(s: str) -> tuple[str, int]:
    """Convert a decimal string to (integer numerator as string, scale k) meaning n / 10^k."""
    d = Decimal(s)
    # Use as_tuple for exact digit reconstruction.
    sign, digits, exp = d.as_tuple()
    digits_int = int("".join(str(x) for x in digits) or "0")
    if sign:
        digits_int = -digits_int
    if exp >= 0:
        numer = digits_int * (10**exp)
        scale = 0
    else:
        numer = digits_int
        scale = -exp
    return str(numer), scale


def lean_int_literal(n: str) -> str:
    """Render a Lean integer literal (handles negatives)."""
    if n.startswith("-"):
        return f"(-{n[1:]})"
    return n


def example_to_lean_theorem(ex: dict[str, Any], idx: int) -> str:
    """Generate a `#eval` / `example` block checking chain sum equals expected."""
    addends = ex["addends"]
    expected = ex["expected_sum"]
    parts = []
    for a in addends:
        numer, scale = decimal_to_rat_parts(a)
        parts.append(f"(mkDec {lean_int_literal(numer)} {scale})")
    exp_n, exp_s = decimal_to_rat_parts(expected)
    addends_lean = ", ".join(parts)
    name = ex["id"].replace("-", "_")
    return (
        f"/-- Auto-generated check for {ex['id']} ({ex['stratum']}) -/\n"
        f"example : decEq (chainSum [{addends_lean}]) "
        f"(mkDec {lean_int_literal(exp_n)} {exp_s}) = true := by native_decide\n"
    )


def emit_generated_lean(dataset_path: Path, out_path: Path, *, limit: int | None = None) -> int:
    rows: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break

    header = (
        "import UnevenDecimal.Decimal\n\n"
        "namespace UnevenDecimal.Generated\n"
        "open UnevenDecimal\n\n"
        "/-! Auto-generated ground-truth checks. Do not edit by hand. -/\n\n"
    )
    body = "\n".join(example_to_lean_theorem(ex, i) for i, ex in enumerate(rows))
    footer = "\nend UnevenDecimal.Generated\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body + footer, encoding="utf-8")
    return len(rows)


def lake_build(lean_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["lake", "build"],
        cwd=str(lean_dir),
        text=True,
        capture_output=True,
        check=False,
    )


def verify_dataset(
    dataset_path: Path,
    lean_dir: Path,
    *,
    generated_rel: str = "UnevenDecimal/Generated.lean",
    limit: int | None = None,
    run_lake: bool = True,
) -> dict[str, Any]:
    out_path = lean_dir / generated_rel
    n = emit_generated_lean(dataset_path, out_path, limit=limit)
    result: dict[str, Any] = {
        "dataset": str(dataset_path),
        "generated": str(out_path),
        "n_examples": n,
        "lake_ran": False,
        "lake_ok": None,
        "stdout": "",
        "stderr": "",
    }
    if not run_lake:
        return result
    proc = lake_build(lean_dir)
    result["lake_ran"] = True
    result["lake_ok"] = proc.returncode == 0
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    result["returncode"] = proc.returncode
    return result
