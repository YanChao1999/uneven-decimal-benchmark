"""Evaluation orchestrator (supports --dry-run with local stub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.src.clients.base import build_client
from experiments.src.evaluate import evaluate_response
from experiments.src.generate import load_jsonl, load_yaml
from experiments.src.prompts import build_messages, condition_expects_partials


def _load_dotenv(root: Path) -> None:
    """Load repo-root ``.env`` into os.environ without overriding existing vars."""
    import os

    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def run_eval(config: dict[str, Any], *, repo_root: Path | None = None) -> Path:
    root = repo_root or Path.cwd()
    _load_dotenv(root)
    if "dataset" not in config or "models" not in config:
        raise KeyError(
            "Eval config must include 'dataset' and 'models'. "
            "If you passed a dataset_*.yaml file, generate data first with "
            "`python -m experiments.scripts.generate_cli --config ...`, "
            "then use an eval_*.yaml (e.g. experiments/configs/eval_factorial_vary_scale.yaml)."
        )
    dataset_path = root / config["dataset"]
    output_dir = root / config.get("output_dir", "experiments/results/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = load_yaml(root / "experiments/configs/models.yaml")["models"]
    examples = load_jsonl(dataset_path)
    conditions = list(config.get("conditions", ["direct"]))
    sampling = dict(config.get("sampling", {}))
    temperatures = list(sampling.get("temperatures", [0.0]))
    n_samples = int(sampling.get("n_samples", 1))
    seed = sampling.get("seed", 0)
    max_tokens = int(sampling.get("max_tokens", 256))
    dry_run = bool(config.get("dry_run", False))

    out_path = output_dir / "raw_results.jsonl"
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for model_name in config["models"]:
            if dry_run:
                client = build_client({"provider": "local_stub", "model": model_name})
            else:
                if model_name not in models_cfg:
                    raise KeyError(f"Model {model_name!r} missing from models.yaml")
                client = build_client(models_cfg[model_name])

            for example in examples:
                for condition in conditions:
                    messages = build_messages(condition, example["addends"])  # type: ignore[arg-type]
                    expect_partials = condition_expects_partials(condition)  # type: ignore[arg-type]
                    for temperature in temperatures:
                        for sample_index in range(n_samples):
                            sample_seed = None if seed is None else int(seed) + n_written
                            response = client.complete(
                                messages,
                                temperature=float(temperature),
                                max_tokens=max_tokens,
                                seed=sample_seed,
                            )
                            result = evaluate_response(
                                example=example,
                                condition=condition,
                                model=model_name,
                                temperature=float(temperature),
                                sample_index=sample_index,
                                raw_text=response.text,
                                expect_partials=expect_partials,
                            )
                            fout.write(json.dumps(result.to_json(), ensure_ascii=True) + "\n")
                            n_written += 1

    meta = {
        "dataset": str(dataset_path),
        "n_results": n_written,
        "dry_run": dry_run,
        "output": str(out_path),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path
