"""Evaluation orchestrator (supports --dry-run with local stub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.src.clients.base import build_client
from experiments.src.evaluate import evaluate_response
from experiments.src.generate import load_jsonl, load_yaml
from experiments.src.prompts import build_messages, condition_expects_partials


def run_eval(config: dict[str, Any], *, repo_root: Path | None = None) -> Path:
    root = repo_root or Path.cwd()
    dataset_path = root / config["dataset"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = load_yaml(root / "experiments/configs/models.yaml")["models"]
    examples = load_jsonl(dataset_path)
    conditions = list(config["conditions"])
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
