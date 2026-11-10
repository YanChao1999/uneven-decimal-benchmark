# uneven-decimal-benchmark

Two **independent** difficulty factors for decimal chain sums:

1. **Alignment (A)** — decimal-point / fractional unevenness (`A0` aligned, `A1` uneven)  
2. **Carry-chain degree (C)** — carry density (`C0` low, `C1` high)

Their product yields sample **clusters** (`A0C0` … `A1C1`). For each cluster we write a
RASP program and estimate **min depth budget** and **min head budget**, then test open models
on two axes:

- **Fixed model → vary clusters**  
- **Fixed clusters → vary model scale**

Scope: relative difficulty trends + error-type correlations (architecture vs training distribution).

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

### Factorial smoke + dry-run

```bash
python -m experiments.scripts.generate_cli \
  --config experiments/configs/dataset_factorial_smoke.yaml

python -m experiments.scripts.rasp_cli \
  --dataset experiments/data/fixtures/factorial_smoke.jsonl

python -m experiments.scripts.eval_cli \
  --config experiments/configs/eval_factorial_dryrun.yaml --dry-run
```

### Ollama axes (after `ollama pull` of small models)

```bash
# Axis 1: one model, all clusters
python -m experiments.scripts.eval_cli \
  --config experiments/configs/eval_fixed_model_vary_clusters.yaml --no-dry-run

# Axis 2: all clusters, several scales
python -m experiments.scripts.eval_cli \
  --config experiments/configs/eval_fixed_clusters_vary_scale.yaml --no-dry-run
```

See README history / earlier sections for Lean (`elan`) and Ollama install notes.

### Lean

```bash
source "$HOME/.elan/env"
python -m experiments.scripts.verify_lean_cli \
  --dataset experiments/data/fixtures/factorial_smoke.jsonl
```

## Clusters

| ID | Alignment | Carry | Typical RASP stress |
|----|-----------|-------|---------------------|
| `A0C0` | aligned | low | baseline |
| `A1C0` | uneven | low | alignment heads/depth |
| `A0C1` | aligned | high | carry depth |
| `A1C1` | uneven | high | both |

## License

Research scaffold; add a license before public release.
