# uneven-decimal-benchmark

RASP-predicted difficulty for uneven decimal **chain sums**.

We implement the summation procedure as a RASP program (align → digitwise add → carry scan),
obtain **theoretical sample difficulty**, then check whether models of different scales show
**relative difficulty rankings** and **failure-mode distributions** that match this
Transformer-native abstraction.

All prompts, paper text, and docs are in **English**.

## Claim (scope)

- Failures on this task need not come only from missing similar training samples; they can
  also reflect **inherent Transformer computational constraints** that RASP idealizes.
- If real models’ error trends track the RASP difficulty gradient, that supports an
  architecture-linked account and helps separate **architecture** vs **training distribution**.
- This study examines **only**:
  1. relative difficulty trends  
  2. correlation of error-type distributions with RASP proxies  

Predicted ladder: **high-carry uneven ≻ alignment / place structure ≻ low-carry**.

## Layout

```
paper/                 LaTeX paper skeleton
experiments/
  lean/                Lean 4 ground-truth checks (Lake)
  src/                 Python generators, prompts, eval, RASP
  configs/             YAML for datasets, models, eval
  data/fixtures/       Tiny committed JSONL
  results/             Eval outputs (mostly gitignored)
  scripts/             CLI entrypoints
```

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env   # cloud APIs only
```

### Lean 4 + Lake (via elan)

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source "$HOME/.elan/env"
lean --version && lake --version
cd experiments/lean && lake build
```

### Local Ollama pilot (~3B)

```bash
ollama serve
ollama pull qwen2.5:3b
python -m experiments.scripts.eval_cli \
  --config experiments/configs/eval_ollama_smoke.yaml \
  --no-dry-run
```

## Smoke path

```bash
python -m experiments.scripts.generate_cli --config experiments/configs/dataset_smoke.yaml
python -m experiments.scripts.rasp_cli --dataset experiments/data/fixtures/smoke.jsonl --check-only
python -m experiments.scripts.eval_cli --config experiments/configs/eval_smoke.yaml --dry-run
```

## Dataset strata

| ID | RASP role |
|----|-----------|
| `uneven_high_carry` | Hardest — dense carries + uneven fractions |
| `aligned` | Middle — alignment cheap; carry remains |
| `uneven_low_carry` | Easiest stress — uneven but sparse carries |

## License

Research scaffold; add a license before public release.
