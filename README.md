# uneven-decimal-benchmark

Reproducible scaffold for studying whether language models learn **explicit decimal-point alignment and carry**, or only approximate chain sums distributionally.

All prompts, paper text, and docs are in **English**.

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

## Hypothesis

1. LLMs may lack a stable **alignment + carry** algorithm for uneven decimals.
2. A **RASP** program for align → digitwise add → carry scan exposes inherent bottlenecks (especially long carry runs).
3. On a **fixed dataset**, accuracy trends and error-type shifts across **model scales** should be predictable from RASP difficulty proxies (`max_carry_run`, `frac_unevenness`, `rasp_hardness`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional: Lean 4 + Lake for formal label checks
# Optional API keys: copy .env.example → .env
```

## Smoke path (no API spend)

```bash
# 1) Generate tiny fixture dataset
python -m experiments.scripts.generate_cli --config experiments/configs/dataset_smoke.yaml

# 2) Emit Lean checks (runs lake build if available)
python -m experiments.scripts.verify_lean_cli --dataset experiments/data/fixtures/smoke.jsonl

# 3) RASP program + difficulty report
python -m experiments.scripts.rasp_cli --dataset experiments/data/fixtures/smoke.jsonl

# 4) Dry-run eval (local stub)
python -m experiments.scripts.eval_cli --config experiments/configs/eval_smoke.yaml --dry-run

# 5) Multi-scale dry-run + RASP×scale join
python -m experiments.scripts.eval_cli --config experiments/configs/eval_scale_smoke.yaml --dry-run
```

## Dataset strata

| ID | Description |
|----|-------------|
| `aligned` | Equal fractional lengths |
| `uneven_low_carry` | Uneven fractions, carry-sparse |
| `uneven_high_carry` | Uneven fractions, carry-dense |

## Prompt conditions

| ID | Behavior |
|----|----------|
| `direct` | Final answer only; no tools |
| `scratch_partial` | Mandatory running partial sums |
| `align_hint` | Final answer + alignment/carry reminder |
| `scratch_align` | Partials + alignment/carry reminder |

## RASP module

`experiments/src/rasp/` implements an executable RASP-style sketch:

1. locate / align on the decimal point  
2. digitwise column sum (`zipmap`)  
3. sequential carry scan (the depth bottleneck)

`udb-rasp` / `rasp_cli` writes `rasp_difficulty.json` with per-example hardness and the scale-trend hypothesis text used in the paper.

## Models

- OpenAI-compatible: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- Hugging Face open models (e.g. Qwen): `HF_TOKEN`, `HF_BASE_URL`
- Scale roster: `experiments/configs/models_scale.yaml`
- Dry-run always uses `local_stub` (no network)

## Paper

```bash
cd paper && make
```

## License

Research scaffold; add a license before public release.
