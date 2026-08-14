# Instruction-Tuned Code Generation

LoRA fine-tuning for code models, with a packed-sequence DataLoader and an eval harness (HumanEval plus correctness / efficiency / safety checks).

## Overview

- Base model (full run): Llama-2-7B
- Data: 15k CodeAlpaca-style instruction examples
- LoRA: rank 16, alpha 32
- Training: sequence packing, bf16 or fp16, gradient checkpointing
- Full target box: one A100 40GB, seq length 2048
- Home GPU path: RTX 3080 10GB (`configs/rtx3080.yaml`, TinyLlama fp16 LoRA)
- Eval: HumanEval plus a 500-example harness

| Config | Use when |
|---|---|
| `configs/a100.yaml` | A100 / 40GB+, Llama-2-7B, full bf16 LoRA |
| `configs/rtx3080.yaml` | RTX 3080 (or similar), TinyLlama fp16 LoRA |
| `configs/rtx3080_llama2.yaml` | RTX 3080, Llama-2-7B QLoRA (needs bitsandbytes + HF access) |
| `configs/smoke.yaml` | CPU / quick wiring check |

## Setup

```bash
python -m pip install -r requirements.txt
```

CUDA torch (home GPU). Important: a normal `pip install torch` often gives a CPU wheel,
which is why you may see `CUDA unavailable; falling back to fp32` even with an RTX 3080.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

You want something like `2.x.x+cu124` and `True` for `is_available()`.

`configs/rtx3080.yaml` (TinyLlama) does **not** need bitsandbytes. Only the Llama-2 QLoRA
config does (`configs/rtx3080_llama2.yaml`). On Windows, bitsandbytes is unreliable; use
WSL2 if you go that route: `pip install -U "bitsandbytes>=0.46.1"`.

Llama-2 runs also need HF access to `meta-llama/Llama-2-7b-hf` and `huggingface-cli login` (or `HF_TOKEN`).

## Smoke (CPU / tiny model)

```bash
python scripts/make_offline_smoke_data.py
# or: python scripts/run.py prepare-data --output data/processed/codealpaca_smoke.jsonl --n 64
python scripts/run.py benchmark-packing --config configs/smoke.yaml
python scripts/run.py train --config configs/smoke.yaml
python scripts/run.py eval --config configs/smoke.yaml --output results/smoke --humaneval-n 10 --harness-n 20
```

## RTX 3080 home run

```bash
python scripts/run.py prepare-data --output data/processed/codealpaca_15k.jsonl --n 15000
python scripts/run.py benchmark-packing --config configs/rtx3080.yaml
python scripts/run.py train --config configs/rtx3080.yaml
python scripts/run.py eval --config configs/rtx3080.yaml --output results/rtx3080 --humaneval-n 164 --harness-n 500
```

Base model (no adapter), then a one-file summary:

```bash
python scripts/run.py eval --config configs/rtx3080.yaml --output results/rtx3080_base --humaneval-n 164 --harness-n 500 --no-adapter
python scripts/write_summary.py --results-dir results/rtx3080 --base-dir results/rtx3080_base --hardware "RTX 3080 10GB" --config configs/rtx3080.yaml
python scripts/compare_humaneval.py --base results/rtx3080_base/humaneval.json --ft results/rtx3080/humaneval.json
```

If you have Llama-2 access and want the 7B QLoRA variant:

```bash
python scripts/run.py train --config configs/rtx3080_llama2.yaml
python scripts/run.py eval --config configs/rtx3080_llama2.yaml --output results/rtx3080_llama2 --humaneval-n 164 --harness-n 500
```

OOM tips: set `per_device_train_batch_size: 1`, drop `max_seq_length` to 512, or raise `gradient_accumulation_steps`.

## Full A100 run

```bash
python scripts/run.py prepare-data --output data/processed/codealpaca_15k.jsonl --n 15000
python scripts/run.py benchmark-packing --config configs/a100.yaml
python scripts/run.py train --config configs/a100.yaml
python scripts/run.py eval --config configs/a100.yaml --output results/a100 --humaneval-n 164 --harness-n 500
```

Compare base vs fine-tuned HumanEval:

```bash
python scripts/compare_humaneval.py --base results/a100/base_humaneval.json --ft results/a100/humaneval.json
```

### Target numbers (A100 / full recipe)

Write measured values under `results/a100/`. Smoke and TinyLlama runs are not Llama-2 HumanEval scores.

| Metric | Target |
|---|---|
| HumanEval vs base | about +23% relative |
| GPU util, naive pad to packed | 67% to 94% |
| Train time cut from packing | about 38% |
| Harness size | 500 cases |

Relative HumanEval gain:

```text
(pass@1_ft - pass@1_base) / pass@1_base
```

## Results (measured)

### RTX 3080 / TinyLlama (`configs/rtx3080.yaml`)

Artifacts: `results/rtx3080/` (fine-tuned) and `results/rtx3080_base/` (no adapter).
Model: TinyLlama-1.1B + LoRA (r=16, alpha=32), fp16, seq 1024, 15k CodeAlpaca examples, 220 steps (~16 min).

| Metric | Base | Fine-tuned |
|---|---|---|
| Naive pad util | | 10.8% |
| Packed util | | 92.3% |
| Packed sequences | | 1764 from 15000 examples |
| HumanEval pass@1 | 0/164 (0%) | 3/164 (~1.83%) |
| Harness correctness | 0% (0/450) | 2% (9/450) |
| Harness efficiency | 50% (1/2) | 0% (0/2) |
| Harness safety | ~95.8% (46/48) | ~68.8% (33/48) |

HumanEval absolute delta: +3/164 (~1.83 pp). Relative gain is undefined because base pass@1 is 0.
Packing is the clear win on this box. Absolute HumanEval is still low for TinyLlama-1.1B; that is not the Llama-2 A100 target.
Safety fell after fine-tuning on this harness; treat that as a real tradeoff of this home-GPU run, not noise to ignore.

### RTX 3080 / Llama-2-7B QLoRA (`configs/rtx3080_llama2.yaml`)

Artifacts: `results/rtx3080_llama2/` and `results/rtx3080_llama2_base/`.
Model: Llama-2-7B QLoRA (r=16, alpha=32), 4-bit NF4 eval, seq 1024, 15k CodeAlpaca, 220 steps (~90 min). Train log had intermittent `loss=nan`.

| Metric | Base | Fine-tuned |
|---|---|---|
| Naive pad util | | 10.8% |
| Packed util | | 92.3% |
| HumanEval pass@1 (4-bit eval) | 0/164 (0%) | 3/164 (~1.83%) |

HumanEval absolute delta: +3/164 (~1.83 pp). Relative gain undefined at base 0.
This is not the A100 bf16 recipe and not a literature Llama-2 HumanEval replicate (published base is often ~12%). Packing util is again the strong measured win. Harness not re-run for this config.

### A100 / Llama-2-7B

Not measured in-repo yet. Targets live in `results/TARGETS.md`.

## Layout

```text
configs/                 a100, rtx3080, smoke
src/instruction_codegen/
  data/                  dataset prep + packing
  training/              LoRA / QLoRA SFT
  eval/                  HumanEval + harness + safety checks
scripts/run.py           prepare-data | train | eval | benchmark-packing
scripts/write_summary.py fold train/eval JSON into SUMMARY.json
tests/
results/
```

## Sequence packing

Without packing, each example is right-padded to `max_seq_length`, so a lot of GPU time goes to pad tokens. The packer concatenates short examples into the same window (EOS between them), which improves token utilization and usually means fewer optimizer steps per epoch.

```bash
python scripts/run.py benchmark-packing --config configs/a100.yaml
```

## Tests

```bash
python -m pytest tests/ -q
```

## Notes

- Smoke runs only check that the code path works (`results/smoke/`).
- Home-GPU measured compare: `results/rtx3080/` vs `results/rtx3080_base/`. Packing util is the strong result; HumanEval absolute +3/164 with base at 0; safety dropped after fine-tune on this harness.
- TinyLlama numbers are not the Llama-2 A100 targets.
- Full Llama-2 bf16 numbers need A100-class VRAM (or cloud) plus HF access.
