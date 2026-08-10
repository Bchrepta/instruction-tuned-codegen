# Metric targets

## A100 / Llama-2-7B (`configs/a100.yaml`)

Fill measured columns after a full run. Write artifacts under `results/a100/`.

| Metric | Target | Measured | How measured |
|---|---|---|---|
| Relative HumanEval gain vs base | about +23% | | `(ft - base) / base` on pass@1 |
| GPU util, naive pad to packed | 67% to 94% | | `benchmark-packing` + trainer logs |
| Train time cut from packing | about 38% | | wall-clock vs naive pad |
| Harness size | 500 | | correctness / efficiency / safety |

## RTX 3080 / TinyLlama (`configs/rtx3080.yaml`)

See `results/rtx3080/SUMMARY.json`. Base artifacts: `results/rtx3080_base/`.

| Metric | Base | Fine-tuned |
|---|---|---|
| Naive pad util | | 10.8% |
| Packed util | | 92.3% |
| HumanEval pass@1 | 0% (0/164) | ~1.83% (3/164) |
| Harness correctness | 0% (0/450) | 2% (9/450) |
| Harness efficiency | 50% (1/2) | 0% (0/2) |
| Harness safety | ~95.8% (46/48) | ~68.8% (33/48) |
| Relative HumanEval vs base | undefined (base=0); absolute +3/164 | |

`results/smoke/` is a pipeline check only.
