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

See `results/rtx3080/SUMMARY.json` for the full record.

| Metric | Measured |
|---|---|
| Naive pad util | 10.8% |
| Packed util | 92.3% |
| HumanEval pass@1 (fine-tuned) | ~1.83% (3/164) |
| Harness (correct / eff / safety) | 2% / 0% / ~68.8% |
| Relative HumanEval vs base | not yet (needs `results/rtx3080_base`) |

`results/smoke/` is a pipeline check only.
