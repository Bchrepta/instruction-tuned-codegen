# Metric targets

Fill this in after a full A100-style run (`configs/a100.yaml`).

| Metric | Target | How measured |
|---|---|---|
| Relative HumanEval gain vs base | about +23% | `(ft - base) / base` on pass@1 |
| GPU util, naive pad to packed | 67% to 94% | `benchmark-packing` + trainer logs |
| Train time cut from packing | about 38% | wall-clock vs naive pad |
| Harness size | 500 | correctness / efficiency / safety |

`results/smoke/` is a pipeline check only. Put home-GPU runs under `results/rtx3080/` (or similar).
