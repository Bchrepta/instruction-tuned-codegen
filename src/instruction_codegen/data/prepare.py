"""Prepare CodeAlpaca instruction data (15k full / smoke subset)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_example(ex: dict) -> dict:
    instruction = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    output = (ex.get("output") or "").strip()
    if inp:
        text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    else:
        text = PROMPT_NO_INPUT.format(instruction=instruction, output=output)
    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "text": text,
    }


def prepare(
    output_path: Path,
    n: int = 15_000,
    seed: int = 42,
    dataset_id: str = "sahil2801/CodeAlpaca-20k",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id, split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: min(n, len(indices))]

    with output_path.open("w", encoding="utf-8") as f:
        for i in indices:
            row = format_example(ds[int(i)])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(indices)} examples -> {output_path}")
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare CodeAlpaca instruction dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=15_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="sahil2801/CodeAlpaca-20k")
    args = p.parse_args()
    prepare(args.output, n=args.n, seed=args.seed, dataset_id=args.dataset)


if __name__ == "__main__":
    main()
