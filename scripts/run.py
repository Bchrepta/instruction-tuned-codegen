#!/usr/bin/env python
"""CLI entrypoints."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cmd_prepare(args: argparse.Namespace) -> None:
    from instruction_codegen.data.prepare import prepare

    prepare(Path(args.output), n=args.n, seed=args.seed, dataset_id=args.dataset)


def cmd_train(args: argparse.Namespace) -> None:
    from instruction_codegen.config import load_config
    from instruction_codegen.training.train import train

    cfg = load_config(args.config)
    train(cfg)


def cmd_eval(args: argparse.Namespace) -> None:
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from instruction_codegen.config import load_config, project_root
    from instruction_codegen.eval.harness import run_harness, save_harness_report
    from instruction_codegen.eval.humaneval_eval import evaluate_humaneval, save_report

    cfg = load_config(args.config)
    root = project_root()
    out = Path(args.output) if args.output else root / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)

    base = args.base_model or cfg["model_name_or_path"]
    adapter = args.adapter
    if adapter is None:
        adapter = cfg.get("output_dir")
    adapter_path = Path(adapter)
    if not adapter_path.is_absolute():
        adapter_path = root / adapter_path

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path.exists() else base, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    use_adapter = not getattr(args, "no_adapter", False)
    if use_adapter and adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
    elif use_adapter and not adapter_path.exists():
        print(f"Warning: adapter not found at {adapter_path}; evaluating base model only")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    ev = cfg.get("eval", {})
    he = evaluate_humaneval(
        model,
        tokenizer,
        n=args.humaneval_n or ev.get("humaneval_n"),
        max_new_tokens=int(ev.get("max_new_tokens", 256)),
        temperature=float(ev.get("temperature", 0.2)),
        top_p=float(ev.get("top_p", 0.95)),
        prompt_mode=getattr(args, "prompt_mode", None) or ev.get("prompt_mode", "raw"),
    )
    save_report(he, out / "humaneval.json")
    print(
        json.dumps(
            {
                k: he[k]
                for k in ("benchmark", "n", "pass_at_1", "passed", "prompt_mode", "error_histogram")
                if k in he
            },
            indent=2,
        )
    )

    harness_n = args.harness_n if args.harness_n is not None else int(ev.get("harness_n", 500))
    if harness_n <= 0:
        print("Skipping harness (--harness-n <= 0)")
        return

    harness = run_harness(
        model,
        tokenizer,
        n=harness_n,
        max_new_tokens=int(ev.get("max_new_tokens", 256)),
        temperature=float(ev.get("temperature", 0.2)),
        top_p=float(ev.get("top_p", 0.95)),
    )
    save_harness_report(harness, out / "harness.json")
    summary = {
        "n_total": harness["n_total"],
        "correctness": harness["correctness"],
        "efficiency": harness["efficiency"],
        "safety": harness["safety"],
    }
    # strip nested details from printed correctness mean fields already ok
    print(json.dumps(summary, indent=2))


def cmd_benchmark_packing(args: argparse.Namespace) -> None:
    import json

    from transformers import AutoTokenizer

    from instruction_codegen.config import load_config, project_root
    from instruction_codegen.data.packing import (
        build_packed_dataloader,
        estimate_naive_utilization,
    )

    cfg = load_config(args.config)
    root = project_root()
    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq = int(cfg["max_seq_length"])
    naive = estimate_naive_utilization(dataset_path, tokenizer, max_seq)
    _, stats = build_packed_dataloader(
        dataset_path,
        tokenizer,
        max_seq_length=max_seq,
        batch_size=int(cfg["per_device_train_batch_size"]),
        shuffle=False,
    )
    # Approximate time savings from fewer sequences processed
    time_reduction = 1.0 - (stats.num_packed_sequences / max(stats.num_examples, 1))
    report = {
        "naive_pad_utilization": round(naive, 4),
        "packed_utilization": stats.utilization,
        "num_examples": stats.num_examples,
        "num_packed_sequences": stats.num_packed_sequences,
        "approx_step_reduction": round(time_reduction, 4),
        "targets": {
            "gpu_util_before": 0.67,
            "gpu_util_after": 0.94,
            "train_time_cut": 0.38,
        },
    }
    out = Path(args.output) if args.output else root / "results" / "packing_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="instruction-codegen")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data", help="Download/format CodeAlpaca subset")
    prep.add_argument("--output", required=True)
    prep.add_argument("--n", type=int, default=15_000)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--dataset", default="sahil2801/CodeAlpaca-20k")
    prep.set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="LoRA fine-tune")
    tr.add_argument("--config", required=True)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="HumanEval + harness evaluation")
    ev.add_argument("--config", required=True)
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--no-adapter", action="store_true", help="Eval base model only (skip LoRA adapter)")
    ev.add_argument("--base-model", default=None)
    ev.add_argument("--output", default=None)
    ev.add_argument("--humaneval-n", type=int, default=None)
    ev.add_argument("--harness-n", type=int, default=None)
    ev.add_argument(
        "--prompt-mode",
        choices=("raw", "code", "chat"),
        default=None,
        help="HumanEval prompt style: raw continuation (default), code instruction wrap, or chat",
    )
    ev.set_defaults(func=cmd_eval)

    bp = sub.add_parser("benchmark-packing", help="Measure packing vs naive pad utilization")
    bp.add_argument("--config", required=True)
    bp.add_argument("--output", default=None)
    bp.set_defaults(func=cmd_benchmark_packing)

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
