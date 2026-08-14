#!/usr/bin/env python
"""Build a compact SUMMARY.json from train + eval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _humaneval_summary(report: dict | None) -> dict | None:
    if not report:
        return None
    out = {
        "n": report.get("n"),
        "pass_at_1": report.get("pass_at_1"),
        "passed": report.get("passed"),
    }
    if report.get("prompt_mode") is not None:
        out["prompt_mode"] = report.get("prompt_mode")
    if report.get("error_histogram") is not None:
        out["error_histogram"] = report.get("error_histogram")
    return out


def _harness_summary(report: dict | None) -> dict | None:
    if not report:
        return None
    return {
        "n_total": report.get("n_total"),
        "correctness_pass_rate": (report.get("correctness") or {}).get("pass_rate"),
        "efficiency_pass_rate": (report.get("efficiency") or {}).get("pass_rate"),
        "safety_pass_rate": (report.get("safety") or {}).get("pass_rate"),
    }


def build_summary(
    results_dir: Path,
    *,
    run_name: str,
    hardware: str | None,
    config: str | None,
    base_dir: Path | None,
    train_metrics: Path | None = None,
) -> dict:
    train = _load(train_metrics) if train_metrics else None
    if train is None:
        train = _load(results_dir / "train_metrics.json")
    ft_he = _load(results_dir / "humaneval.json")
    ft_harness = _load(results_dir / "harness.json")
    cfg = _load_config(Path(config)) if config else None

    base_he = _load(base_dir / "humaneval.json") if base_dir else None
    base_harness = _load(base_dir / "harness.json") if base_dir else None
    compare = None
    if base_he and ft_he and base_he.get("pass_at_1") is not None:
        b = float(base_he["pass_at_1"])
        f = float(ft_he["pass_at_1"])
        rel = (f - b) / b if b > 0 else None
        compare = {
            "base_pass_at_1": b,
            "ft_pass_at_1": f,
            "absolute_delta": f - b,
            "relative_improvement": rel,
            "relative_improvement_pct": None if rel is None else round(100 * rel, 2),
        }
        if rel is None and b == 0.0:
            compare["relative_improvement_note"] = (
                "undefined when base pass@1 is 0; report absolute_delta instead"
            )

    packing = (train or {}).get("packing") or {}
    eval_base = None
    if base_he or base_harness:
        eval_base = {
            "humaneval": _humaneval_summary(base_he),
            "harness": _harness_summary(base_harness),
        }

    model_name = (train or {}).get("model_name_or_path") or (cfg or {}).get("model_name_or_path")
    adapter = (train or {}).get("output_dir") or (cfg or {}).get("output_dir")
    summary = {
        "run": run_name,
        "hardware": hardware,
        "config": config,
        "model_name_or_path": model_name,
        "adapter": adapter,
        "train": None
        if train is None
        else {
            "steps": train.get("steps"),
            "elapsed_sec": train.get("elapsed_sec"),
            "precision": train.get("precision"),
            "max_seq_length": train.get("max_seq_length"),
            "lora_r": (train.get("lora") or {}).get("r") or ((cfg or {}).get("lora") or {}).get("r"),
            "lora_alpha": (train.get("lora") or {}).get("lora_alpha")
            or ((cfg or {}).get("lora") or {}).get("lora_alpha"),
            "naive_pad_utilization": train.get("naive_pad_utilization"),
            "packed_utilization": packing.get("utilization"),
            "num_examples": packing.get("num_examples"),
            "num_packed_sequences": packing.get("num_packed_sequences"),
        },
        "eval_finetuned": {
            "humaneval": _humaneval_summary(ft_he),
            "harness": _harness_summary(ft_harness),
        },
        "eval_base": eval_base,
        "compare_humaneval": compare,
    }
    notes: list[str] = []
    train_block = summary.get("train") or {}
    naive = train_block.get("naive_pad_utilization")
    packed = train_block.get("packed_utilization")
    if naive is not None and packed is not None:
        notes.append(
            f"Packing is the strongest measured result on this run "
            f"({100 * float(naive):.1f}% naive pad to {100 * float(packed):.1f}% packed)."
        )
    ft_he_s = (summary.get("eval_finetuned") or {}).get("humaneval") or {}
    if ft_he_s.get("pass_at_1") is not None:
        model_note = model_name or "this model"
        notes.append(
            f"HumanEval pass@1 (fine-tuned) is {100 * float(ft_he_s['pass_at_1']):.2f}% "
            f"({ft_he_s.get('passed')}/{ft_he_s.get('n')}) for {model_note}. "
            "Home-GPU QLoRA/TinyLlama numbers are not the A100 bf16 target."
        )
    if compare and compare.get("relative_improvement") is None and compare.get("base_pass_at_1") == 0.0:
        passed = ft_he_s.get("passed")
        n = ft_he_s.get("n")
        delta = compare.get("absolute_delta")
        notes.append(
            f"Base HumanEval was 0/{n}, so relative gain is undefined; "
            f"absolute delta is +{passed}/{n} ({100 * float(delta):.2f} pp)."
        )
    elif eval_base is None:
        notes.append(
            "Base eval not present; run with --no-adapter into a base results dir and re-run this script."
        )
    base_harness_s = (eval_base or {}).get("harness") if eval_base else None
    ft_harness_s = (summary.get("eval_finetuned") or {}).get("harness") or {}
    if base_harness_s and ft_harness_s.get("safety_pass_rate") is not None:
        notes.append(
            f"Harness safety: base {100 * float(base_harness_s['safety_pass_rate']):.1f}% vs "
            f"fine-tuned {100 * float(ft_harness_s['safety_pass_rate']):.1f}%."
        )
    summary["notes"] = notes
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Write results/*/SUMMARY.json")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--base-dir", type=Path, default=None, help="Optional base-model eval dir")
    p.add_argument("--train-metrics", type=Path, default=None, help="Optional train_metrics.json path")
    p.add_argument("--run-name", default=None)
    p.add_argument("--hardware", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    results_dir = args.results_dir
    run_name = args.run_name or results_dir.name
    out = args.output or (results_dir / "SUMMARY.json")
    summary = build_summary(
        results_dir,
        run_name=run_name,
        hardware=args.hardware,
        config=args.config,
        base_dir=args.base_dir,
        train_metrics=args.train_metrics,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
