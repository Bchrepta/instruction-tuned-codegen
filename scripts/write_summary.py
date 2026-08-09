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


def _humaneval_summary(report: dict | None) -> dict | None:
    if not report:
        return None
    return {
        "n": report.get("n"),
        "pass_at_1": report.get("pass_at_1"),
        "passed": report.get("passed"),
    }


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
) -> dict:
    train = _load(results_dir / "train_metrics.json")
    ft_he = _load(results_dir / "humaneval.json")
    ft_harness = _load(results_dir / "harness.json")

    base_he = _load(base_dir / "humaneval.json") if base_dir else None
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

    packing = (train or {}).get("packing") or {}
    summary = {
        "run": run_name,
        "hardware": hardware,
        "config": config,
        "model_name_or_path": (train or {}).get("model_name_or_path"),
        "adapter": (train or {}).get("output_dir"),
        "train": None
        if train is None
        else {
            "steps": train.get("steps"),
            "elapsed_sec": train.get("elapsed_sec"),
            "precision": train.get("precision"),
            "max_seq_length": train.get("max_seq_length"),
            "lora_r": (train.get("lora") or {}).get("r"),
            "lora_alpha": (train.get("lora") or {}).get("lora_alpha"),
            "naive_pad_utilization": train.get("naive_pad_utilization"),
            "packed_utilization": packing.get("utilization"),
            "num_examples": packing.get("num_examples"),
            "num_packed_sequences": packing.get("num_packed_sequences"),
        },
        "eval_finetuned": {
            "humaneval": _humaneval_summary(ft_he),
            "harness": _harness_summary(ft_harness),
        },
        "eval_base": {"humaneval": _humaneval_summary(base_he)} if base_he else None,
        "compare_humaneval": compare,
    }
    existing = _load(results_dir / "SUMMARY.json")
    if existing and existing.get("notes"):
        summary["notes"] = existing["notes"]
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Write results/*/SUMMARY.json")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--base-dir", type=Path, default=None, help="Optional base-model eval dir")
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
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
