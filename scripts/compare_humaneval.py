"""Compare base vs fine-tuned HumanEval pass@1 (relative gain)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True, help="base humaneval.json")
    p.add_argument("--ft", type=Path, required=True, help="fine-tuned humaneval.json")
    args = p.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    ft = json.loads(args.ft.read_text(encoding="utf-8"))
    b = float(base["pass_at_1"])
    f = float(ft["pass_at_1"])
    rel = (f - b) / b if b > 0 else None
    out = {
        "base_pass_at_1": b,
        "ft_pass_at_1": f,
        "absolute_delta": f - b,
        "relative_improvement": rel,
        "relative_improvement_pct": None if rel is None else round(100 * rel, 2),
        "target_relative_pct": 23.0,
    }
    if rel is None and b == 0.0:
        out["relative_improvement_note"] = (
            "undefined when base pass@1 is 0; report absolute_delta instead"
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
