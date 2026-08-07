"""Create a tiny offline smoke dataset without HuggingFace download."""

from __future__ import annotations

import json
from pathlib import Path

EXAMPLES = [
    {
        "instruction": "Write a Python function that adds two numbers.",
        "input": "",
        "output": "def add(a, b):\n    return a + b",
    },
    {
        "instruction": "Write a function that checks if a string is a palindrome.",
        "input": "",
        "output": "def is_palindrome(s):\n    return s == s[::-1]",
    },
    {
        "instruction": "Implement factorial.",
        "input": "",
        "output": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
    },
    {
        "instruction": "Reverse a list.",
        "input": "",
        "output": "def reverse_list(xs):\n    return xs[::-1]",
    },
    {
        "instruction": "Count vowels in a string.",
        "input": "",
        "output": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
    },
    {
        "instruction": "Clamp a value between lo and hi.",
        "input": "",
        "output": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    },
    {
        "instruction": "Return unique elements preserving order.",
        "input": "",
        "output": "def unique(xs):\n    out, seen = [], set()\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
    },
    {
        "instruction": "Compute the mean of a list of floats.",
        "input": "",
        "output": "def mean(xs):\n    if not xs:\n        raise ValueError('empty')\n    return sum(xs) / len(xs)",
    },
]


def format_row(ex: dict) -> dict:
    if ex["input"]:
        text = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}"
        )
    return {**ex, "text": text}


def main() -> None:
    out = Path("data/processed/codealpaca_smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [format_row(ex) for ex in EXAMPLES]
    # Repeat to give packing something to chew on
    rows = rows * 8
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
