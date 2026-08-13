"""HumanEval-style functional correctness evaluation.

Uses the openai_humaneval dataset when available. Falls back to a bundled
mini subset so smoke runs work offline / without extra deps.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Compact offline subset for smoke tests (subset of classic HumanEval ideas).
SMOKE_TASKS = [
    {
        "task_id": "smoke/0",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return a + b."""\n',
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
        "entry_point": "add",
    },
    {
        "task_id": "smoke/1",
        "prompt": 'def is_palindrome(s: str) -> bool:\n    """Return True if s is a palindrome."""\n',
        "canonical_solution": "    return s == s[::-1]\n",
        "test": "def check(candidate):\n    assert candidate('aba') is True\n    assert candidate('ab') is False\n",
        "entry_point": "is_palindrome",
    },
    {
        "task_id": "smoke/2",
        "prompt": 'def factorial(n: int) -> int:\n    """Return n! for n >= 0."""\n',
        "canonical_solution": "    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
        "test": "def check(candidate):\n    assert candidate(0) == 1\n    assert candidate(5) == 120\n",
        "entry_point": "factorial",
    },
    {
        "task_id": "smoke/3",
        "prompt": 'def max_of_two(a: int, b: int) -> int:\n    """Return the larger of a and b."""\n',
        "canonical_solution": "    return a if a >= b else b\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 2\n    assert candidate(5, 5) == 5\n",
        "entry_point": "max_of_two",
    },
    {
        "task_id": "smoke/4",
        "prompt": 'def reverse_list(xs: list) -> list:\n    """Return a new list with elements reversed."""\n',
        "canonical_solution": "    return xs[::-1]\n",
        "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == [3, 2, 1]\n    assert candidate([]) == []\n",
        "entry_point": "reverse_list",
    },
    {
        "task_id": "smoke/5",
        "prompt": 'def count_vowels(s: str) -> int:\n    """Count vowels aeiou in s (case-insensitive)."""\n',
        "canonical_solution": "    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "test": "def check(candidate):\n    assert candidate('hello') == 2\n    assert candidate('xyz') == 0\n",
        "entry_point": "count_vowels",
    },
    {
        "task_id": "smoke/6",
        "prompt": 'def clamp(x: float, lo: float, hi: float) -> float:\n    """Clamp x into [lo, hi]."""\n',
        "canonical_solution": "    return max(lo, min(hi, x))\n",
        "test": "def check(candidate):\n    assert candidate(5, 0, 3) == 3\n    assert candidate(-1, 0, 3) == 0\n",
        "entry_point": "clamp",
    },
    {
        "task_id": "smoke/7",
        "prompt": 'def unique(xs: list) -> list:\n    """Return unique elements preserving order."""\n',
        "canonical_solution": "    out = []\n    seen = set()\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        "test": "def check(candidate):\n    assert candidate([1, 1, 2, 2, 3]) == [1, 2, 3]\n",
        "entry_point": "unique",
    },
    {
        "task_id": "smoke/8",
        "prompt": 'def mean(xs: list[float]) -> float:\n    """Arithmetic mean of xs; raise ValueError if empty."""\n',
        "canonical_solution": "    if not xs:\n        raise ValueError('empty')\n    return sum(xs) / len(xs)\n",
        "test": "def check(candidate):\n    assert abs(candidate([1, 2, 3]) - 2.0) < 1e-9\n",
        "entry_point": "mean",
    },
    {
        "task_id": "smoke/9",
        "prompt": 'def starts_with_vowel(s: str) -> bool:\n    """True if s starts with a vowel."""\n',
        "canonical_solution": "    return bool(s) and s[0].lower() in 'aeiou'\n",
        "test": "def check(candidate):\n    assert candidate('apple') is True\n    assert candidate('banana') is False\n",
        "entry_point": "starts_with_vowel",
    },
]


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    error: str | None = None


def load_humaneval_tasks(n: int | None = None) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test")
        tasks = [dict(ds[i]) for i in range(len(ds))]
    except Exception as e:  # noqa: BLE001
        logger.warning("Falling back to smoke HumanEval subset: %s", e)
        tasks = list(SMOKE_TASKS)
    if n is not None:
        tasks = tasks[:n]
    return tasks


def extract_code(completion: str, prompt: str) -> str:
    """Extract Python solution body from model completion."""
    text = completion
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                body = part
                if body.startswith("python"):
                    body = body[len("python") :]
                text = body.strip()
                break
    # If model echoed the prompt signature, keep from first def
    if "def " in text and "def " in prompt:
        # Prefer completion-only: strip leading prompt echo
        if text.startswith(prompt):
            text = text[len(prompt) :]
    return text


def unsafe_execute(code: str, timeout_s: float = 3.0) -> tuple[bool, str | None]:
    """Execute code in-process with a soft timeout (Unix signal; best-effort on Windows)."""

    def _run() -> None:
        exec(code, {"__builtins__": __builtins__})  # noqa: S102

    try:
        # Silence prints from candidate solutions / tests during scoring.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if hasattr(signal, "SIGALRM"):
                def handler(signum, frame):  # noqa: ANN001
                    raise TimeoutError("timeout")

                old = signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
                try:
                    _run()
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
            else:
                _run()
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def check_correctness(task: dict, completion: str, timeout_s: float = 3.0) -> EvalResult:
    prompt = task["prompt"]
    entry = task["entry_point"]
    test = task["test"]
    body = extract_code(completion, prompt)
    # Assemble program: prompt + completion body + tests
    program = prompt + body
    if "def check(" not in program:
        program = program + "\n" + test + f"\ncheck({entry})\n"
    else:
        program = program + f"\ncheck({entry})\n"

    # Quick syntax gate
    try:
        ast.parse(program)
    except SyntaxError as e:
        return EvalResult(task_id=task["task_id"], passed=False, error=f"SyntaxError: {e}")

    ok, err = unsafe_execute(program, timeout_s=timeout_s)
    return EvalResult(task_id=task["task_id"], passed=ok, error=err)


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    # Avoid transformers warning when both max_length and max_new_tokens are set.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text


def evaluate_humaneval(
    model,
    tokenizer,
    n: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> dict:
    tasks = load_humaneval_tasks(n)
    results: list[EvalResult] = []
    logger.info("HumanEval start: n=%d", len(tasks))
    for i, task in enumerate(tasks, start=1):
        completion = generate_completion(
            model,
            tokenizer,
            task["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(check_correctness(task, completion))
        if i == 1 or i % 10 == 0 or i == len(tasks):
            logger.info(
                "HumanEval %d/%d passed=%d",
                i,
                len(tasks),
                sum(1 for r in results if r.passed),
            )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    report = {
        "benchmark": "HumanEval",
        "n": total,
        "pass_at_1": passed / total if total else 0.0,
        "passed": passed,
        "details": [r.__dict__ for r in results],
    }
    return report


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
