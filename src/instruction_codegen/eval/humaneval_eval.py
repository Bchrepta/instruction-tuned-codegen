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


# HumanEval / Codex-style stops. Applied to the completion only (not the prompt).
STOP_SEQUENCES = (
    "\nclass ",
    "\nclass\t",
    "\ndef ",
    "\ndef\t",
    "\n#",
    "\nif __name__",
    "\nprint(",
    "\nassert ",
    "\n@@@",
)


def wrap_code_completion_prompt(code_prompt: str) -> str:
    """Match CodeAlpaca SFT format so instruction-tuned adapters stay on-distribution."""
    return (
        "### Instruction:\n"
        "Complete the following Python function. "
        "Only write the continuation of the function body, preserving indentation. "
        "Do not repeat the function signature or docstring.\n\n"
        f"### Input:\n{code_prompt}\n\n"
        "### Response:\n"
    )


def wrap_chat_prompt(user_text: str) -> str:
    return f"### Instruction:\n{user_text.strip()}\n\n### Response:\n"


def truncate_at_stop_sequences(text: str) -> str:
    cut = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx > 0:
            cut = min(cut, idx)
    return text[:cut]


def strip_markdown_fence(text: str) -> str:
    if "```" not in text:
        return text
    parts = text.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1:
            body = part
            first, _, rest = body.partition("\n")
            if first.strip().lower() in {"python", "py", "python3"}:
                body = rest
            elif first.strip() == "":
                body = rest
            return body.strip("\n")
    return text


def _drop_rewritten_signature(text: str, entry_point: str | None) -> str:
    """If the model re-emits `def entry(...):`, keep only the function body."""
    stripped = text.lstrip("\n")
    if not stripped.lstrip().startswith("def "):
        return text
    first_line, _, remainder = stripped.partition("\n")
    head = first_line.lstrip()
    if entry_point and f"def {entry_point}" not in head and "def " in head:
        # Different top-level def: treat as stop junk and drop everything.
        return ""
    # Body is everything after the signature line.
    return remainder


def _indent_body_if_needed(body: str) -> str:
    """HumanEval prompts end inside a function; bare top-level lines need indent."""
    if not body.strip():
        return body
    lines = body.splitlines(True)
    first = next((ln for ln in lines if ln.strip()), "")
    if first.startswith((" ", "\t")):
        return body
    out: list[str] = []
    for ln in lines:
        if not ln.strip() or ln.startswith((" ", "\t")):
            out.append(ln)
        else:
            ending = "\n" if ln.endswith("\n") else ""
            core = ln[:-1] if ln.endswith("\n") else ln
            out.append("    " + core + ending)
    return "".join(out)


def extract_code(completion: str, prompt: str, entry_point: str | None = None) -> str:
    """Turn a model completion into a function-body continuation for `prompt + body`."""
    text = strip_markdown_fence(completion)
    text = truncate_at_stop_sequences(text)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    for marker in ("### Response:", "### Instruction:", "### Input:"):
        if marker in text:
            text = text.split(marker)[-1]
    text = strip_markdown_fence(text)
    text = truncate_at_stop_sequences(text)
    text = _drop_rewritten_signature(text, entry_point)
    text = _indent_body_if_needed(text)
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
    body = extract_code(completion, prompt, entry_point=entry)
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
    *,
    prompt_mode: str = "code",
) -> str:
    import torch

    if prompt_mode == "code":
        gen_prompt = wrap_code_completion_prompt(prompt)
    elif prompt_mode == "chat":
        gen_prompt = wrap_chat_prompt(prompt)
    else:
        gen_prompt = prompt
    inputs = tokenizer(gen_prompt, return_tensors="pt")
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

    # Prefer native stop strings when the installed transformers supports them.
    if prompt_mode == "code":
        gen_kwargs["stop_strings"] = ["\nclass", "\ndef", "\n#", "\nif __name__", "\nprint("]
        gen_kwargs["tokenizer"] = tokenizer

    # Avoid transformers warning when both max_length and max_new_tokens are set.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    with torch.no_grad():
        try:
            out = model.generate(**inputs, **gen_kwargs)
        except TypeError:
            gen_kwargs.pop("stop_strings", None)
            gen_kwargs.pop("tokenizer", None)
            out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    if prompt_mode == "code":
        text = truncate_at_stop_sequences(text)
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
            prompt_mode="code",
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
