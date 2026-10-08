#!/usr/bin/env python3
"""
Reflection pattern, ops view: does self-critique improve output, and what does it cost?

The model writes is_valid_ipv4(s). A cheap unit-test check gates reflection:
  - tests pass  -> stop (no extra cost)
  - tests fail  -> model critiques its own code using the failures, then rewrites it
Repeats up to --max-rounds times. Tracks tests passed, tokens and time per round.

Setup:
  1. Install Ollama: https://ollama.com
  2. ollama pull llama3.2
  3. python reflection_demo.py --runs 5

Options:
  --model llama3.2        any model you've pulled
  --runs 5                repeat the experiment (LLM output varies run to run)
  --max-rounds 2          cap on reflection rounds (the "stop rule")
  --always-reflect        reflect even when tests already pass (shows wasted cost)

Note: Ollama may skip re-processing cached prompt tokens, so on a paid API
the real token multiplier is often higher than what you see here.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

TASK = """Write a Python function is_valid_ipv4(s: str) -> bool.
Return True only for a strict dotted-decimal IPv4 address:
- exactly four parts separated by dots
- each part is digits only (no signs, spaces, newlines or other characters)
- each part is between 0 and 255
- no leading zeros ("01" is invalid, "0" is valid)
Return only the code in a single ```python block."""

REFLECT = """Review your function critically.
{feedback}
First list the bugs or weaknesses you find in 1-3 short bullets.
Then give the full corrected function in a single ```python block."""

TESTS = [
    ("192.168.1.1", True),
    ("0.0.0.0", True),
    ("255.255.255.255", True),
    ("10.0.0.10", True),
    ("256.1.1.1", False),
    ("1.2.3", False),
    ("1.2.3.4.5", False),
    ("01.2.3.4", False),
    ("1..3.4", False),
    (" 1.2.3.4", False),
    ("1.2.3.+4", False),
    ("a.b.c.d", False),
    ("", False),
    ("1.2.3.4\n", False),
]

HARNESS = """
import json
{code}

tests = {tests!r}
results = []
for inp, expected in tests:
    try:
        got = bool(is_valid_ipv4(inp))
    except Exception as e:
        got = f"error: {{e}}"
    results.append([inp, expected, got])
print("__RESULTS__" + json.dumps(results))
"""


def chat(model, messages):
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    secs = time.time() - t0
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    return data["message"]["content"], tokens, secs


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def run_tests(code):
    """The cheap check: run the model's code against fixed test cases in a subprocess."""
    src = HARNESS.format(code=code, tests=TESTS)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        out = subprocess.run([sys.executable, path], capture_output=True,
                             text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return 0, ["(code timed out)"]
    finally:
        os.unlink(path)
    m = re.search(r"__RESULTS__(.*)", out.stdout)
    if not m:
        err = (out.stderr.strip().splitlines() or ["unknown error"])[-1]
        return 0, [f"(code did not run: {err})"]
    results = json.loads(m.group(1))
    failures = [f"is_valid_ipv4({inp!r}) returned {got}, expected {exp}"
                for inp, exp, got in results if got != exp]
    return len(results) - len(failures), failures


def one_run(model, max_rounds, always_reflect, run_no, save_dir):
    messages = [{"role": "user", "content": TASK}]
    reply, tokens, secs = chat(model, messages)
    code = extract_code(reply)
    passed, failures = run_tests(code)
    rounds = [{"passed": passed, "tokens": tokens, "secs": secs}]
    save(save_dir, run_no, 0, code)

    for i in range(1, max_rounds + 1):
        if not failures and not always_reflect:
            break  # stop rule: the cheap check passed, don't pay for reflection
        feedback = ("These checks failed:\n" + "\n".join(failures)) if failures \
            else "All checks passed."
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": REFLECT.format(feedback=feedback)}]
        reply, tokens, secs = chat(model, messages)
        code = extract_code(reply)
        passed, failures = run_tests(code)
        rounds.append({"passed": passed, "tokens": tokens, "secs": secs})
        save(save_dir, run_no, i, code)
    return rounds


def save(save_dir, run_no, round_no, code):
    with open(os.path.join(save_dir, f"run{run_no}_round{round_no}.py"), "w") as f:
        f.write(code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--always-reflect", action="store_true")
    args = p.parse_args()

    save_dir = "runs"
    os.makedirs(save_dir, exist_ok=True)
    n = len(TESTS)
    all_runs = []

    for r in range(1, args.runs + 1):
        rounds = one_run(args.model, args.max_rounds, args.always_reflect, r, save_dir)
        all_runs.append(rounds)
        path = " -> ".join(f"{x['passed']}/{n}" for x in rounds)
        tok = sum(x["tokens"] for x in rounds)
        ratio = tok / rounds[0]["tokens"] if rounds[0]["tokens"] else 0
        print(f"Run {r}: tests {path:<22} tokens {rounds[0]['tokens']:>5} -> {tok:>6}  ({ratio:.1f}x)")

    k = len(all_runs)
    draft_ok = sum(run[0]["passed"] == n for run in all_runs)
    final_ok = sum(run[-1]["passed"] == n for run in all_runs)
    avg = lambda xs: sum(xs) / len(xs)
    d_pass = avg([run[0]["passed"] for run in all_runs])
    f_pass = avg([run[-1]["passed"] for run in all_runs])
    d_tok = avg([run[0]["tokens"] for run in all_runs])
    f_tok = avg([sum(x["tokens"] for x in run) for run in all_runs])
    d_sec = avg([run[0]["secs"] for run in all_runs])
    f_sec = avg([sum(x["secs"] for x in run) for run in all_runs])
    reflections = sum(len(run) - 1 for run in all_runs)
    skipped = sum(len(run) == 1 for run in all_runs)

    mode = "always reflect" if args.always_reflect else "reflect only if tests fail"
    print(f"\n=== Reflection experiment | {args.model} | {k} runs | {mode} ===")
    print(f"First draft passed all tests : {draft_ok}/{k} runs")
    print(f"After reflection passed all  : {final_ok}/{k} runs")
    print(f"Avg tests passed             : {d_pass:.1f}/{n} -> {f_pass:.1f}/{n}")
    print(f"Avg tokens per run           : {d_tok:,.0f} -> {f_tok:,.0f}  ({f_tok / d_tok:.1f}x)")
    print(f"Avg time per run             : {d_sec:.1f}s -> {f_sec:.1f}s  ({f_sec / d_sec:.1f}x)")
    print(f"Reflection rounds used       : {reflections}  (skipped entirely in {skipped} runs)")
    print(f"Code from every round saved in ./{save_dir}/")


if __name__ == "__main__":
    main()
