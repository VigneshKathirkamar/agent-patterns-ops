# Reflection: quality vs. cost

**Question:** If an LLM critiques and rewrites its own output, how much better does
it get, and what does that cost in tokens and latency?

## The setup

1. The model writes `is_valid_ipv4(s)` from a short spec.
2. A **cheap check** runs first: 14 unit tests, including edge cases like
   `" 1.2.3.4"`, `"1.2.3.+4"`, `"01.2.3.4"` and a trailing newline.
3. **Stop rule:** if all tests pass, stop. No reflection, no extra cost.
4. If tests fail, the model gets the failures, critiques its own code and rewrites it.
   Up to `--max-rounds` times (default 2).

Tokens, time and tests passed are recorded for every round. The code from every
round is saved to `./runs/` so you can diff what reflection actually changed.

## Run it

```bash
ollama pull llama3.2
python reflection_demo.py --runs 5                    # reflect only when tests fail
python reflection_demo.py --runs 5 --always-reflect   # reflect every time (no stop rule)
python reflection_demo.py --model qwen2.5 --runs 5    # try another model
```

## Results

_Model: [model] · Machine: [laptop] · Date: [date]_

| Mode | First draft passed all | After reflection passed all | Token multiplier | Time multiplier |
|---|---|---|---|---|
| Reflect only if tests fail | [ ]/5 | [ ]/5 | [ ]x | [ ]x |
| Always reflect | [ ]/5 | [ ]/5 | [ ]x | [ ]x |

**What I noticed:** [your observations]

## Notes

- Ollama may skip re-processing cached prompt tokens, so on a paid API the real
  token multiplier is often higher than shown here.
- The model's code runs in a subprocess with a 10s timeout. It's a local toy
  experiment; don't point this at untrusted models or tasks.
