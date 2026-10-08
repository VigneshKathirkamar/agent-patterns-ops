# agent-patterns-ops

Agentic design patterns, looked at like production systems.

Most write-ups explain what agent patterns *can do*. This repo measures what they
**cost**, how they **fail**, and how you'd **gate and monitor** them. Each pattern
gets one small, runnable experiment on a laptop, no cloud account needed.

| Pattern | Question | Status |
|---|---|---|
| [Reflection](reflection/) | Does self-critique improve output, and what does it cost? | ✅ |
| Tool use | How much do tool descriptions affect reliability? | planned |
| Planning | When does planning help, and when is it overhead? | planned |
| Multi-agent | Where do failures hide in agent handoffs? | planned |

## Requirements

- Python 3.8+ (standard library only, nothing to `pip install`)
- [Ollama](https://ollama.com) running locally with a pulled model, e.g. `ollama pull llama3.2`

## Caveats

These are small experiments with small local models on narrow tasks. Treat the
numbers as a starting point for your own measurements, not as general benchmarks.
Results vary run to run; rerun with your own model and share what you get.
