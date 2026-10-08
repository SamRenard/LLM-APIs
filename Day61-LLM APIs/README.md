# Day 61 — LLM APIs: First Chat Completion Script

Part of **NIZAM AI · 150-Day AI Engineering Protocol** · Month 3 · Block Day 1/5

## 📌 Concept

LLMs read and write **tokens** (word fragments). The **context window** is
the max tokens (prompt + reply) the model sees at once. **Temperature**
controls randomness (≈0 = near-deterministic, high = varied). **top_p**
samples only from the most probable tokens. The **system prompt** sets the
model's role/rules before the user message.

## 📂 Files

| File | Description |
|---|---|
| `llm_compare.py` | `ask()` wrapper around the Anthropic Messages API + 3 experiments: temperature, system prompt, `max_tokens` truncation |
| `test_llm_compare.py` | 5 tests using a fake client (no API key / network needed) |

## ▶️ Usage

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"   # never commit this
python3 llm_compare.py                # live experiments
python3 test_llm_compare.py           # offline tests
```

Model defaults to `claude-sonnet-5-5`; override with `CLAUDE_MODEL=...`.

## ✅ Results

```
5/5 offline checks passed
```

The tests verify request construction (parameters passed correctly,
optional ones omitted, `temperature` + `top_p` rejected together).
The live experiments in `llm_compare.py` need your own API key and were
not run in the sandbox this was written in.

## 📚 Sources

- Anthropic API Docs
- OpenAI API Docs
- Prompting Guide

---
*NIZAM AI Day 61/150*
