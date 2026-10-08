# Day 62 — Prompt Engineering: Strategy Comparison Tool

Part of **NIZAM AI · 150-Day AI Engineering Protocol** · Month 3 · Block Day 2/5

## 📌 Concept

Same model, different prompt, different quality.
- **Zero-shot**: just the task, no examples.
- **Few-shot**: 2-3 worked examples inside the prompt; the model copies the format and criteria.
- **Chain-of-thought**: ask for step-by-step reasoning before the answer.
- **Role prompting**: a system prompt sets perspective and tone.
- **Structured output**: constrain the reply format so code can parse it.

## 📂 Files

| File | Description |
|---|---|
| `prompt_compare.py` | Builds 5 prompt strategies for one task (review sentiment), runs them over 6 labeled reviews (incl. sarcasm and mixed-feeling cases), prints a side-by-side accuracy table |
| `test_prompt_compare.py` | 7 offline tests with a fake client (no API key / network) |

## ▶️ Usage

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"   # never commit this
python3 prompt_compare.py             # live comparison table
python3 test_prompt_compare.py        # offline tests
```

Model defaults to `claude-sonnet-5-5`; override with `CLAUDE_MODEL=...`.

## ✅ Results

```
7/7 offline checks passed
```

The tests verify prompt construction, label parsing, `temperature=0`,
and scoring logic. The live accuracy table needs your own API key and
was **not run** in the sandbox this was written in, so no accuracy
numbers are claimed here. With only 6 reviews, differences between
strategies will be small and noisy; add more labeled reviews for a
more meaningful comparison.

## 📚 Sources

- Anthropic API Docs
- OpenAI API Docs
- Prompting Guide

---
*NIZAM AI Day 62/150*
