# Day 63 — Structured Output & Tool Calling: Mini Agent

Part of **NIZAM AI · 150-Day AI Engineering Protocol** · Month 3 · Block Day 3/5

## 📌 Concept

**Tool calling**: you describe tools (name, description, JSON input schema).
The model never runs them. When it wants one, it returns a `tool_use` block;
your code runs the tool and sends back a `tool_result`; the model then writes
the final answer. Repeating that is the **agent loop**.

**Structured output**: define a JSON schema as a tool and force it with
`tool_choice`; the model's reply is a dict that matches the schema.

## 📂 Files

| File | Description |
|---|---|
| `tool_agent.py` | `get_weather` + `calculator` tools, `run_agent()` loop (with a step cap), `extract_structured()` using a forced schema tool |
| `test_tool_agent.py` | 10 offline tests with a scripted fake client (no API key / network) |

## ▶️ Usage

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"   # never commit this
python3 tool_agent.py                 # live demo
python3 test_tool_agent.py            # offline tests
```

Model defaults to `claude-sonnet-5-5`; override with `CLAUDE_MODEL=...`.

## ⚠️ Notes

- `get_weather` returns **fixed sample data** for 3 cities (Baku, London,
  Tokyo), not live weather. Replace it with a real weather API call for
  real use.
- `calculator` parses with `ast` and allows only numbers and `+ - * / % **`
  (exponent capped), so it can't execute arbitrary code the way `eval()` would.
- Tool errors are returned to the model with `is_error=True` so it can recover.

## ✅ Results

```
10/10 offline checks passed
```

Tests cover the tools (values, rejected unsafe input, errors), the agent
loop (tool_use -> tool_result with matching id, multiple tools in one
step, error pass-through, step cap), and forced structured output. The
live demo needs your API key and was **not run** in the sandbox this was
written in.

## 📚 Sources

- Anthropic API Docs (tool use)
- OpenAI API Docs (function calling)
- Prompting Guide

---
*NIZAM AI Day 63/150*
