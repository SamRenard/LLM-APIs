# Day 64 — LLM APIs (OpenAI & Anthropic)

A streaming command-line chatbot built on the OpenAI and Anthropic APIs, with conversation history, resilient retry/backoff logic, and per-response token and cost tracking.

Part of the **NIZAM AI** 150-day AI engineering protocol · Month 3 · LLM & RAG systems.

## Features

- **Streaming responses** — tokens are printed as they arrive, for both providers.
- **Conversation history** — multi-turn context with a configurable window (`--max-history`).
- **Retry with exponential backoff** — full-jitter backoff for rate limits (429), server errors (5xx) and network failures. Non-transient errors (e.g. 400, 401) fail fast.
- **Safe stream retries** — a stream is retried only if it fails *before* the first chunk, so output is never duplicated.
- **Token counting** — a pre-flight estimate (`tiktoken`) plus exact usage reported by the API.
- **Cost calculator** — per-response and per-session cost in USD, driven by `pricing.json`.

## Project structure

```
day64-llm-apis/
├── src/
│   ├── chatbot.py      # CLI loop, history, token/cost output
│   ├── providers.py    # OpenAI & Anthropic streaming adapters
│   ├── retry.py        # retry_call / retry_stream with backoff + jitter
│   └── cost.py         # token estimation and cost calculation
├── tests/
│   └── test_retry_cost.py
├── pricing.json        # USD per 1M tokens, per model
├── requirements.txt
├── .env.example
└── README.md
```

## Getting started

**Requirements:** Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# then add your API keys to .env
```

## Usage

Run from the project root:

```bash
# OpenAI (default)
python -m src.chatbot

# Anthropic
python -m src.chatbot --provider anthropic

# Custom model and system prompt
python -m src.chatbot --provider openai --model gpt-4o-mini --system "You are a Python tutor."
```

| Option | Default | Description |
|---|---|---|
| `--provider` | `openai` | `openai` or `anthropic` |
| `--model` | provider default | Model name (must exist in `pricing.json` for cost output) |
| `--system` | concise assistant | System prompt |
| `--max-history` | `20` | Maximum messages kept in context |

**In-chat commands:** `/reset` clears history · `/cost` shows session totals · `/quit` exits.

### Example output

```
you> Explain exponential backoff in one sentence.
bot> Exponential backoff retries a failed request with progressively longer waits ...
[tokens in=31 (est ~29) out=42 | cost=$0.000030]
```

## How it works

### Retry and backoff

Delay before retry *n* is drawn uniformly from `[0, min(cap, base × 2ⁿ)]` ("full jitter"), which avoids synchronized retries across clients. SDK-level retries are disabled (`max_retries=0`) so all retry behaviour is explicit and visible in one place.

### Token counting and cost

Cost is computed from the exact `usage` returned by each provider:

```
cost = input_tokens / 1e6 × input_price + output_tokens / 1e6 × output_price
```

The `tiktoken` estimate is shown only as a pre-flight approximation and may differ slightly from billed tokens, especially for Anthropic models.

### Pricing configuration

Prices live in `pricing.json` (USD per 1M tokens). **Verify them against the providers' official pricing pages** and add an entry for any model you use; unknown models display `n/a` instead of a wrong number.

## Testing

```bash
pytest -q
```

Tests cover retry success/failure paths, the "retry only before first chunk" stream rule, and cost calculation. They run offline and require no API keys.

## Resources

- [Anthropic API Docs](https://docs.anthropic.com)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Prompting Guide](https://www.promptingguide.ai)

## License

MIT
