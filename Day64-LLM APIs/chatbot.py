"""Streaming CLI chatbot with history and per-response token/cost tracking."""
import argparse
import sys

from dotenv import load_dotenv

from .cost import calculate_cost, estimate_tokens, load_pricing
from .providers import DEFAULT_MODELS, PROVIDERS, Usage
from .retry import retry_stream

HELP = "Commands: /reset  clear history | /cost  session totals | /quit  exit"


def format_cost(value) -> str:
    return "n/a (add model to pricing.json)" if value is None else f"${value:.6f}"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Streaming LLM chatbot")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default="You are a concise, helpful assistant.")
    parser.add_argument("--max-history", type=int, default=20,
                        help="max messages kept in context")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    stream_fn = PROVIDERS[args.provider]
    pricing = load_pricing()
    history: list = []
    total_in = total_out = 0
    total_cost = 0.0

    print(f"{args.provider} / {model}\n{HELP}\n")

    def on_retry(n, delay, exc):
        print(f"\n[retry {n}: {type(exc).__name__}, waiting {delay:.1f}s]", file=sys.stderr)

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history.clear()
            print("[history cleared]")
            continue
        if user == "/cost":
            print(f"[session] in={total_in} out={total_out} cost={format_cost(total_cost)}")
            continue

        history.append({"role": "user", "content": user})
        history[:] = history[-args.max_history:]
        est = estimate_tokens(" ".join(m["content"] for m in history))
        reply, usage = [], None

        print("bot> ", end="", flush=True)
        try:
            for event in retry_stream(
                lambda: stream_fn(model, history, args.system), on_retry=on_retry
            ):
                if isinstance(event, Usage):
                    usage = event
                else:
                    reply.append(event)
                    print(event, end="", flush=True)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as exc:
            print(f"\n[error: {type(exc).__name__}: {exc}]")
            history.pop()  # drop the unanswered user turn
            continue
        print()

        history.append({"role": "assistant", "content": "".join(reply)})
        if usage:
            cost = calculate_cost(model, usage.input_tokens, usage.output_tokens, pricing)
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_cost += cost or 0.0
            print(f"[tokens in={usage.input_tokens} (est ~{est}) out={usage.output_tokens} "
                  f"| cost={format_cost(cost)}]\n")


if __name__ == "__main__":
    main()
