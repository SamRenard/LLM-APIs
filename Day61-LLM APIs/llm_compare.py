"""
Day 61 — First LLM API script: chat completion + parameter comparison
NIZAM AI · 150-Day AI Engineering Protocol

Calls the Anthropic Messages API and compares how temperature, system
prompt, and max_tokens change the result.

Setup:  export ANTHROPIC_API_KEY="your-key"
Run:    python3 llm_compare.py
"""

import os
import sys

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")


def ask(client, prompt, *, system=None, temperature=None, top_p=None, max_tokens=150):
    """Send one prompt, return (text, input_tokens, output_tokens, stop_reason).

    Set temperature OR top_p, not both -- some models reject the combination.
    """
    if temperature is not None and top_p is not None:
        raise ValueError("Set either temperature or top_p, not both.")

    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    resp = client.messages.create(**kwargs)
    return (
        resp.content[0].text,
        resp.usage.input_tokens,
        resp.usage.output_tokens,
        resp.stop_reason,
    )


def compare_temperatures(client, prompt, temps=(0.0, 1.0), runs=2):
    """Run the same prompt `runs` times at each temperature."""
    results = {}
    for t in temps:
        results[t] = [ask(client, prompt, temperature=t) for _ in range(runs)]
    return results


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    from anthropic import Anthropic
    client = Anthropic()

    prompt = "Give me a name for a coffee shop run by robots. One name only."

    print("=" * 60)
    print("1) Temperature: same prompt, 2 runs each")
    print("=" * 60)
    for t, runs in compare_temperatures(client, prompt).items():
        for i, (text, tin, tout, _) in enumerate(runs, 1):
            print(f"temp={t}  run {i}: {text.strip()!r}  (in={tin}, out={tout} tokens)")
    print("-> temp 0 repeats itself; temp 1 varies.\n")

    print("=" * 60)
    print("2) System prompt changes the style")
    print("=" * 60)
    q = "Explain what a neural network is."
    for system in [None, "You are a pirate. Answer in 2 sentences."]:
        text, *_ = ask(client, q, system=system, temperature=0.0)
        print(f"system={system!r}\n  {text.strip()}\n")

    print("=" * 60)
    print("3) max_tokens cuts the answer off")
    print("=" * 60)
    text, tin, tout, stop = ask(client, "Write a long story about a dragon.", max_tokens=20)
    print(f"output tokens={tout}, stop_reason={stop!r}")
    print(f"text: {text.strip()!r}")
    print("-> stop_reason 'max_tokens' means the reply was truncated.")


if __name__ == "__main__":
    main()
