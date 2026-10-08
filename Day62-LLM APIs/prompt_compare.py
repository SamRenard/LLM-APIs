"""
Day 62 — Prompt comparison tool: same task, 5 strategies, side by side
NIZAM AI · 150-Day AI Engineering Protocol

Task: classify short customer reviews as positive / negative / neutral.
Each strategy builds a different prompt for the same review; we score
every strategy against known correct labels and print a comparison.

Setup:  export ANTHROPIC_API_KEY="your-key"
Run:    python3 prompt_compare.py
"""

import os
import re
import sys

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
LABELS = ("positive", "negative", "neutral")

# Small labeled dataset (review, correct label). Includes tricky cases:
# sarcasm and mixed feelings, where prompt strategy tends to matter.
REVIEWS = [
    ("Absolutely love it, works perfectly!", "positive"),
    ("Broke after two days. Total waste of money.", "negative"),
    ("It arrived on Tuesday in a brown box.", "neutral"),
    ("Oh great, another charger that dies in a week. Just what I needed.", "negative"),
    ("Battery is amazing but the screen scratches way too easily.", "neutral"),
    ("Does what it says. Nothing special, nothing wrong.", "neutral"),
]

FEW_SHOT_EXAMPLES = [
    ("Fast delivery and great quality, very happy.", "positive"),
    ("Stopped working immediately, want a refund.", "negative"),
    ("The package contained one cable.", "neutral"),
]

TASK = "Classify the sentiment of this customer review as positive, negative, or neutral."
FORMAT_RULE = "Reply with exactly one word: positive, negative, or neutral."


def build_prompt(strategy, review):
    """Return (system_prompt_or_None, user_prompt) for a strategy."""
    if strategy == "zero_shot":
        return None, f"{TASK}\n\nReview: {review}"

    if strategy == "role":
        system = "You are an experienced customer-feedback analyst."
        return system, f"{TASK}\n\nReview: {review}"

    if strategy == "few_shot":
        shots = "\n".join(f"Review: {r}\nSentiment: {l}\n" for r, l in FEW_SHOT_EXAMPLES)
        return None, f"{TASK}\n\nExamples:\n{shots}\nReview: {review}\nSentiment:"

    if strategy == "chain_of_thought":
        return None, (
            f"{TASK}\n\nReview: {review}\n\n"
            "Think step by step about the tone first. "
            "Then finish with a final line in the form: Answer: <label>"
        )

    if strategy == "structured":
        return None, f"{TASK}\n\nReview: {review}\n\n{FORMAT_RULE}"

    raise ValueError(f"Unknown strategy: {strategy}")


STRATEGIES = ["zero_shot", "role", "few_shot", "chain_of_thought", "structured"]


def parse_label(text):
    """Extract the predicted label from a model reply.

    Prefers an explicit 'Answer: <label>' line (used by chain-of-thought);
    otherwise takes the LAST label word that appears, since reasoning text
    may mention other labels before the final decision.
    """
    lowered = text.lower()
    match = re.search(r"answer:\s*(positive|negative|neutral)", lowered)
    if match:
        return match.group(1)
    found = re.findall(r"positive|negative|neutral", lowered)
    return found[-1] if found else None


def classify(client, strategy, review, max_tokens=300):
    system, user = build_prompt(strategy, review)
    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return parse_label(resp.content[0].text)


def compare_strategies(client, reviews=REVIEWS, strategies=STRATEGIES):
    """Run every strategy on every review. Returns {strategy: {...}}."""
    results = {}
    for strategy in strategies:
        preds = [classify(client, strategy, review) for review, _ in reviews]
        correct = sum(p == gold for p, (_, gold) in zip(preds, reviews))
        results[strategy] = {
            "predictions": preds,
            "correct": correct,
            "accuracy": correct / len(reviews),
        }
    return results


def print_table(results, reviews=REVIEWS):
    names = list(results)
    print(f"{'#':<3}{'gold':<10}" + "".join(f"{n:<18}" for n in names))
    for i, (_, gold) in enumerate(reviews):
        row = f"{i + 1:<3}{gold:<10}"
        for n in names:
            pred = results[n]["predictions"][i]
            mark = "ok " if pred == gold else "ERR"
            row += f"{str(pred):<14}{mark} "
        print(row)
    print("-" * (13 + 18 * len(names)))
    print(f"{'acc':<13}" + "".join(f"{results[n]['accuracy']:<18.2f}" for n in names))


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    from anthropic import Anthropic
    client = Anthropic()

    print(f"Model: {MODEL}  |  {len(REVIEWS)} reviews x {len(STRATEGIES)} strategies\n")
    results = compare_strategies(client)
    print_table(results)


if __name__ == "__main__":
    main()
