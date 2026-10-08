"""Day 62 — Offline tests (fake client, no API key or network needed)."""

from types import SimpleNamespace
from prompt_compare import (
    build_prompt, parse_label, classify, compare_strategies,
    STRATEGIES, REVIEWS, FEW_SHOT_EXAMPLES,
)


class FakeClient:
    """Always replies with a fixed text; records every request."""

    def __init__(self, reply="positive"):
        self.reply = reply
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


def test_few_shot_prompt_contains_all_examples():
    _, user = build_prompt("few_shot", "some review")
    for review, label in FEW_SHOT_EXAMPLES:
        assert review in user and label in user
    print("[PASS] few-shot prompt includes every example")


def test_chain_of_thought_asks_for_step_by_step():
    _, user = build_prompt("chain_of_thought", "x")
    assert "step by step" in user and "Answer:" in user
    print("[PASS] chain-of-thought prompt asks for reasoning + final Answer line")


def test_role_strategy_sets_system_prompt_others_do_not():
    system, _ = build_prompt("role", "x")
    assert system and "analyst" in system
    for s in ["zero_shot", "few_shot", "chain_of_thought", "structured"]:
        assert build_prompt(s, "x")[0] is None
    print("[PASS] only the role strategy uses a system prompt")


def test_unknown_strategy_raises():
    try:
        build_prompt("magic", "x")
        assert False, "should raise"
    except ValueError:
        pass
    print("[PASS] unknown strategy raises ValueError")


def test_parse_label_variants():
    assert parse_label("positive") == "positive"
    assert parse_label("Sentiment: Negative.") == "negative"
    # reasoning mentions other labels first -> explicit Answer line wins
    assert parse_label("Not positive, tone is harsh.\nAnswer: negative") == "negative"
    # no Answer line -> last label word wins
    assert parse_label("Could be positive, but overall neutral") == "neutral"
    assert parse_label("I don't know") is None
    print("[PASS] parse_label handles plain, Answer-line, and multi-label replies")


def test_classify_uses_temperature_zero_and_returns_label():
    c = FakeClient("negative")
    assert classify(c, "zero_shot", "bad") == "negative"
    assert c.calls[0]["temperature"] == 0.0
    print("[PASS] classify sends temperature=0 and parses the reply")


def test_compare_strategies_scores_correctly():
    # Fake model always says "positive": only the 1 positive review is right.
    c = FakeClient("positive")
    results = compare_strategies(c)
    assert set(results) == set(STRATEGIES)
    expected_correct = sum(1 for _, gold in REVIEWS if gold == "positive")
    for r in results.values():
        assert r["correct"] == expected_correct
        assert abs(r["accuracy"] - expected_correct / len(REVIEWS)) < 1e-9
    assert len(c.calls) == len(STRATEGIES) * len(REVIEWS)
    print("[PASS] compare_strategies scores every strategy against gold labels")


if __name__ == "__main__":
    tests = [
        test_few_shot_prompt_contains_all_examples,
        test_chain_of_thought_asks_for_step_by_step,
        test_role_strategy_sets_system_prompt_others_do_not,
        test_unknown_strategy_raises,
        test_parse_label_variants,
        test_classify_uses_temperature_zero_and_returns_label,
        test_compare_strategies_scores_correctly,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
