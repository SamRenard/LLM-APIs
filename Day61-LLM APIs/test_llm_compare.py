"""Day 61 — Tests using a fake client (no API key or network needed)."""

from types import SimpleNamespace
from llm_compare import ask, compare_temperatures


class FakeClient:
    """Records the arguments of every messages.create() call."""

    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text="hello")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=2),
            stop_reason="end_turn",
        )


def test_ask_returns_text_tokens_and_stop_reason():
    text, tin, tout, stop = ask(FakeClient(), "hi")
    assert (text, tin, tout, stop) == ("hello", 5, 2, "end_turn")
    print("[PASS] ask returns text, token counts, stop_reason")


def test_ask_passes_parameters():
    c = FakeClient()
    ask(c, "hi", system="be brief", temperature=0.3, max_tokens=50)
    call = c.calls[0]
    assert call["system"] == "be brief"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 50
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    print("[PASS] ask passes system, temperature, max_tokens, messages")


def test_optional_parameters_omitted_when_not_set():
    c = FakeClient()
    ask(c, "hi")
    call = c.calls[0]
    assert "system" not in call
    assert "temperature" not in call
    assert "top_p" not in call
    print("[PASS] unset optional parameters are not sent")


def test_temperature_and_top_p_together_rejected():
    try:
        ask(FakeClient(), "hi", temperature=0.5, top_p=0.9)
        assert False, "should have raised"
    except ValueError:
        pass
    print("[PASS] temperature + top_p together raises ValueError")


def test_compare_temperatures_structure():
    c = FakeClient()
    results = compare_temperatures(c, "hi", temps=(0.0, 1.0), runs=3)
    assert set(results) == {0.0, 1.0}
    assert all(len(v) == 3 for v in results.values())
    assert len(c.calls) == 6
    print("[PASS] compare_temperatures runs 3 calls per temperature")


if __name__ == "__main__":
    tests = [
        test_ask_returns_text_tokens_and_stop_reason,
        test_ask_passes_parameters,
        test_optional_parameters_omitted_when_not_set,
        test_temperature_and_top_p_together_rejected,
        test_compare_temperatures_structure,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
