"""Day 63 — Offline tests: tools + agent loop with a scripted fake client."""

from types import SimpleNamespace as NS
from tool_agent import (
    calculator, get_weather, run_tool, run_agent, extract_structured, TOOL_SCHEMAS,
)


def text_resp(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


def tool_resp(*calls):
    """calls: (id, name, input) tuples -> a response asking to use tools."""
    return NS(stop_reason="tool_use",
              content=[NS(type="tool_use", id=i, name=n, input=a) for i, n, a in calls])


class ScriptedClient:
    """Returns pre-written responses in order; records a snapshot of each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        snap = dict(kwargs)
        if "messages" in snap:
            snap["messages"] = list(snap["messages"])
        self.calls.append(snap)
        return self.responses.pop(0)


# ---- tools ----

def test_calculator_values():
    assert calculator("12 * (3 + 4)") == "84"
    assert calculator("10 / 4") == "2.5"
    assert calculator("-3 + 5") == "2"
    assert calculator("2 ** 10") == "1024"
    print("[PASS] calculator computes correct values")


def test_calculator_rejects_code_and_huge_powers():
    for bad in ["__import__('os').system('ls')", "abs(-1)", "x + 1", "9 ** 9 ** 9"]:
        out, is_err = run_tool("calculator", {"expression": bad})
        assert is_err, bad
    print("[PASS] calculator rejects code execution, names, huge exponents")


def test_calculator_division_by_zero_is_reported():
    out, is_err = run_tool("calculator", {"expression": "1 / 0"})
    assert is_err and out.startswith("error:")
    print("[PASS] division by zero returns an error result, not a crash")


def test_weather_known_unknown_and_case():
    assert get_weather("Baku") == "Baku: 22°C, partly cloudy"
    assert get_weather("  LONDON ") == "London: 14°C, rain"
    out, is_err = run_tool("get_weather", {"city": "Atlantis"})
    assert is_err and "no data" in out
    print("[PASS] weather: known city, case-insensitive, unknown city -> error")


def test_unknown_tool_is_error():
    out, is_err = run_tool("hack", {})
    assert is_err and "unknown tool" in out
    print("[PASS] unknown tool name returns an error result")


# ---- agent loop ----

def test_agent_runs_tool_then_answers():
    c = ScriptedClient([
        tool_resp(("t1", "calculator", {"expression": "12 * (3 + 4)"})),
        text_resp("The answer is 84."),
    ])
    answer, trace = run_agent(c, "What is 12 * (3 + 4)?")
    assert answer == "The answer is 84."
    assert trace == [{"tool": "calculator", "input": {"expression": "12 * (3 + 4)"},
                      "output": "84", "is_error": False}]
    assert c.calls[0]["tools"] == TOOL_SCHEMAS
    # second request must carry the tool_result tied to the right tool_use id
    result_msg = c.calls[1]["messages"][-1]
    assert result_msg["role"] == "user"
    assert result_msg["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "84", "is_error": False}
    ]
    print("[PASS] agent: tool_use -> tool_result (matching id) -> final answer")


def test_agent_handles_multiple_tool_calls_in_one_step():
    c = ScriptedClient([
        tool_resp(("a", "get_weather", {"city": "Tokyo"}),
                  ("b", "get_weather", {"city": "London"})),
        text_resp("Tokyo is 5 degrees warmer."),
    ])
    answer, trace = run_agent(c, "Tokyo vs London?")
    assert [t["output"] for t in trace] == ["Tokyo: 19°C, clear", "London: 14°C, rain"]
    ids = [r["tool_use_id"] for r in c.calls[1]["messages"][-1]["content"]]
    assert ids == ["a", "b"]
    print("[PASS] agent returns one tool_result per tool_use in the same step")


def test_agent_passes_tool_errors_to_model():
    c = ScriptedClient([
        tool_resp(("e1", "get_weather", {"city": "Atlantis"})),
        text_resp("Sorry, I have no data for Atlantis."),
    ])
    answer, trace = run_agent(c, "Weather in Atlantis?")
    assert trace[0]["is_error"] is True
    sent = c.calls[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "no data" in sent["content"]
    print("[PASS] tool errors are sent back to the model with is_error=True")


def test_agent_stops_after_max_steps():
    looping = [tool_resp((f"x{i}", "calculator", {"expression": "1+1"})) for i in range(10)]
    c = ScriptedClient(looping)
    answer, trace = run_agent(c, "loop forever", max_steps=3)
    assert answer == "Stopped: too many tool steps."
    assert len(c.calls) == 3 and len(trace) == 3
    print("[PASS] agent loop is capped by max_steps")


# ---- structured output ----

def test_extract_structured_forces_tool_and_returns_dict():
    data = {"name": "Leyla Aliyeva", "email": "leyla@example.com", "city": "Ganja"}
    c = ScriptedClient([NS(stop_reason="tool_use",
                           content=[NS(type="tool_use", id="s", name="record_contact", input=data)])])
    out = extract_structured(c, "Reach Leyla ...")
    assert out == data
    assert c.calls[0]["tool_choice"] == {"type": "tool", "name": "record_contact"}
    print("[PASS] extract_structured forces the schema tool and returns its JSON")


if __name__ == "__main__":
    tests = [
        test_calculator_values,
        test_calculator_rejects_code_and_huge_powers,
        test_calculator_division_by_zero_is_reported,
        test_weather_known_unknown_and_case,
        test_unknown_tool_is_error,
        test_agent_runs_tool_then_answers,
        test_agent_handles_multiple_tool_calls_in_one_step,
        test_agent_passes_tool_errors_to_model,
        test_agent_stops_after_max_steps,
        test_extract_structured_forces_tool_and_returns_dict,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
