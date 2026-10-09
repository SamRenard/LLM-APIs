"""
Day 63 — Structured output & tool calling: a mini agent
NIZAM AI · 150-Day AI Engineering Protocol

Two tools (weather lookup, calculator) plus the agent loop:
  1. send the question + tool schemas to the model
  2. if the model replies with tool_use blocks, run those tools ourselves
  3. send the results back as tool_result blocks, repeat until plain text

Also includes extract_structured(): forcing JSON output via a tool schema.

Setup:  export ANTHROPIC_API_KEY="your-key"
Run:    python3 tool_agent.py
"""

import ast
import operator
import os
import sys

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

# STUB DATA: fixed sample values, NOT live weather. Swap get_weather()
# for a real weather API call to make it real.
SAMPLE_WEATHER = {
    "baku": (22, "partly cloudy"),
    "london": (14, "rain"),
    "tokyo": (19, "clear"),
}

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def _eval_node(node):
    """Evaluate a parsed arithmetic expression. Only numbers and + - * / % **
    are allowed, so this can never run arbitrary code (unlike eval())."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def calculator(expression):
    result = _eval_node(ast.parse(expression, mode="eval").body)
    return str(int(result)) if float(result).is_integer() else str(round(result, 6))


def get_weather(city):
    key = city.strip().lower()
    if key not in SAMPLE_WEATHER:
        raise ValueError(f"no data for city '{city}'")
    temp, cond = SAMPLE_WEATHER[key]
    return f"{city.strip().title()}: {temp}°C, {cond}"


TOOL_FUNCTIONS = {"calculator": calculator, "get_weather": get_weather}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression using + - * / % ** and parentheses.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. 12 * (3 + 4)"}},
            "required": ["expression"],
        },
    },
]


def run_tool(name, args):
    """Run one tool. Returns (result_text, is_error). Errors are returned,
    not raised, so the model can see them and recover."""
    if name not in TOOL_FUNCTIONS:
        return f"unknown tool '{name}'", True
    try:
        return TOOL_FUNCTIONS[name](**args), False
    except Exception as e:  # tool failures are data for the model, not crashes
        return f"error: {e}", True


# ---------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------

def run_agent(client, user_message, max_steps=5):
    """Returns (final_text, trace). trace lists every tool call made."""
    messages = [{"role": "user", "content": user_message}]
    trace = []

    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, tools=TOOL_SCHEMAS, messages=messages
        )

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, trace

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                output, is_error = run_tool(block.name, block.input)
                trace.append({"tool": block.name, "input": block.input,
                              "output": output, "is_error": is_error})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    return "Stopped: too many tool steps.", trace


# ---------------------------------------------------------------------
# Structured output via a forced tool
# ---------------------------------------------------------------------

CONTACT_SCHEMA = {
    "name": "record_contact",
    "description": "Record contact details found in the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "city": {"type": "string"},
        },
        "required": ["name"],
    },
}


def extract_structured(client, text):
    """Force the model to answer by filling CONTACT_SCHEMA -> returns a dict."""
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        tools=[CONTACT_SCHEMA],
        tool_choice={"type": "tool", "name": "record_contact"},
        messages=[{"role": "user", "content": f"Extract the contact details:\n\n{text}"}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    return None


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    from anthropic import Anthropic
    client = Anthropic()

    for q in [
        "What's the weather in Baku?",
        "What is 12 * (3 + 4)?",
        "Is it warmer in Tokyo or London? Give the difference in degrees.",
    ]:
        answer, trace = run_agent(client, q)
        print(f"Q: {q}")
        for t in trace:
            print(f"   tool: {t['tool']}({t['input']}) -> {t['output']}")
        print(f"A: {answer}\n")

    print(extract_structured(client, "Reach Leyla Aliyeva at leyla@example.com, she lives in Ganja."))


if __name__ == "__main__":
    main()
