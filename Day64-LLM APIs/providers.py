"""Streaming adapters for OpenAI and Anthropic.

Each adapter yields text chunks (str) and finally one Usage object.
"""
from dataclasses import dataclass
from typing import Iterator, List, Union


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


Event = Union[str, Usage]


def stream_openai(model: str, messages: List[dict], system: str) -> Iterator[Event]:
    from openai import OpenAI

    client = OpenAI(max_retries=0)  # we handle retries ourselves
    full = [{"role": "system", "content": system}] + messages
    stream = client.chat.completions.create(
        model=model,
        messages=full,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if chunk.usage:
            yield Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)


def stream_anthropic(model: str, messages: List[dict], system: str) -> Iterator[Event]:
    import anthropic

    client = anthropic.Anthropic(max_retries=0)
    with client.messages.stream(
        model=model, max_tokens=1024, system=system, messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield text
        usage = stream.get_final_message().usage
        yield Usage(usage.input_tokens, usage.output_tokens)


PROVIDERS = {"openai": stream_openai, "anthropic": stream_anthropic}
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5-5"}
