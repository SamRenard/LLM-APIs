"""
Day 65 - LLM APIs (OpenAI / Anthropic)
Module 1: Multimodal Capabilities - Image Analysis (Vision)

Covers:
    - Sending an image to an LLM as base64 data
    - How the request format differs between Anthropic and OpenAI
    - Free-form description vs. structured (JSON) extraction
    - Defensive parsing of model output

Usage:
    python 01_vision_api_basics.py path/to/image.jpg
    python 01_vision_api_basics.py path/to/image.png --provider openai
    python 01_vision_api_basics.py path/to/image.png --provider both --structured
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from llm_utils import (
    DEFAULT_DESCRIBE_PROMPT,
    describe_image_anthropic,
    describe_image_openai,
)

STRUCTURED_PROMPT = (
    "Analyze this image and respond with ONLY a valid JSON object, with no "
    "markdown fences and no extra commentary. Use exactly these keys:\n"
    '  "scene": short string describing the overall setting,\n'
    '  "objects": list of the main visible objects,\n'
    '  "dominant_colors": list of up to 3 color names,\n'
    '  "visible_text": string with any readable text, or null if none.'
)


def parse_json_response(text: str) -> dict | None:
    """Parse JSON from a model response, tolerating accidental code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_provider(name: str, image_path: str, prompt: str, structured: bool) -> None:
    print("\n" + "=" * 60)
    print(f"Provider: {name}")
    print("=" * 60)

    describe = describe_image_anthropic if name == "anthropic" else describe_image_openai

    try:
        response_text = describe(image_path, prompt)
    except Exception as error:  # surface API/config problems clearly
        print(f"Request failed: {type(error).__name__}: {error}")
        return

    if structured:
        parsed = parse_json_response(response_text)
        if parsed is not None:
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
            return
        print("Could not parse JSON; showing raw response instead:")

    print(response_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze an image with an LLM vision API.")
    parser.add_argument("image_path", help="Path to a JPG, PNG, GIF, or WEBP image")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "both"],
        default="anthropic",
        help="Which API to call (default: anthropic)",
    )
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides the default)")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Ask for a JSON object and parse it instead of free-form text",
    )
    args = parser.parse_args()

    prompt = args.prompt or (STRUCTURED_PROMPT if args.structured else DEFAULT_DESCRIBE_PROMPT)
    providers = ["anthropic", "openai"] if args.provider == "both" else [args.provider]

    for provider in providers:
        run_provider(provider, args.image_path, prompt, args.structured)

    return 0


if __name__ == "__main__":
    sys.exit(main())
