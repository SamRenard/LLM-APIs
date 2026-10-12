"""Token estimation and cost calculation."""
import json
from pathlib import Path
from typing import Optional

PRICING_PATH = Path(__file__).resolve().parent.parent / "pricing.json"


def load_pricing(path: Path = PRICING_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int, pricing: dict
) -> Optional[float]:
    """Return cost in USD, or None if the model has no pricing entry."""
    entry = pricing.get(model)
    if entry is None:
        return None
    return (
        input_tokens / 1_000_000 * entry["input_per_1m"]
        + output_tokens / 1_000_000 * entry["output_per_1m"]
    )


def estimate_tokens(text: str) -> int:
    """Pre-flight estimate. Exact usage always comes from the API response."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)
