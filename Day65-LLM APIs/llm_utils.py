"""
Day 65 - LLM APIs (OpenAI / Anthropic)
Shared helpers used by all scripts in this module.

Provides:
    - Safe image loading and base64 encoding
    - Vision calls for both Anthropic and OpenAI
    - Embedding calls (OpenAI) plus an offline, dependency-free local embedder
    - Cosine similarity and top-k retrieval utilities

Model names and API keys are configured through environment variables
(see .env.example) so no secrets or model IDs are hard-coded in the scripts.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # Optional convenience: load variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)


def openai_vision_model() -> str:
    return os.getenv("OPENAI_VISION_MODEL", DEFAULT_OPENAI_VISION_MODEL)


def openai_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)


SUPPORTED_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
IMAGE_EXTENSIONS = tuple(SUPPORTED_MEDIA_TYPES)
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # conservative per-image limit (5 MB)

DEFAULT_DESCRIBE_PROMPT = (
    "Describe this image in 2-3 factual sentences for a search index. "
    "Mention the main objects, the setting, dominant colors, and any visible "
    "text. Do not speculate beyond what is visible."
)


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #

def load_image_base64(image_path: str | Path) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    media_type = SUPPORTED_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        supported = ", ".join(IMAGE_EXTENSIONS)
        raise ValueError(f"Unsupported image type '{path.suffix}'. Supported: {supported}")

    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(raw) / 1_048_576:.1f} MB; the limit used here is "
            f"{MAX_IMAGE_BYTES / 1_048_576:.0f} MB. Resize it before sending."
        )

    return base64.standard_b64encode(raw).decode("utf-8"), media_type


# --------------------------------------------------------------------------- #
# Vision APIs
# --------------------------------------------------------------------------- #

def describe_image_anthropic(
    image_path: str | Path,
    prompt: str = DEFAULT_DESCRIBE_PROMPT,
    model: str | None = None,
    max_tokens: int = 500,
) -> str:
    """Send an image plus a text prompt to the Anthropic Messages API."""
    import anthropic  # imported lazily so other code paths work without the SDK

    data, media_type = load_image_base64(image_path)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = client.messages.create(
        model=model or anthropic_model(),
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def describe_image_openai(
    image_path: str | Path,
    prompt: str = DEFAULT_DESCRIBE_PROMPT,
    model: str | None = None,
    max_tokens: int = 500,
) -> str:
    """Send an image plus a text prompt to the OpenAI Chat Completions API."""
    from openai import OpenAI  # imported lazily

    data, media_type = load_image_base64(image_path)
    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    response = client.chat.completions.create(
        model=model or openai_vision_model(),
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    },
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def describe_image(
    provider: str,
    image_path: str | Path,
    prompt: str = DEFAULT_DESCRIBE_PROMPT,
) -> str:
    """Dispatch to the selected vision provider."""
    if provider == "anthropic":
        return describe_image_anthropic(image_path, prompt)
    if provider == "openai":
        return describe_image_openai(image_path, prompt)
    raise ValueError(f"Unknown vision provider: {provider!r}")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

def embed_texts_openai(
    texts: Sequence[str],
    model: str | None = None,
    batch_size: int = 100,
) -> np.ndarray:
    """Embed texts with the OpenAI embeddings endpoint (batched)."""
    from openai import OpenAI

    client = OpenAI()
    model = model or openai_embedding_model()

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    return np.asarray(vectors, dtype=np.float32)


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to with".split()
)


def embed_texts_local(texts: Sequence[str], dim: int = 512) -> np.ndarray:
    """
    Offline, deterministic embedder based on hashed bag-of-words.

    This is NOT a semantic model: it only matches shared words. It exists so
    the retrieval pipeline can be developed and tested without any embedding
    API key, and to contrast lexical matching with true semantic embeddings.
    """
    vectors = np.zeros((len(texts), dim), dtype=np.float32)

    for row, text in enumerate(texts):
        for token in _TOKEN_PATTERN.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            column = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vectors[row, column] += sign

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(embedder: str, texts: Sequence[str]) -> np.ndarray:
    """Dispatch to the selected embedder ('openai' or 'local')."""
    if embedder == "openai":
        return embed_texts_openai(texts)
    if embedder == "local":
        return embed_texts_local(texts)
    raise ValueError(f"Unknown embedder: {embedder!r}")


# --------------------------------------------------------------------------- #
# Similarity search
# --------------------------------------------------------------------------- #

def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Return the (n_queries, n_corpus) matrix of cosine similarities."""
    return _normalize_rows(np.atleast_2d(queries)) @ _normalize_rows(np.atleast_2d(corpus)).T


def top_k_matches(
    query_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return [(corpus_index, score), ...] for the k most similar vectors."""
    scores = cosine_similarity_matrix(query_vector, corpus_vectors)[0]
    k = min(k, len(scores))
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best]
