"""
Day 65 - LLM APIs (OpenAI / Anthropic)
Module 2: Embedding APIs

Covers:
    - Turning text into dense vectors with an embedding API
    - Cosine similarity as the standard closeness measure
    - Ranking a small corpus against a query
    - Semantic (OpenAI) vs. lexical (local hashed bag-of-words) matching

Note on providers:
    Anthropic does not offer its own embedding model; its documentation points
    to third-party providers (for example Voyage AI). OpenAI provides an
    embeddings endpoint, so this script uses OpenAI for the semantic path and
    a local lexical embedder as an offline baseline.

Usage:
    python 02_embeddings_basics.py                    # offline, local embedder
    python 02_embeddings_basics.py --embedder openai  # requires OPENAI_API_KEY
    python 02_embeddings_basics.py --query "a pet sleeping"
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from llm_utils import cosine_similarity_matrix, embed_texts, top_k_matches

CORPUS = [
    "A golden retriever runs across a grassy park.",
    "A kitten sleeps on a sofa in a sunny living room.",
    "A red sports car is parked beside a modern building.",
    "A bowl of pasta with tomato sauce on a wooden table.",
    "Snow-covered mountains reflected in a calm lake at sunrise.",
    "A cyclist rides along a coastal road near the sea.",
]

DEFAULT_QUERY = "a cat resting indoors"


def print_similarity_matrix(texts: list[str], vectors: np.ndarray) -> None:
    matrix = cosine_similarity_matrix(vectors, vectors)
    labels = [f"T{i}" for i in range(len(texts))]

    print("\nTexts:")
    for label, text in zip(labels, texts):
        print(f"  {label}: {text}")

    print("\nPairwise cosine similarity:")
    print("      " + "".join(f"{label:>7}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"{label:>5} " + "".join(f"{value:>7.2f}" for value in row))


def rank_corpus(query: str, embedder: str, texts: list[str], vectors: np.ndarray) -> None:
    query_vector = embed_texts(embedder, [query])
    matches = top_k_matches(query_vector, vectors, k=len(texts))

    print(f'\nQuery: "{query}"')
    print("Ranked results:")
    for rank, (index, score) in enumerate(matches, start=1):
        print(f"  {rank}. score={score:+.3f}  {texts[index]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore text embeddings and similarity.")
    parser.add_argument("--embedder", choices=["local", "openai"], default="local")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args()

    print(f"Embedder: {args.embedder}")

    try:
        vectors = embed_texts(args.embedder, CORPUS)
    except Exception as error:
        print(f"Embedding request failed: {type(error).__name__}: {error}")
        return 1

    print(f"Embedding matrix shape: {vectors.shape}  (texts x dimensions)")

    print_similarity_matrix(CORPUS, vectors)
    rank_corpus(args.query, args.embedder, CORPUS, vectors)

    if args.embedder == "local":
        print(
            "\nNote: the local embedder only matches shared words, so a query like "
            f'"{DEFAULT_QUERY}" cannot connect "cat" with "kitten". '
            "Run with --embedder openai to see semantic matching in action."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
