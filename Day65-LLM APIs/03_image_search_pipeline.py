"""
Day 65 - LLM APIs (OpenAI / Anthropic)
Module 3: Practical Pipeline - Image -> LLM Description -> Search Over Descriptions

Pipeline:
    1. INDEX : for every image in a folder, ask a vision LLM for a description
    2. EMBED : embed all descriptions into vectors
    3. STORE : save descriptions + vectors to a JSON index file
    4. SEARCH: embed a text query and return the most similar images

Usage:
    # Build the index (Anthropic vision + OpenAI embeddings)
    python 03_image_search_pipeline.py index --images-dir ./images \
        --index-file index.json --vision anthropic --embedder openai

    # Build the index with only an Anthropic key (lexical local embedder)
    python 03_image_search_pipeline.py index --images-dir ./images \
        --index-file index.json --vision anthropic --embedder local

    # Search the index
    python 03_image_search_pipeline.py search --index-file index.json \
        --query "a dog outdoors" --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from llm_utils import IMAGE_EXTENSIONS, describe_image, embed_texts, top_k_matches

INDEX_VERSION = 1


def find_images(images_dir: Path) -> list[Path]:
    """Return supported image files in a directory, sorted for stable output."""
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_index(images_dir: Path, index_file: Path, vision: str, embedder: str) -> int:
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}")
        return 1

    images = find_images(images_dir)
    if not images:
        print(f"No supported images found in {images_dir} ({', '.join(IMAGE_EXTENSIONS)})")
        return 1

    print(f"Found {len(images)} image(s). Describing with '{vision}'...")

    records: list[dict] = []
    for position, image_path in enumerate(images, start=1):
        try:
            description = describe_image(vision, image_path)
        except Exception as error:  # keep going if a single image fails
            print(f"  [{position}/{len(images)}] SKIPPED {image_path.name}: "
                  f"{type(error).__name__}: {error}")
            continue

        records.append({"image": str(image_path), "description": description})
        print(f"  [{position}/{len(images)}] {image_path.name}: {description[:80]}...")

    if not records:
        print("No images could be described; index not written.")
        return 1

    print(f"Embedding {len(records)} description(s) with '{embedder}'...")
    vectors = embed_texts(embedder, [record["description"] for record in records])

    for record, vector in zip(records, vectors):
        record["embedding"] = [round(float(value), 6) for value in vector]

    payload = {
        "version": INDEX_VERSION,
        "vision": vision,
        "embedder": embedder,
        "records": records,
    }
    index_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Index saved to {index_file} ({len(records)} records).")
    return 0


def search_index(index_file: Path, query: str, top_k: int) -> int:
    if not index_file.is_file():
        print(f"Index file not found: {index_file}. Run the 'index' command first.")
        return 1

    payload = json.loads(index_file.read_text(encoding="utf-8"))
    records = payload["records"]
    embedder = payload["embedder"]  # the query MUST use the same embedder as the index

    corpus = np.asarray([record["embedding"] for record in records], dtype=np.float32)
    query_vector = embed_texts(embedder, [query])

    print(f'Query: "{query}"  (embedder: {embedder})\n')
    for rank, (index, score) in enumerate(top_k_matches(query_vector, corpus, top_k), start=1):
        record = records[index]
        print(f"{rank}. {record['image']}  (score {score:+.3f})")
        print(f"   {record['description']}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Image search over LLM-generated descriptions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Describe images and build the index")
    index_parser.add_argument("--images-dir", type=Path, required=True)
    index_parser.add_argument("--index-file", type=Path, default=Path("index.json"))
    index_parser.add_argument("--vision", choices=["anthropic", "openai"], default="anthropic")
    index_parser.add_argument("--embedder", choices=["openai", "local"], default="local")

    search_parser = subparsers.add_parser("search", help="Search an existing index")
    search_parser.add_argument("--index-file", type=Path, default=Path("index.json"))
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "index":
        return build_index(args.images_dir, args.index_file, args.vision, args.embedder)
    return search_index(args.index_file, args.query, args.top_k)


if __name__ == "__main__":
    sys.exit(main())
