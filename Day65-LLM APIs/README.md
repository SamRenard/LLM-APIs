# Day 65 — LLM APIs (OpenAI / Anthropic): Vision & Embeddings

Part of the *NIZAM AI — 150-Day AI Engineering Protocol*, Month 3
("LLM & RAG Systems"), Block Day 5 of 5.

## Overview

This module covers the multimodal side of LLM APIs: sending images to a
vision-capable model, and converting text into embedding vectors for
similarity search. It closes with a practical pipeline that ties both
together: **image → LLM description → semantic search over descriptions**.

## Learning Objectives

By the end of this module you should be able to:

- Send an image to the Anthropic and OpenAI APIs as base64 data and explain
  how the two request formats differ
- Request structured (JSON) output from a vision model and parse it defensively
- Explain what an embedding is and why cosine similarity is the standard
  closeness measure
- Distinguish semantic embeddings from lexical (word-overlap) matching
- Build an index → embed → store → search pipeline and understand why a query
  must be embedded with the same model as the index

## File Structure

```
day65_llm_apis/
├── llm_utils.py                 # Shared helpers: image encoding, vision, embeddings, similarity
├── 01_vision_api_basics.py      # Image analysis via Anthropic / OpenAI (free-form or JSON)
├── 02_embeddings_basics.py      # Embeddings, similarity matrix, ranking
├── 03_image_search_pipeline.py  # Image -> description -> embedding -> search (index/search CLI)
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for API keys and optional model overrides
├── .gitignore                   # Keeps secrets and generated indexes out of Git
└── README.md                    # This file
```

## Requirements

- Python 3.9+
- An Anthropic API key and/or an OpenAI API key (see the matrix below)

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
```

| Feature                               | Key required                           |
|----------------------------------------|-----------------------------------------|
| Vision with Anthropic                   | `ANTHROPIC_API_KEY`                      |
| Vision with OpenAI                       | `OPENAI_API_KEY`                          |
| Semantic embeddings (`--embedder openai`) | `OPENAI_API_KEY`                          |
| Local lexical embeddings (`--embedder local`) | none (offline)                         |

Anthropic does not provide its own embedding model, which is why the semantic
embedding path uses OpenAI. Model names can be changed through the optional
variables in `.env.example`; check each provider's documentation for the
currently available models.

## Usage

**1. Vision basics**

```bash
python 01_vision_api_basics.py photo.jpg
python 01_vision_api_basics.py photo.jpg --provider both --structured
```

**2. Embeddings basics**

```bash
python 02_embeddings_basics.py                       # offline baseline
python 02_embeddings_basics.py --embedder openai --query "a cat resting indoors"
```

**3. Image search pipeline**

```bash
# Build the index
python 03_image_search_pipeline.py index --images-dir ./images \
    --index-file index.json --vision anthropic --embedder openai

# Search it
python 03_image_search_pipeline.py search --index-file index.json \
    --query "a dog outdoors" --top-k 3
```

The index file records which embedder built it, and `search` automatically
uses the same one for the query.

## Design Notes

- **No hard-coded secrets or models:** keys come from the environment; models
  are overridable through environment variables.
- **Lazy SDK imports:** the offline path works without either SDK installed.
- **Input validation:** unsupported formats and oversized images fail early with
  clear messages instead of opaque API errors.
- **Resilient indexing:** one failing image is skipped and reported rather than
  aborting the whole run.
- **Lexical vs. semantic:** the local embedder is a deliberate baseline. It
  cannot link "cat" with "kitten", which makes the value of real embeddings
  easy to see.

## Key Concepts Reference

| Concept              | Idea                                                         |
|------------------------|----------------------------------------------------------------|
| Vision input             | Image sent as base64 alongside a text prompt                    |
| Structured output         | Prompt for JSON only, then parse defensively                     |
| Embedding                  | Dense vector representing the meaning of a text                   |
| Cosine similarity            | Angle-based closeness between two vectors (range -1 to 1)          |
| Retrieval                      | Rank stored vectors by similarity to the query vector               |
| Same-model rule                  | Query and index vectors must come from the same embedding model      |

## Suggested Workflow (matches the daily plan)

1. **Deep theory (2h):** Read the vision and embeddings sections of the
   Anthropic and OpenAI API docs, then run scripts 01 and 02.
2. **Practical coding (2h):** Put 5–10 varied images in a folder, build the
   index with script 03, and test queries that need semantic understanding.
3. **Consolidation:** Write a 5–10 sentence summary in your own words plus
   the key terms below (second-brain habit).
4. **Commit:** Push this folder to GitHub (without `.env`), wait 5 minutes,
   re-read your own code, and improve one thing before finalizing.

## Key Terms

`multimodal`, `vision`, `base64`, `embedding`, `cosine similarity`,
`semantic search`, `index`, `top-k retrieval`, `structured output`

## Reference Sources

- Anthropic API Docs
- OpenAI API Docs
- Prompting Guide

## Progress

Day 65 / 150 — Streak: 1 🔥 — Overall completion: 43%
