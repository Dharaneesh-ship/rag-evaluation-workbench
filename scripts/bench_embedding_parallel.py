#!/usr/bin/env python3
"""Benchmark: OpenAIEmbeddingProvider batch concurrency, sequential vs parallel.

Embeds the same set of texts twice — once forcing sequential batches
(``max_workers=1``) and once with the default concurrent thread pool — and
reports the wall-clock difference. Unlike Vision-LLM transcription
(rate-limited, ~$0.01-0.03/page), OpenAI's embedding API is cheap enough
(``text-embedding-3-small``, $0.02/1M tokens) that this is a real, live,
unsimulated benchmark rather than one calibrated from a single measurement.

Requires OPENAI_API_KEY. Total cost for the default settings is a fraction
of a cent.

Usage:
    python scripts/bench_embedding_parallel.py
    python scripts/bench_embedding_parallel.py --n-texts 1280 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _make_texts(n: int) -> list[str]:
    """Synthetic but non-trivial-length texts (benchmarking latency, not
    embedding quality, so content doesn't need to be real corpus text)."""
    topics = [
        "distributed systems", "retrieval augmented generation", "hash indexes",
        "concurrency control", "vector databases", "language models",
        "information retrieval", "database indexing", "network protocols",
        "cache coherence",
    ]
    return [
        f"This is benchmark sentence number {i}, discussing {topics[i % len(topics)]} "
        f"and its practical implications for large-scale information systems."
        for i in range(n)
    ]


def _run(texts: list[str], batch_size: int, max_workers: int) -> tuple[float, int]:
    from dotenv import load_dotenv

    load_dotenv()
    from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-3-small", dim=1536,
        batch_size=batch_size, max_workers=max_workers,
    )

    start = time.monotonic()
    result = provider.embed_with_usage(texts)
    elapsed = time.monotonic() - start

    return elapsed, len(result.vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-texts", type=int, default=1280, help="Number of texts to embed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Texts per API call.")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent batches for the 'after' run.")
    args = parser.parse_args()

    texts = _make_texts(args.n_texts)
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(f"Texts: {len(texts)}, batch_size={args.batch_size} -> {n_batches} API calls")

    t_before, n_before = _run(texts, args.batch_size, max_workers=1)
    print(f"[BEFORE] max_workers=1 (sequential): {t_before:.1f}s for {n_before} vectors")

    t_after, n_after = _run(texts, args.batch_size, max_workers=args.max_workers)
    print(f"[AFTER]  max_workers={args.max_workers} (concurrent):  {t_after:.1f}s for {n_after} vectors")

    print(f"\nSpeedup: {t_before / t_after:.1f}x")
    print(f"Time saved: {t_before - t_after:.1f}s ({(1 - t_after / t_before) * 100:.0f}% reduction)")


if __name__ == "__main__":
    main()
