#!/usr/bin/env python3
"""ASQA retrieval pilot — human-labeled recall@k eval on a real HTML corpus.

Unlike BEIR (pre-chunked passages), ASQA (https://huggingface.co/din0s/asqa)
ships only human-labeled question/answer pairs plus the Wikipedia pages the
answers came from — no pre-made corpus. This script builds one:

1. Sample N examples from the ASQA dev split (948 total).
2. Collect every ``wikipages`` URL referenced by those examples, dedupe, and
   fetch+cache the HTML (network + disk cache; free).
3. Ingest the cached pages through the *real* IngestPipeline (same parser/
   cleaner/chunker every PDF goes through) into an *isolated* BM25 (+ FAISS)
   index at one ``--token-budget``.
4. For every ``qa_pairs`` sub-question, run it through QueryPipeline and
   check whether any human-labeled ``short_answers`` string appears in the
   top-k retrieved chunks (case-insensitive substring match). Report
   Recall@5 / Recall@10.

Your production data in data/ is NEVER touched — everything lives in
data/eval/asqa/.

This is a PILOT: validates the fetch → ingest → retrieve → score pipeline
end-to-end on a small sample before running the full chunk_size sweep
(scripts/asqa_chunk_sweep.py once this is validated).

Usage:
    # BM25-only pilot (free, no API key needed), 150 sampled examples:
    python scripts/asqa_eval.py

    # Larger sample, different token budget:
    python scripts/asqa_eval.py --n-samples 300 --token-budget 512

    # Re-fetch pages even if cached:
    python scripts/asqa_eval.py --force-refetch

Prerequisites:
    pip install datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("asqa_eval")


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------


def _load_sample(n_samples: int, seed: int):
    """Load the ASQA dev split and deterministically sample N examples.

    Args:
        n_samples: Number of dev-split examples to sample.
        seed: RNG seed for reproducible sampling.

    Returns:
        List of example dicts (``ambiguous_question``, ``qa_pairs``,
        ``wikipages``, ``annotations``, ``sample_id``).
    """
    from datasets import load_dataset

    print(f"[dataset] Loading din0s/asqa dev split...")
    ds = load_dataset("din0s/asqa", split="dev")
    n_samples = min(n_samples, len(ds))
    indices = random.Random(seed).sample(range(len(ds)), n_samples)
    examples = [ds[i] for i in indices]
    print(f"[dataset] Sampled {len(examples)} of {len(ds)} dev examples (seed={seed}).")
    return examples


def _collect_eval_units(examples: list[dict]) -> list[dict]:
    """Flatten qa_pairs into individual retrieval-eval units.

    Args:
        examples: Sampled ASQA dev examples.

    Returns:
        List of ``{"question": str, "short_answers": list[str], "sample_id": str}``
        — one per qa_pair with non-empty short_answers.
    """
    units = []
    for ex in examples:
        for qa in ex["qa_pairs"]:
            answers = [a for a in (qa.get("short_answers") or []) if a and a.strip()]
            question = (qa.get("question") or "").strip()
            if answers and question:
                units.append(
                    {
                        "question": question,
                        "short_answers": answers,
                        "sample_id": ex["sample_id"],
                    }
                )
    return units


def _collect_corpus_urls(examples: list[dict]) -> dict[str, str]:
    """Collect deduped {url: title} pairs from every example's wikipages.

    Args:
        examples: Sampled ASQA dev examples.

    Returns:
        Dict mapping URL to article title (dedupes repeated references).
    """
    urls: dict[str, str] = {}
    for ex in examples:
        for wp in ex["wikipages"]:
            url = wp.get("url")
            title = wp.get("title") or url
            if url:
                urls[url] = title
    return urls


# ---------------------------------------------------------------------------
# Corpus fetch + cache
# ---------------------------------------------------------------------------


def _cache_path(corpus_dir: Path, url: str) -> Path:
    """Return the on-disk cache path for a URL (content-addressed by hash)."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:20]
    return corpus_dir / f"{digest}.html"


def _fetch_corpus(
    urls: dict[str, str], corpus_dir: Path, force_refetch: bool
) -> list[tuple[Path, str]]:
    """Fetch (or reuse cached) HTML for every corpus URL.

    Args:
        urls: {url: title} pairs to fetch.
        corpus_dir: Directory for cached HTML files (persists across runs).
        force_refetch: Re-download even if a cached copy exists.

    Returns:
        List of ``(cache_path, title)`` for successfully available pages.
    """
    import trafilatura

    corpus_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, str]] = []
    n_cached = n_fetched = n_failed = 0

    for i, (url, title) in enumerate(urls.items(), start=1):
        path = _cache_path(corpus_dir, url)
        if path.exists() and not force_refetch:
            n_cached += 1
            results.append((path, title))
            continue

        try:
            html = trafilatura.fetch_url(url)
        except Exception as exc:
            logger.warning("Fetch failed for '%s': %s", url, exc)
            n_failed += 1
            continue

        if not html:
            logger.warning("Empty response for '%s' — skipping.", url)
            n_failed += 1
            continue

        path.write_text(html, encoding="utf-8")
        n_fetched += 1
        results.append((path, title))

        if i % 25 == 0 or i == len(urls):
            print(f"[fetch] {i}/{len(urls)} URLs processed...")

    print(
        f"[fetch] Done: {n_cached} cached, {n_fetched} freshly fetched, "
        f"{n_failed} failed (of {len(urls)} total)."
    )
    return results


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _ingest_corpus(
    pages: list[tuple[Path, str]],
    db_path: Path,
    index_dir: Path,
    token_budget: int,
    embedding: str,
) -> tuple["QueryPipeline", dict]:  # noqa: F821
    """Ingest cached HTML pages into an isolated BM25 (+FAISS) index.

    Args:
        pages: ``(cache_path, title)`` pairs from ``_fetch_corpus``.
        db_path: SQLite DocStore/TraceStore path (isolated eval DB).
        index_dir: Directory to persist BM25/FAISS indexes.
        token_budget: Chunk token budget for this ingest run.
        embedding: "none" | "openai" | "multilingual".

    Returns:
        ``(query_pipeline, ingest_stats)`` — a ready QueryPipeline over the
        freshly built indexes, and a dict of ok/failed/skipped counts.
    """
    from rag.infra.indexes.bm25_local import BM25LocalIndex
    from rag.infra.indexes.faiss_local import FaissLocalIndex
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline
    from rag.pipelines.query_pipeline import QueryPipeline

    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocStore(str(db_path))
    trace_store = SQLiteTraceStore(str(db_path))

    embedding_provider = None
    vector_index = None
    if embedding == "openai":
        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider

        embedding_provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small", dimensions=1536
        )
        vector_index = FaissLocalIndex()
    elif embedding == "multilingual":
        from rag.infra.embedding.multilingual_embedding import (
            MultilingualEmbeddingProvider,
        )

        embedding_provider = MultilingualEmbeddingProvider(
            model="paraphrase-multilingual-mpnet-base-v2", dim=768
        )
        vector_index = FaissLocalIndex()

    keyword_index = BM25LocalIndex()

    pipeline = IngestPipeline(
        doc_store=doc_store,
        trace_store=trace_store,
        token_budget=token_budget,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        keyword_index=keyword_index,
        index_dir=index_dir,
    )

    stats = {"ok": 0, "skipped": 0, "failed": 0, "other": 0}
    for i, (path, title) in enumerate(pages, start=1):
        canonical = f"{title}.html"
        try:
            result = pipeline.ingest(str(path), canonical_name=canonical)
            status = getattr(result, "status", "ok")
            stats[status if status in ("ok", "skipped", "failed") else "other"] += 1
        except Exception as exc:
            logger.warning("Ingest failed for '%s': %s", title, exc)
            stats["failed"] += 1

        if i % 25 == 0 or i == len(pages):
            print(f"[ingest] {i}/{len(pages)} pages ingested (token_budget={token_budget})...")

    print(f"[ingest] Result: {stats}")

    query_pipeline = QueryPipeline(
        keyword_index=keyword_index,
        trace_store=trace_store,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        top_k=20,  # over-fetch; we slice to k when scoring recall@5/@10
    )
    return query_pipeline, stats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _is_hit(short_answers: list[str], texts: list[str]) -> bool:
    """Return True if any short_answer appears as a substring of any text."""
    haystack = " ".join(texts).lower()
    return any(ans.lower() in haystack for ans in short_answers)


def _score(query_pipeline, units: list[dict], top_k_values: list[int]) -> dict:
    """Run every eval unit through the query pipeline and score recall@k.

    Args:
        query_pipeline: QueryPipeline built over the ASQA corpus index.
        units: Eval units from ``_collect_eval_units``.
        top_k_values: Cut-offs to report recall at (e.g. [5, 10]).

    Returns:
        Dict with recall@k scores, per-unit detail, and timing.
    """
    max_k = max(top_k_values)
    hits_at_k = {k: 0 for k in top_k_values}
    detail = []
    start = time.monotonic()

    for i, unit in enumerate(units, start=1):
        result = query_pipeline.query(unit["question"])
        candidates = result.candidates[:max_k]
        texts = [c.display_text for c in candidates]

        row = {"question": unit["question"], "short_answers": unit["short_answers"]}
        for k in top_k_values:
            hit = _is_hit(unit["short_answers"], texts[:k])
            row[f"hit@{k}"] = hit
            if hit:
                hits_at_k[k] += 1
        detail.append(row)

        if i % 25 == 0 or i == len(units):
            print(f"[score] {i}/{len(units)} questions evaluated...")

    elapsed = time.monotonic() - start
    n = len(units) or 1
    return {
        "n_questions": len(units),
        "recall": {f"recall@{k}": hits_at_k[k] / n for k in top_k_values},
        "elapsed_sec": elapsed,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-samples", type=int, default=150, help="Dev examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--token-budget", type=int, default=256, help="Chunk token budget.")
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[5, 10], help="Recall@k cut-offs to report."
    )
    parser.add_argument(
        "--embedding",
        choices=["none", "openai", "multilingual"],
        default="none",
        help="Embedding provider for hybrid retrieval (none = BM25-only, free).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/eval/asqa"), help="Isolated eval data root."
    )
    parser.add_argument(
        "--force-refetch", action="store_true", help="Re-download pages even if cached."
    )
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON to this path.")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    examples = _load_sample(args.n_samples, args.seed)
    units = _collect_eval_units(examples)
    urls = _collect_corpus_urls(examples)
    print(f"[dataset] {len(units)} eval questions, {len(urls)} unique corpus URLs.")

    corpus_dir = args.data_dir / "corpus"
    pages = _fetch_corpus(urls, corpus_dir, args.force_refetch)

    # Suffix the DB/index path by embedding mode so switching --embedding
    # on the same token_budget never collides with an existing DB (which
    # would otherwise "skip" unchanged docs and leave a freshly-constructed,
    # empty in-process index unpopulated for this run).
    suffix = "" if args.embedding == "none" else f"_{args.embedding}"
    db_path = args.data_dir / f"tb{args.token_budget}{suffix}.db"
    index_dir = args.data_dir / f"tb{args.token_budget}{suffix}_indexes"
    query_pipeline, ingest_stats = _ingest_corpus(
        pages, db_path, index_dir, args.token_budget, args.embedding
    )

    scores = _score(query_pipeline, units, args.top_k)

    print()
    print("=" * 60)
    print(f"ASQA pilot — {scores['n_questions']} questions, "
          f"token_budget={args.token_budget}, embedding={args.embedding}")
    print("=" * 60)
    for k, v in scores["recall"].items():
        print(f"  {k:<12} {v * 100:.1f}%")
    print(f"  {'query time':<12} {scores['elapsed_sec']:.1f}s total, "
          f"{scores['elapsed_sec'] / max(scores['n_questions'], 1) * 1000:.0f}ms/query")
    print(f"  {'ingest':<12} {ingest_stats}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "n_samples": args.n_samples,
                "seed": args.seed,
                "token_budget": args.token_budget,
                "top_k": args.top_k,
                "embedding": args.embedding,
            },
            "ingest_stats": ingest_stats,
            "n_corpus_urls": len(urls),
            **scores,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\n[output] Results saved to {args.output}")


if __name__ == "__main__":
    main()
