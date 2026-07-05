#!/usr/bin/env python3
"""Benchmark: BM25 index rebuild cost, batch mode on vs off.

Ingests the same corpus twice into two fresh, isolated indexes — once with
IngestPipeline's default per-document rebuild, once wrapped in
begin_batch()/end_batch() (defers the rebuild to once at the end) — and
reports the wall-clock difference.

Requires the ASQA corpus to already be cached (run scripts/asqa_eval.py
at least once first, or point --corpus-dir at any directory of ingestable
files).

Usage:
    python scripts/bench_batch_ingest.py
    python scripts/bench_batch_ingest.py --corpus-dir data/eval/asqa/corpus
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _run(pages: list[Path], db_dir: Path, use_batch_mode: bool) -> tuple[float, int]:
    """Ingest every page into a fresh index and return (elapsed_sec, chunk_count)."""
    from rag.infra.indexes.bm25_local import BM25LocalIndex
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True)

    db_path = str(db_dir / "t.db")
    bm25 = BM25LocalIndex()
    pipeline = IngestPipeline(
        SQLiteDocStore(db_path), SQLiteTraceStore(db_path), keyword_index=bm25
    )

    start = time.monotonic()
    if use_batch_mode:
        pipeline.begin_batch()
    for page in pages:
        pipeline.ingest(str(page), canonical_name=f"{page.stem}.html")
    if use_batch_mode:
        pipeline.end_batch()
    elapsed = time.monotonic() - start

    return elapsed, len(bm25._chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/eval/asqa/corpus"),
        help="Directory of files to ingest (defaults to the cached ASQA corpus).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/eval/bench_batch_ingest"),
        help="Scratch directory for the two benchmark indexes (deleted/recreated).",
    )
    args = parser.parse_args()

    pages = sorted(args.corpus_dir.glob("*.html"))
    if not pages:
        raise SystemExit(
            f"No .html files found in {args.corpus_dir}. "
            f"Run scripts/asqa_eval.py first to populate the corpus cache, "
            f"or pass --corpus-dir."
        )
    print(f"Corpus: {len(pages)} files from {args.corpus_dir}")

    t_before, n_before = _run(pages, args.work_dir / "before", use_batch_mode=False)
    print(f"[BEFORE] add() per document (no batch mode): "
          f"{t_before:.1f}s for {len(pages)} docs, {n_before} chunks")

    t_after, n_after = _run(pages, args.work_dir / "after", use_batch_mode=True)
    print(f"[AFTER]  begin_batch()/end_batch():           "
          f"{t_after:.1f}s for {len(pages)} docs, {n_after} chunks")

    print(f"\nSpeedup: {t_before / t_after:.1f}x")
    print(f"Time saved: {t_before - t_after:.1f}s ({(1 - t_after / t_before) * 100:.0f}% reduction)")


if __name__ == "__main__":
    main()
