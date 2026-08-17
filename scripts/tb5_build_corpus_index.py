"""One-off script: ingest eval/fixtures/tb5_retrieval/corpus/ with a real
BM25/vector index attached.

rag.cli.ingest populates the docstore but never wires a keyword/vector index
(IngestPipeline needs one explicitly passed in) — the same construction the
MCP server's ingest tool uses is in rag.app.mcp_server.wiring.build_ingest_pipeline.
Reusing it here keeps this script's index on the same code path production
ingest uses, instead of hand-rolling a second one.

BM25-only (embedding_provider="none") — TB5's retrieval eval doesn't need a
vector index to sanity-check relevance-by-event_id, and it avoids requiring
API keys / the FAISS+voyage OpenMP crash surface just to build a fixture.

Usage::

    python scripts/tb5_build_corpus_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_CORPUS_DIR = _ROOT / "eval" / "fixtures" / "tb5_retrieval" / "corpus"


def main() -> int:
    import sys as _sys

    from dotenv import load_dotenv

    load_dotenv(override=False)

    from rag.app.mcp_server.wiring import build_ingest_pipeline
    from rag.pipelines.batch_ingest import BatchIngestRunner

    embedding_provider = _sys.argv[1] if len(_sys.argv) > 1 else "none"
    suffix = "" if embedding_provider == "none" else "_hybrid"
    db_path = _ROOT / "data" / f"tb5{suffix}.db"
    index_dir = _ROOT / "data" / f"tb5{suffix}_indexes"

    pipeline, mgr = build_ingest_pipeline(
        db_path=str(db_path),
        index_dir=str(index_dir),
        embedding_provider=embedding_provider,
    )

    runner = BatchIngestRunner(pipeline, pipeline._trace_store)
    pipeline.begin_batch()
    report = runner.run(
        _CORPUS_DIR,
        recursive=True,
        on_progress=lambda line: print(line, flush=True),
    )
    pipeline.end_batch()
    mgr.save()

    print()
    print(report.format_summary())
    return 1 if report.has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
