"""CLI entry point for document ingestion.

Usage::

    python -m rag.cli.ingest --path document.pdf
    python -m rag.cli.ingest --path document.pdf --collection my_collection
    python -m rag.cli.ingest --path ./docs                # whole directory
    python -m rag.cli.ingest --path ./docs --no-recursive
    python -m rag.cli.ingest --path ./docs --workers 8     # parallel parse/embed

When ``--path`` is a directory, every non-hidden file underneath it is
ingested and a per-status summary report is printed. The exit code is
non-zero when any file ends up unretrievable (failed / unsupported /
empty content), so silent ingest gaps can't slip through scripts or CI.
"""

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.cli.ingest",
        description="Ingest a document into the RAG framework.",
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Path to the file to ingest, or a directory to batch-ingest.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When --path is a directory, do not descend into subdirectories.",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Target collection name (used to derive the database path). "
             "Defaults to 'default'.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Explicit path to the SQLite database file. "
             "If omitted, defaults to data/<collection>.db.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=512,
        help="Approximate token budget per chunk. Defaults to 512.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="When --path is a directory, number of concurrent prepare-phase "
             "threads (parse/clean/chunk/embed). DocStore writes and index "
             "updates always run one file at a time. Defaults to 1 (sequential).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ingest CLI.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    # Import here to keep startup fast and errors surfaced at call time.
    from dotenv import load_dotenv
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    # API keys from .env enable the optional pdf_vision parser tier.
    load_dotenv(override=False)

    args = _build_parser().parse_args(argv)

    source_path: Path = args.path.resolve()
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        return 1

    db_path: Path = args.db if args.db else Path("data") / f"{args.collection}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)

    pipeline = IngestPipeline(
        doc_store=doc_store,
        trace_store=trace_store,
        token_budget=args.token_budget,
    )

    # Directory → batch ingest with a per-status summary report
    if source_path.is_dir():
        from rag.pipelines.batch_ingest import BatchIngestRunner

        runner = BatchIngestRunner(pipeline, trace_store)
        report = runner.run(
            source_path,
            recursive=not args.no_recursive,
            progress=True,
            workers=args.workers,
        )
        print()
        print(report.format_summary())
        return 1 if report.has_problems else 0

    result = pipeline.ingest(source_path)

    if result.error:
        print(f"Error: ingestion failed for '{source_path.name}'", file=sys.stderr)
        print(f"  {result.error}", file=sys.stderr)
        return 1

    elapsed_s = result.elapsed_ms / 1000
    status_note = f" [{result.status}]" if result.status != "ok" else ""
    cost_note = f", ${result.cost_usd:.4f}" if result.cost_usd else ""
    print(
        f"Ingested '{source_path.name}' — "
        f"{result.chunk_count} chunks, "
        f"{result.block_count} blocks, "
        f"{elapsed_s:.2f}s{cost_note}{status_note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
