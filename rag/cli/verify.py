"""CLI entry point for ingest coverage verification.

Reconciles a source directory against the DocStore and reports files that
were never ingested, produced no chunks, or extracted suspiciously little
text. Run it after a batch ingest as an independent safety net.

Usage::

    python -m rag.cli.verify --path ./docs
    python -m rag.cli.verify --path ./docs --collection my_collection
    python -m rag.cli.verify --path ./docs --min-chars-per-kb 10
"""

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.cli.verify",
        description="Verify that every file in a directory was ingested "
        "into the DocStore with retrievable content.",
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Directory whose files should all be present in the DocStore.",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name (used to derive the database path). "
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
        "--no-recursive",
        action="store_true",
        help="Do not descend into subdirectories.",
    )
    parser.add_argument(
        "--min-chars-per-kb",
        type=float,
        default=5.0,
        help="Flag files whose extracted text density is below this many "
             "characters per KB of file size. Defaults to 5.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the verify CLI.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 when coverage is complete, 1 when any file is
        missing/empty/suspicious or the inputs are invalid.
    """
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.pipelines.verify_audit import verify_directory

    args = _build_parser().parse_args(argv)

    root: Path = args.path.resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    db_path: Path = args.db if args.db else Path("data") / f"{args.collection}.db"
    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        return 1

    doc_store = SQLiteDocStore(db_path)
    report = verify_directory(
        doc_store,
        root,
        recursive=not args.no_recursive,
        min_chars_per_kb=args.min_chars_per_kb,
    )

    print(report.format_summary())
    return 1 if report.has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
