"""CLI entry point for retrieval evaluation.

Usage::

    python -m rag.cli.eval --suite example_queries
    python -m rag.cli.eval --suite failure_cases
    python -m rag.cli.eval --suite example_queries --k 5 --top-k 10
    python -m rag.cli.eval --answer-quality
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from rag.eval.split_guard import (
    assert_not_test_probe_query,
    declare_test_split_reveal,
    filter_by_split,
    warn_if_test_split_involved,
)

# Default location for built-in eval suites.
_DEFAULT_SUITE_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"

# Built-in suite filenames (created in Task 10.5).
_SUITE_FILES = {
    "example_queries": "example_queries.json",
    "failure_cases": "failure_cases.json",
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.cli.eval",
        description="Run retrieval evaluation over a labelled query suite.",
    )

    suite_group = parser.add_mutually_exclusive_group(required=True)
    suite_group.add_argument(
        "--suite",
        choices=list(_SUITE_FILES.keys()),
        help="Built-in eval suite to run ('example_queries' or 'failure_cases').",
    )
    suite_group.add_argument(
        "--answer-quality",
        action="store_true",
        dest="answer_quality",
        help="Run RAGAS answer-quality evaluation (requires Phase 14 dependencies).",
    )

    parser.add_argument(
        "--suite-file",
        type=Path,
        default=None,
        dest="suite_file",
        help="Override: path to a custom JSON/JSONL suite file.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to the SQLite database. Defaults to data/default.db.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        dest="index_dir",
        help="Directory containing BM25/FAISS indexes. "
             "Defaults to data/default_indexes/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top_k",
        help="Number of candidates to retrieve per query. Defaults to 10.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Metric cut-off depth (Recall@K, MRR, nDCG@K). Defaults to 10.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case retrieved chunk IDs.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="server_default",
        dest="embedding_provider",
        choices=["server_default", "none", "openai"],
        help="Embedding provider for retrieval. Defaults to 'server_default' "
             "(the service's configured default; see README §8).",
    )
    parser.add_argument(
        "--reranker-provider",
        default="server_default",
        dest="reranker_provider",
        choices=["server_default", "none", "crossencoder", "voyage"],
        help="Reranker provider for retrieval. Defaults to 'server_default'.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="附录 CI.4 (rag-side counterpart of MAEDA's 附录 CC.1 "
             "measure_noise.py --split fix): filters loaded suite entries "
             "by their self-declared 'split' field (written by "
             "eval/fixtures/tb5_retrieval/build_suite.py). Defaults to "
             "'dev' -- entries with split=='test' (eval/fixtures/"
             "tb5_retrieval/test.json is FROZEN for the one-time A/B "
             "reveal, 硬约束 1) are excluded no matter how the suite was "
             "selected, including --suite-file pointed straight at "
             "test.json -- the filter runs on loaded entry content, not "
             "on the filename. 'test'/'all' must be requested explicitly "
             "and print an unmissable warning naming every test-split "
             "query id about to run. Suites without a 'split' field "
             "(example_queries, failure_cases) are unaffected.",
    )
    return parser


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------


def _resolve_suite_path(suite_name: str, suite_file: Path | None) -> Path:
    """Return the path to the suite file.

    Args:
        suite_name: Built-in suite key (e.g. ``"example_queries"``).
        suite_file: Optional explicit override path.

    Returns:
        Resolved ``Path`` to the suite JSON file.
    """
    if suite_file is not None:
        return suite_file.resolve()
    filename = _SUITE_FILES[suite_name]
    return (_DEFAULT_SUITE_DIR / filename).resolve()


def _load_suite(path: Path) -> list[dict]:
    """Load a JSON or JSONL suite file.

    Each entry must have at minimum a ``"query"`` key.

    Args:
        path: Path to the suite file.

    Returns:
        List of entry dicts.

    Raises:
        ValueError: If the file format is unrecognised.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Split-aware case selection (附录 CI.4 / CP.4)
# ---------------------------------------------------------------------------
#
# These two functions used to be *defined* here, and 附录 CP.4 showed what
# that cost: a repo-wide grep found `filter_by_split` called from exactly one
# place, this file, while the probe scripts of 附录 CJ.3 and 附录 CO read the
# frozen event script directly and never came near it. The guard is now a
# shared mechanism in `rag.eval.split_guard`, with the check that actually
# fails closed sitting at execution (`QueryPipeline.query`) rather than at any
# one entry point. They are re-exported here unchanged, because
# `rag.cli.eval.filter_by_split` is the name the CI.4 tests and prior
# appendices refer to.

__all__ = ["main", "filter_by_split", "warn_if_test_split_involved"]


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def _run_queries(
    entries: list[dict],
    db_path: Path,
    index_dir: Path,
    top_k: int,
    embedding_provider: str = "server_default",
    reranker_provider: str = "server_default",
) -> list[dict]:
    """Run each suite entry through the query pipeline and collect results.

    Args:
        entries: Suite entries (each must have ``"query"``).
        db_path: SQLite database path.
        index_dir: Index directory path.
        top_k: Number of candidates to retrieve.
        embedding_provider: ``"server_default"``, ``"none"``, or an explicit
            provider key — resolved via the same
            ``rag.app.mcp_server.retrieval_config`` path as the MCP
            retrieve tools (R2's "评测入口" shares config resolution too).
        reranker_provider: ``"server_default"``, ``"none"``, or an explicit
            reranker key.

    Returns:
        List of result dicts suitable for ``run_eval()``.
    """
    # 附录 CP.4: same content-based check QueryPipeline.query() enforces, run
    # once over the whole batch before any index is loaded or any embedding
    # provider is constructed. Redundant by design — this one only makes the
    # refusal cheap and early; the one that closes the hole is downstream and
    # cannot be routed around.
    assert_not_test_probe_query(
        [e.get("query", "") for e in entries], context="rag.cli.eval._run_queries"
    )

    from rag.app.mcp_server.wiring import build_query_pipeline

    pipeline = build_query_pipeline(
        db_path=str(db_path),
        index_dir=str(index_dir),
        top_k=top_k,
        enable_generation=False,  # skip LLM generation in eval
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
    )

    results = []
    for entry in entries:
        query: str = entry.get("query", "")
        expected_sources: list[str] = entry.get("expected_sources", [])
        expected_behavior: str = entry.get("expected_behavior", "answer")
        query_id: str = entry.get("query_id", query[:40])

        t0 = time.perf_counter()
        qr = pipeline.query(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [c.chunk_id for c in qr.candidates]

        results.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieved": retrieved_ids,
                "relevant": expected_sources,
                "candidates": qr.candidates,
                "expected_behavior": expected_behavior,
                "query_latency_ms": elapsed_ms,
                "error": qr.error,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"  # no expected_sources supplied — can't score


def _case_outcome(entry: dict, k: int) -> str:
    """Return PASS / FAIL / SKIP for a single result entry."""
    relevant = entry.get("relevant", [])
    if not relevant:
        return _SKIP
    retrieved_top_k = set(entry.get("retrieved", [])[:k])
    relevant_set = set(relevant)
    return _PASS if retrieved_top_k & relevant_set else _FAIL


def _print_per_case(results: list[dict], k: int, verbose: bool) -> int:
    """Print per-case results and return the number of failures.

    Args:
        results: List of result dicts from ``_run_queries()``.
        k: Cut-off depth for outcome determination.
        verbose: Print retrieved chunk IDs.

    Returns:
        Number of FAIL cases.
    """
    failures = 0
    width = max(len(r.get("query", "")) for r in results) if results else 40
    width = min(width, 60)

    print(f"\n{'Query':<{width}}  {'Expected':<8}  {'Outcome':<6}")
    print("-" * (width + 18))

    for entry in results:
        query = entry.get("query", "")[:width]
        expected = entry.get("expected_behavior", "answer")
        outcome = _case_outcome(entry, k)
        if outcome == _FAIL:
            failures += 1
        marker = "✓" if outcome == _PASS else ("?" if outcome == _SKIP else "✗")
        print(f"{query:<{width}}  {expected:<8}  {marker} {outcome}")
        if verbose and entry.get("retrieved"):
            ids_preview = ", ".join(entry["retrieved"][:5])
            print(f"  retrieved: [{ids_preview}{'...' if len(entry['retrieved']) > 5 else ''}]")
        if entry.get("error"):
            print(f"  ERROR: {entry['error']}")

    print()
    return failures


def _print_aggregate(report: Any) -> None:
    """Print aggregate metrics from an EvalReport."""
    print("─" * 46)
    print("Aggregate metrics")
    print("─" * 46)
    print(f"  Queries evaluated : {report.num_queries}")
    print(f"  Mean Recall@{report.k:<5}: {report.mean_recall_at_k:.4f}")
    print(f"  MRR               : {report.mrr:.4f}")
    print(f"  Mean nDCG@{report.k:<5}  : {report.mean_ndcg_at_k:.4f}")

    sa = report.source_attribution
    if sa.total_candidates > 0:
        print("\nSource attribution (across all retrieved candidates)")
        print(f"  bm25_only   : {sa.bm25_only:.1%}")
        print(f"  vector_only : {sa.vector_only:.1%}")
        print(f"  both        : {sa.both:.1%}")
        print(f"  total       : {sa.total_candidates} candidates")

    eff = report.efficiency
    if any(v is not None for v in [
        eff.mean_query_latency_ms,
        eff.mean_ingest_latency_ms,
        eff.token_saved_est,
    ]):
        print("\nEfficiency")
        if eff.mean_query_latency_ms is not None:
            print(f"  Mean query latency : {eff.mean_query_latency_ms:.1f} ms")
        if eff.mean_ingest_latency_ms is not None:
            print(f"  Ingest latency     : {eff.mean_ingest_latency_ms:.1f} ms")
        if eff.token_saved_est is not None:
            print(f"  Token saved (est.) : {eff.token_saved_est:.0f} tokens/query")
        if eff.skipped_chunks is None:
            print("  Skipped chunks     : N/A (requires Task 11.2)")
        if eff.changed_chunks is None:
            print("  Changed chunks     : N/A (requires Task 11.2)")

    print("─" * 46)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the eval CLI.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success or expected failures, 1 on unexpected errors.
    """
    args = _build_parser().parse_args(argv)

    # --answer-quality: RAGAS stub until Phase 14
    if args.answer_quality:
        print("RAGAS not available — install Phase 14 dependencies to enable answer-quality evaluation.")
        return 0

    # Resolve suite file path
    suite_path = _resolve_suite_path(args.suite, args.suite_file)
    if not suite_path.exists():
        print(
            f"Error: suite file not found: {suite_path}\n"
            f"Hint: run Task 10.5 to generate the fixture files, or pass --suite-file.",
            file=sys.stderr,
        )
        return 1

    # Load suite
    try:
        raw_entries = _load_suite(suite_path)
    except Exception as exc:
        print(f"Error: failed to load suite '{suite_path}': {exc}", file=sys.stderr)
        return 1

    if not raw_entries:
        print("Warning: suite is empty — nothing to evaluate.")
        return 0

    # 附录 CI.4: exclude/gate test-split entries before anything runs.
    entries = filter_by_split(raw_entries, args.split)
    warn_if_test_split_involved(entries)

    if not entries:
        print(
            f"Warning: all {len(raw_entries)} case(s) in this suite are "
            f"test-split (附录 CI.4) and --split is '{args.split}' — "
            "nothing to evaluate. Pass --split test (or --split all) only "
            "if this is the intended, one-time test-split reveal (硬约束 1)."
        )
        return 0

    print(f"Evaluating suite '{args.suite}' — {len(entries)} case(s)  [k={args.k}, split={args.split}]")

    # Resolve DB and index paths
    db_path: Path = args.db if args.db else Path("data") / "default.db"
    index_dir: Path = args.index_dir if args.index_dir else Path("data") / "default_indexes"

    if not db_path.exists():
        print(
            f"Error: database not found at '{db_path}'. "
            "Run rag.cli.ingest first.",
            file=sys.stderr,
        )
        return 1

    # 附录 CP.4: `--split test`/`--split all` with test entries actually
    # selected IS the sanctioned reveal path — the operator typed the
    # non-default value and read the CI.4 banner above. Declare it explicitly
    # so the execution-time guard lets these through; every *other* way of
    # reaching a test-split probe query stays closed.
    has_test = any(e.get("split") == "test" for e in entries)
    reveal: Any = (
        declare_test_split_reveal(
            f"rag.cli.eval --split {args.split} over {suite_path} "
            f"({sum(1 for e in entries if e.get('split') == 'test')} test-split case(s))"
        )
        if has_test
        else contextlib.nullcontext()
    )

    # Run queries
    try:
        with reveal:
            results = _run_queries(
                entries,
                db_path,
                index_dir,
                top_k=args.top_k,
                embedding_provider=args.embedding_provider,
                reranker_provider=args.reranker_provider,
            )
    except Exception as exc:
        print(f"Error: unexpected failure during query execution: {exc}", file=sys.stderr)
        return 1

    # Per-case output
    num_failures = _print_per_case(results, k=args.k, verbose=args.verbose)

    # Aggregate metrics
    try:
        from rag.pipelines.eval_pipeline import run_eval

        report = run_eval(results, k=args.k)
        _print_aggregate(report)
    except Exception as exc:
        print(f"Error: failed to compute aggregate metrics: {exc}", file=sys.stderr)
        return 1

    # Summary line
    total = len(results)
    skipped = sum(1 for r in results if _case_outcome(r, args.k) == _SKIP)
    passed = total - num_failures - skipped
    print(
        f"\nResult: {passed}/{total} passed"
        + (f", {num_failures} failed" if num_failures else "")
        + (f", {skipped} skipped (no expected_sources)" if skipped else "")
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
