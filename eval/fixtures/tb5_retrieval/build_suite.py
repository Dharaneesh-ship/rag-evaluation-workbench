#!/usr/bin/env python3
"""Build the TB5 retrieval-eval suite from the frozen event script.

Ground truth is judged by event_id, per 计划 X.6/X.7 ("检索层评测:基于
event_id 判定"), not by string matching — a probe_query's relevant set is
"every chunk ingested from a document filed under this event_id's corpus
directory", regardless of what words that chunk happens to contain. This
also structurally rules out probe_query-vs-corpus leakage: relevance is
positional (which directory a chunk's source file lives in), not lexical.

Only the 10 events that carry corpus_evidence (7 normal + 3 distractor) are
in scope — the 2 unexplained events have no corpus by design (that's the
generation-layer hallucination test in stage 4's A/B eval, not a retrieval
question: recall/precision against an empty relevant set isn't meaningful).

Usage::

    python eval/fixtures/tb5_retrieval/build_suite.py

Reads:
    eval/fixtures/event_script/event_script.yaml  (frozen, tag event-script-v1)
    data/tb5.db                                   (rag.cli.ingest output)

Writes:
    eval/fixtures/tb5_retrieval/dev.json   — dev-split queries, safe to use
                                              for retrieval tuning now
    eval/fixtures/tb5_retrieval/test.json  — test-split queries, FROZEN —
                                              per X.7, only run at the final
                                              A/B reveal, not for tuning
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 附录 CP.4: this script used to `yaml.safe_load` the frozen event script
# itself. It no longer does — every in-repository reader of that file goes
# through rag.eval.split_guard, which is what makes "read the event script"
# a single, greppable, testable act instead of a habit each caller re-invents
# (tests/test_split_guard.py enforces that there is no second reader).
# Building both suites legitimately needs every split, so it declares that.
from rag.eval.split_guard import load_event_script_document  # noqa: E402

_DB_PATH = _ROOT / "data" / "tb5.db"
_OUT_DIR = Path(__file__).parent


def _load_chunk_ids_by_event(db_path: Path = _DB_PATH) -> dict[str, list[str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT c.chunk_id, d.source_path
            FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
            """
        )
        by_event: dict[str, list[str]] = defaultdict(list)
        for chunk_id, source_path in cur.fetchall():
            m = re.search(r"/tb5_retrieval/corpus/([^/]+)/", source_path)
            if not m:
                continue
            by_event[m.group(1)].append(chunk_id)
        return dict(by_event)
    finally:
        conn.close()


def build_suites(
    events: list[dict], chunk_ids_by_event: dict[str, list[str]], *, v1_only: bool = False
) -> tuple[dict[str, list[dict]], list[str]]:
    """Assemble ``{"dev": [...], "test": [...]}`` and the list of events with no chunks.

    附录 FR.1 point 3 / FR.6.1: the TB5 retrieval *test* holdout is defined by
    the frozen batch-v1 test probes and nothing else. v2 test events (sealed)
    never reach this function, and any public event with ``batch == "v2"`` is
    kept out of the test suite even if one were to carry ``split: test``, so
    ``test.json`` stays byte-identical to its pre-v2 content. Retrieval
    entries use ``probe_queries`` only; ``h2_probes`` belong to the H2 eval.
    """
    suites: dict[str, list[dict]] = {"dev": [], "test": []}
    missing: list[str] = []
    for event in events:
        if not event.get("corpus_evidence"):
            continue  # unexplained events carry no corpus: out of scope
        batch = event.get("batch", "v1")
        if v1_only and batch != "v1":
            continue
        split = event["split"]
        if split == "test" and batch != "v1":
            continue
        event_id = event["event_id"]
        relevant = chunk_ids_by_event.get(event_id, [])
        if not relevant:
            missing.append(event_id)
            continue
        for i, query in enumerate(event["probe_queries"], start=1):
            suites[split].append(
                {
                    "query_id": f"{event_id}#{i}",
                    "query": query,
                    "expected_behavior": "answer",
                    "expected_sources": relevant,
                    "notes": f"[{event['event_type']}] {event['business_description']}",
                    # 附录 CI.4: self-declaring split marker, checked by the
                    # reveal guard regardless of which file loaded this entry.
                    "split": split,
                }
            )
    return suites, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=_DB_PATH, help="ingested corpus database")
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--v1-only", action="store_true", help="only batch-v1 events (reproduces the v1 suites)")
    args = ap.parse_args(argv)

    data = load_event_script_document(
        reason="build_suite.py writes both dev.json and test.json — needs every split"
    )
    suites, missing = build_suites(
        data["events"], _load_chunk_ids_by_event(args.db), v1_only=args.v1_only
    )
    if missing:
        raise SystemExit(
            f"No ingested chunks found for event(s): {missing}. "
            f"Run rag.cli.ingest (or scripts/tb5_build_corpus_index.py) over "
            f"data/tb5_corpus/ first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in suites.items():
        out_path = args.out_dir / f"{split}.json"
        out_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {out_path} — {len(entries)} quer(ies)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
