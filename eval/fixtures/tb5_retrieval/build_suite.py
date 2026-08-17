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

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent.parent.parent
_EVENT_SCRIPT = _ROOT / "eval" / "fixtures" / "event_script" / "event_script.yaml"
_DB_PATH = _ROOT / "data" / "tb5.db"
_OUT_DIR = Path(__file__).parent


def _load_chunk_ids_by_event() -> dict[str, list[str]]:
    conn = sqlite3.connect(str(_DB_PATH))
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


def main() -> int:
    data = yaml.safe_load(_EVENT_SCRIPT.read_text(encoding="utf-8"))
    events = data["events"]
    chunk_ids_by_event = _load_chunk_ids_by_event()

    suites: dict[str, list[dict]] = {"dev": [], "test": []}
    missing: list[str] = []

    for event in events:
        if not event.get("corpus_evidence"):
            continue  # unexplained events carry no corpus — out of scope

        event_id = event["event_id"]
        split = event["split"]
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
                    "notes": (
                        f"[{event['event_type']}] {event['business_description']}"
                    ),
                }
            )

    if missing:
        raise SystemExit(
            f"No ingested chunks found for event(s): {missing}. "
            f"Run rag.cli.ingest (or scripts/tb5_build_corpus_index.py) over "
            f"data/tb5_corpus/ first."
        )

    for split, entries in suites.items():
        out_path = _OUT_DIR / f"{split}.json"
        out_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {out_path} — {len(entries)} quer(ies)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
