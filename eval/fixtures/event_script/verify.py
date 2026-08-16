#!/usr/bin/env python3
"""Validate (and, at freeze time, hash) the TB5 event script.

Implements the mechanical checks frozen in ECOSYSTEM_INTEGRATION_PLAN.md
附录 X.2–X.6: field/enum validity, the 12/7/3/2 composition constraints,
the X.4 deterministic dev/test split assignment, and the X.6 rule-two
lexical-leak check (a probe_query must not contain a required_fact
verbatim). Canonicalization/hashing follows X.5 exactly.

Usage::

    python verify.py                  # check only, exit 1 on any failure
    python verify.py --write-manifest # also write event_script.manifest.json

``--write-manifest`` is the X.7 step-5 freeze action — do not run it before
the three-party content review in step 4 has actually happened. It fills in
``sha256``/``counts``/``split_counts`` but deliberately leaves
``frozen_at``/``git_tag`` for whoever performs the actual freeze ceremony to
fill in by hand, since a git tag doesn't exist yet when this file is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
_YAML_PATH = _HERE / "event_script.yaml"
_MANIFEST_PATH = _HERE / "event_script.manifest.json"

_EVENT_ID_RE = re.compile(r"^EVT-[A-Z0-9-]{4,48}$")
_VALID_EVENT_TYPES = {"normal", "distractor", "unexplained"}
_VALID_SPLITS = {"dev", "test", "stress"}
_VALID_METRICS = {"revenue", "units"}
_VALID_DIRECTIONS = {"up", "down"}
_VALID_SHAPES = {"step", "ramp", "spike"}
_VALID_RELEVANCE = {"relevant", "partial"}
_VALID_SOURCE_TYPES = {"email", "meeting_notes", "policy_memo", "promo_calendar"}
_VALID_LEXICAL_DIVERGENCE = {"synonym", "abbreviation", "colloquial", "implicit"}

# Real column values, read from MAEDA's data/demo/sales_data.csv on 2026-08-15
# (12240 rows). This is a static copy, not a live read — rag-framework has no
# access to the MAEDA repo at runtime. If MAEDA's demo dataset categories
# ever change, this dict must be re-synced by hand, or this check will
# reject events written against the old categories (or worse, silently
# accept segment values that no longer exist in the real data).
_KNOWN_SEGMENT_VALUES = {
    "region": {"North", "South", "East", "West", "Central"},
    "product": {"Widget A", "Widget B", "Gadget Lite", "Gadget Pro", "Service Pack"},
    "channel": {"Direct", "Online", "Partner", "Retail"},
}

_DATE_MIN = date(2022, 1, 1)
_DATE_MAX = date(2024, 12, 31)

# X.3 frozen composition — do not adjust post-hoc based on results.
_EXPECTED_TOTAL = 12
_EXPECTED_TYPE_COUNTS = {"normal": 7, "distractor": 3, "unexplained": 2}
_MIN_PROBES_NORMAL_DISTRACTOR = 2
_MAX_EVENTS_PER_REGION = 4
_MIN_DISTINCT_MONTHS = 6
_MIN_IMPLICIT_NORMAL_EVENTS = 3


def _parse_date(raw: str, field: str, event_id: str, errors: list[str]) -> date | None:
    try:
        y, m, d = (int(x) for x in raw.split("-"))
        return date(y, m, d)
    except Exception:
        errors.append(f"{event_id}: {field}='{raw}' is not a valid ISO date (YYYY-MM-DD)")
        return None


def _check_data_signal(event_id: str, ds: dict, errors: list[str]) -> None:
    if ds.get("metric") not in _VALID_METRICS:
        errors.append(f"{event_id}: data_signal.metric={ds.get('metric')!r} invalid")

    segment = ds.get("segment")
    if not isinstance(segment, dict):
        errors.append(f"{event_id}: data_signal.segment must be a dict (empty dict = global)")
    else:
        for key, value in segment.items():
            if key not in _KNOWN_SEGMENT_VALUES:
                errors.append(f"{event_id}: segment key {key!r} not one of region/product/channel")
            elif value not in _KNOWN_SEGMENT_VALUES[key]:
                errors.append(f"{event_id}: segment {key}={value!r} is not a real column value")

    if ds.get("direction") not in _VALID_DIRECTIONS:
        errors.append(f"{event_id}: data_signal.direction={ds.get('direction')!r} invalid")

    magnitude = ds.get("magnitude")
    if not isinstance(magnitude, (int, float)) or isinstance(magnitude, bool) or not (0.05 <= magnitude <= 0.90):
        errors.append(f"{event_id}: data_signal.magnitude={magnitude!r} outside [0.05, 0.90]")

    if ds.get("shape") not in _VALID_SHAPES:
        errors.append(f"{event_id}: data_signal.shape={ds.get('shape')!r} invalid")


def _check_corpus_evidence(event_id: str, ce: dict, errors: list[str]) -> None:
    if ce.get("relevance") not in _VALID_RELEVANCE:
        errors.append(f"{event_id}: corpus_evidence.relevance={ce.get('relevance')!r} invalid")

    source_types = ce.get("source_types") or []
    if not source_types or not set(source_types) <= _VALID_SOURCE_TYPES:
        errors.append(f"{event_id}: corpus_evidence.source_types={source_types!r} invalid or empty")

    facts = ce.get("required_facts") or []
    if not facts or not all(isinstance(f, str) and f.strip() for f in facts):
        errors.append(f"{event_id}: corpus_evidence.required_facts must be a non-empty list of strings")

    divergence = ce.get("lexical_divergence") or []
    if not divergence or not set(divergence) <= _VALID_LEXICAL_DIVERGENCE:
        errors.append(f"{event_id}: corpus_evidence.lexical_divergence={divergence!r} invalid or empty")


def _check_leakage(event_id: str, corpus_evidence: dict, probes: list[str], errors: list[str]) -> None:
    """X.6 rule two: a probe_query must not contain a required_fact verbatim.

    Chinese text has no whitespace word boundaries, so "complete token" is
    implemented as substring containment against the whole required_fact
    phrase (required_facts are already short, meaningful labels like "断货"
    or "补货时间" — not single characters), rather than segmenting the
    probe_query into words. This is deliberately narrower than a general
    stopword-aware tokenizer would be, but it exactly targets the actual
    leak scenario the rule exists to prevent: a probe asking a question
    that already states its own answer.
    """
    for fact in corpus_evidence.get("required_facts") or []:
        fact_norm = fact.strip()
        if not fact_norm:
            continue
        for probe in probes:
            if fact_norm in probe:
                errors.append(
                    f"{event_id}: probe_query {probe!r} leaks required_fact {fact_norm!r} verbatim"
                )


def _check_event(event: dict, errors: list[str]) -> str | None:
    """Validate one event's fields. Returns its event_id if usable for cross-event checks."""
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not _EVENT_ID_RE.match(event_id):
        errors.append(f"{event_id!r}: event_id does not match ^EVT-[A-Z0-9-]{{4,48}}$")

    event_type = event.get("event_type")
    if event_type not in _VALID_EVENT_TYPES:
        errors.append(f"{event_id}: event_type={event_type!r} invalid")
        return event_id  # can't check type-dependent rules without a valid type

    if event.get("split") not in _VALID_SPLITS:
        errors.append(f"{event_id}: split={event.get('split')!r} invalid")

    vf = _parse_date(event.get("valid_from", ""), "valid_from", event_id, errors)
    vt = _parse_date(event.get("valid_to", ""), "valid_to", event_id, errors)
    if vf is not None and not (_DATE_MIN <= vf <= _DATE_MAX):
        errors.append(f"{event_id}: valid_from {vf} outside {_DATE_MIN}..{_DATE_MAX}")
    if vt is not None and not (_DATE_MIN <= vt <= _DATE_MAX):
        errors.append(f"{event_id}: valid_to {vt} outside {_DATE_MIN}..{_DATE_MAX}")
    if vf is not None and vt is not None and vf > vt:
        errors.append(f"{event_id}: valid_from {vf} is after valid_to {vt}")

    data_signal = event.get("data_signal")
    corpus_evidence = event.get("corpus_evidence")

    # X.2's event_type -> required-blocks table
    if event_type == "normal":
        if not data_signal:
            errors.append(f"{event_id}: normal event must have data_signal")
        if not corpus_evidence:
            errors.append(f"{event_id}: normal event must have corpus_evidence")
    elif event_type == "distractor":
        if data_signal:
            errors.append(f"{event_id}: distractor event must NOT have data_signal")
        if not corpus_evidence:
            errors.append(f"{event_id}: distractor event must have corpus_evidence")
    elif event_type == "unexplained":
        if not data_signal:
            errors.append(f"{event_id}: unexplained event must have data_signal")
        if corpus_evidence:
            errors.append(f"{event_id}: unexplained event must NOT have corpus_evidence")

    if data_signal:
        _check_data_signal(event_id, data_signal, errors)
    if corpus_evidence:
        _check_corpus_evidence(event_id, corpus_evidence, errors)

    probes = event.get("probe_queries")
    if not probes or not isinstance(probes, list) or not all(isinstance(p, str) and p.strip() for p in probes):
        errors.append(f"{event_id}: probe_queries must be a non-empty list of strings")
        probes = []
    elif event_type in ("normal", "distractor") and len(probes) < _MIN_PROBES_NORMAL_DISTRACTOR:
        errors.append(
            f"{event_id}: probe_queries has {len(probes)}, need >= {_MIN_PROBES_NORMAL_DISTRACTOR}"
        )

    if corpus_evidence and probes:
        _check_leakage(event_id, corpus_evidence, probes, errors)

    return event_id


def _check_composition(events: list[dict], errors: list[str]) -> None:
    if len(events) != _EXPECTED_TOTAL:
        errors.append(f"total events = {len(events)}, expected {_EXPECTED_TOTAL}")

    type_counts: dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type")
        type_counts[et] = type_counts.get(et, 0) + 1
    for event_type, expected in _EXPECTED_TYPE_COUNTS.items():
        actual = type_counts.get(event_type, 0)
        if actual != expected:
            errors.append(f"event_type '{event_type}' count = {actual}, expected {expected}")

    region_counts: dict[str, int] = {}
    months: set[str] = set()
    implicit_normal = 0
    for ev in events:
        valid_from = ev.get("valid_from", "")
        if len(valid_from) >= 7:
            months.add(valid_from[:7])

        data_signal = ev.get("data_signal") or {}
        region = (data_signal.get("segment") or {}).get("region")
        if region:
            region_counts[region] = region_counts.get(region, 0) + 1

        if ev.get("event_type") == "normal":
            divergence = (ev.get("corpus_evidence") or {}).get("lexical_divergence") or []
            if "implicit" in divergence:
                implicit_normal += 1

    for region, count in region_counts.items():
        if count > _MAX_EVENTS_PER_REGION:
            errors.append(f"region '{region}' appears in {count} events, max is {_MAX_EVENTS_PER_REGION}")

    if len(months) < _MIN_DISTINCT_MONTHS:
        errors.append(f"events span {len(months)} distinct months, need >= {_MIN_DISTINCT_MONTHS}")

    if implicit_normal < _MIN_IMPLICIT_NORMAL_EVENTS:
        errors.append(
            f"only {implicit_normal} normal event(s) use 'implicit' divergence, "
            f"need >= {_MIN_IMPLICIT_NORMAL_EVENTS}"
        )


def _compute_expected_splits(events: list[dict]) -> dict[str, str]:
    """X.4: group by event_type, sort by event_id, round-robin dev/test (dev first)."""
    by_type: dict[str, list[str]] = {}
    for ev in events:
        by_type.setdefault(ev.get("event_type"), []).append(ev.get("event_id"))

    expected: dict[str, str] = {}
    for ids in by_type.values():
        for i, event_id in enumerate(sorted(ids)):
            expected[event_id] = "dev" if i % 2 == 0 else "test"
    return expected


def _check_splits(events: list[dict], errors: list[str]) -> None:
    expected = _compute_expected_splits(events)
    for ev in events:
        event_id = ev.get("event_id")
        exp = expected.get(event_id)
        actual = ev.get("split")
        if exp is not None and actual != exp:
            errors.append(f"{event_id}: split={actual!r} but X.4's rule computes {exp!r}")


def _check_file_hygiene(text: str, errors: list[str]) -> None:
    """X.5: UTF-8 without BOM, LF line endings, exactly one trailing newline."""
    if text.startswith("﻿"):
        errors.append("event_script.yaml has a UTF-8 BOM — must be UTF-8 without BOM")
    if "\r" in text:
        errors.append("event_script.yaml contains CR — must be LF-only line endings")
    if text != text.rstrip("\n") + "\n":
        errors.append("event_script.yaml must end with exactly one trailing newline")


def _canonical_bytes(data: dict) -> bytes:
    """X.5 canonicalization: YAML formatting/comments never affect the hash."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _compute_manifest(data: dict, events: list[dict]) -> dict:
    type_counts = {"total": len(events)}
    for event_type in _EXPECTED_TYPE_COUNTS:
        type_counts[event_type] = sum(1 for e in events if e.get("event_type") == event_type)

    split_counts = {"dev": 0, "test": 0, "stress": 0}
    for ev in events:
        split = ev.get("split")
        if split in split_counts:
            split_counts[split] += 1

    return {
        "schema_version": data.get("schema_version"),
        "script_version": data.get("script_version"),
        "canonicalization": "json-sorted-keys-utf8",
        "sha256": hashlib.sha256(_canonical_bytes(data)).hexdigest(),
        "frozen_at": None,  # filled in by hand at the actual X.7 step-5 freeze
        "git_tag": None,  # filled in by hand once the freeze commit is tagged
        "counts": type_counts,
        "split_counts": split_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write event_script.manifest.json (X.7 step-5 freeze action only).",
    )
    args = parser.parse_args()

    text = _YAML_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    events = data.get("events", [])

    errors: list[str] = []
    _check_file_hygiene(text, errors)
    for event in events:
        _check_event(event, errors)
    _check_composition(events, errors)
    _check_splits(events, errors)

    if errors:
        print(f"FAIL — {len(errors)} issue(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    manifest = _compute_manifest(data, events)
    print("PASS")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.write_manifest:
        _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {_MANIFEST_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
