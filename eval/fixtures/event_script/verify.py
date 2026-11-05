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
    python verify.py --data-path sales_data.csv  # step-6 detectability gate

``--write-manifest`` is the X.7 step-5 freeze action — do not run it before
the three-party content review in step 4 has actually happened. It fills in
``sha256``/``counts``/``split_counts`` but deliberately leaves
``frozen_at``/``git_tag`` for whoever performs the actual freeze ceremony to
fill in by hand, since a git tag doesn't exist yet when this file is edited.

``--data-path`` is the X.7 step-6 acceptance gate (附录 AF.1/AF.3, folding in
AB.4's detectability check and AA.2/AE.1's IQR-clipping check). It does not
run at step 4/5 — there is no rendered CSV to check yet, and the hard
sequencing constraint (计划 7.3) forbids rendering before the script is
frozen. Once MAEDA renders the final ``sales_data.csv``, this flag re-checks
every event's ``data_signal`` against the real distribution: is it
statistically detectable (z >= 3, or z >= 5 for ``unexplained`` events,
附录 AB.4), and does its peak stay under the IQR upper bound that
``handle_outliers`` would otherwise clip it at (附录 AA.2)? A failure here
blocks step 6 — per 附录 AF.1, only *rendering* parameters (row count, noise
coefficient, the ``units`` price-band generation) may be adjusted to fix it;
the frozen script fields (segment, shape, magnitude, dates, wording) may not
be silently loosened to pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 附录 CP.4: verify.py used to read event_script.yaml itself. It now goes
# through the one sanctioned reader, rag.eval.split_guard, like every other
# in-repository consumer — verifying the frozen script legitimately needs
# every split (the manifest hash is over the whole document), so it declares
# that rather than quietly helping itself. tests/test_split_guard.py fails if
# a second direct reader ever appears.
from rag.eval.split_guard import (  # noqa: E402
    PROBE_DIGEST_LIST_PATH,
    build_probe_digest_list,
    load_event_script_source,
    load_sealed_source,
    load_v1_tag_events,
)

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import v2_checks as v2  # noqa: E402

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

# Step-6 detectability gate (附录 AB.4 / AC.5 / AF.1) — only evaluated when
# --data-path is given, never at step 4/5.
_MIN_Z_DEFAULT = 3.0
_MIN_Z_UNEXPLAINED = 5.0
# 附录 AF.1/AF.2: mean of the "1±magnitude" time weight across the event
# window. step = constant offset for the whole window (weight 1.0); ramp and
# spike are triangular (linear-up or peak-at-midpoint), averaging to ~0.5.
_SHAPE_MEAN_WEIGHT = {"step": 1.0, "ramp": 0.5, "spike": 0.5}


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


def _iqr_bounds(values: list[float]) -> tuple[float, float]:
    """附录 AE.2's exact formula: linear-interpolated quantiles, matching both
    pandas' ``Series.quantile()`` default and cleaner's ``_compute_bounds()``
    (``mcp_server/tools/handle_outliers.py``, ``method="iqr"``). Must stay
    byte-for-byte identical to AE.2, or this check could say "safe" while
    cleaner's real IQR clipping disagrees.
    """
    s = sorted(values)

    def quantile(p: float) -> float:
        idx = p * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    q1, q3 = quantile(0.25), quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _robust_rel_std(values: list[float]) -> float:
    """Approximates 附录 AC.3's "quality-noise-stripped" rel_std without
    access to a noise-free render: AC.3 found that generate_demo_data.py's
    1%-of-rows x10 outlier injection, not the 20% per-row business noise,
    dominates the raw sample std. An IQR-based robust std (IQR / 1.349, the
    standard normal-equivalent scale) is insensitive to that ~1% contaminated
    tail the same way trimming those rows out would be, so it stands in for
    re-running the generator without its quality-noise path. If this
    consistently over/under-shoots what AC.3's clean-vs-dirty comparison
    would give on the actual final data, replace it with a real ablation
    render instead of tightening this approximation further.
    """
    s = sorted(values)

    def quantile(p: float) -> float:
        idx = p * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    iqr = quantile(0.75) - quantile(0.25)
    robust_std = iqr / 1.349
    mean = statistics.fmean(values)
    return robust_std / mean if mean else math.inf


def _load_metric_rows(data_path: Path) -> list[dict[str, str]]:
    with data_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _segment_matches(row: dict[str, str], segment: dict[str, str]) -> bool:
    return all(row.get(key) == value for key, value in segment.items())


def _numeric_metric_values(records: list[dict[str, str]], metric: str) -> list[float]:
    """Read ``metric`` off each record, skipping blanks.

    3% of ``revenue`` rows are deliberately blank — that's the quality-noise
    missingness the generator injects by design (X-V3), not a rendering
    defect (附录 AJ.2). A missing reading carries no information about
    whether the business signal moved, so it should drop out of the sample
    rather than crash the float() conversion or get miscounted as a 0.0.
    """
    values = []
    for r in records:
        raw = r.get(metric)
        if raw is None or raw == "":
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _check_detectability_and_clipping(
    events: list[dict], rows: list[dict[str, str]], errors: list[str]
) -> None:
    """X.7 step-6 gate: 附录 AB.4 (z >= 3 / z >= 5 for unexplained) merged
    with 附录 AA.2/AE.1 (peak must stay under the whole-table IQR upper
    bound). Both halves must use the *same* rendered CSV, per 附录 AC.6.
    """
    for event in events:
        data_signal = event.get("data_signal")
        if not data_signal:
            continue  # distractor events carry no data_signal to check

        event_id = event.get("event_id")
        metric = data_signal.get("metric")
        segment = data_signal.get("segment") or {}
        shape = data_signal.get("shape")
        magnitude = data_signal.get("magnitude")
        try:
            vf = date.fromisoformat(event.get("valid_from", ""))
            vt = date.fromisoformat(event.get("valid_to", ""))
        except ValueError as exc:
            errors.append(f"{event_id}: invalid valid_from/valid_to for detectability check: {exc}")
            continue

        if rows and metric not in rows[0]:
            errors.append(f"{event_id}: metric {metric!r} is not a column in --data-path")
            continue

        segment_rows = [r for r in rows if _segment_matches(r, segment)]
        segment_values = _numeric_metric_values(segment_rows, metric)
        all_values = _numeric_metric_values(rows, metric)

        window_rows = [r for r in segment_rows if vf <= date.fromisoformat(r["date"]) <= vt]
        window_values = _numeric_metric_values(window_rows, metric)
        n = len(window_values)

        if not segment_values or n == 0:
            errors.append(f"{event_id}: zero non-blank rows match segment {segment!r} / window in --data-path")
            continue

        rel_std = _robust_rel_std(segment_values)
        weight = _SHAPE_MEAN_WEIGHT.get(shape, 0.5)
        z = magnitude * weight * math.sqrt(n) / rel_std if rel_std else math.inf

        threshold = _MIN_Z_UNEXPLAINED if event.get("event_type") == "unexplained" else _MIN_Z_DEFAULT
        if z < threshold:
            errors.append(
                f"{event_id}: detectability z={z:.2f} < required {threshold} "
                f"(n={n}, rel_std={rel_std:.3f}, shape={shape}, magnitude={magnitude})"
            )

        peak = statistics.fmean(window_values)
        _, upper = _iqr_bounds(all_values)
        if peak > upper:
            errors.append(
                f"{event_id}: window peak {peak:.1f} exceeds whole-table IQR upper bound "
                f"{upper:.1f} — handle_outliers would clip this event's signal"
            )


_SEALED_DIR = _ROOT / "eval" / "sealed"
_PLAN_PATH = _SEALED_DIR / "slot_plan.yaml"
_RECEIPT_PATH = _SEALED_DIR / "RECEIPT.json"


def _run_v2(args, text: str, data: dict) -> int:
    """Script version 2 (附录 FR/FS/FU). Prints aggregates and rule names only."""
    import yaml

    errors: list[str] = []
    notes: dict = {}
    public_events = data.get("events", [])
    sealed_text, sealed = load_sealed_source(reason="verify.py v2: hash, structure and gate over the sealed test events")
    sealed_events = sealed.get("events", []) if sealed else []
    overlay = (sealed or {}).get("h2_overlay") or {}

    _check_file_hygiene(text, errors)
    if sealed_text:
        sealed_errs: list[str] = []
        _check_file_hygiene(sealed_text, sealed_errs)
        errors.extend("sealed file: " + e for e in sealed_errs)
    elif not args.public_only:
        errors.append("sealed v2 test file is absent (use --public-only to check the public part alone)")

    v1_events = [e for e in public_events if e.get("batch") == "v1"]
    v2_public = [e for e in public_events if e.get("batch") == "v2"]
    if len(v1_events) + len(v2_public) != len(public_events):
        errors.append("every public event must carry batch: v1 or v2")
    if [e.get("batch") for e in public_events] != ["v1"] * len(v1_events) + ["v2"] * len(v2_public):
        errors.append("public file must list the v1 events first, then v2")

    try:
        v2.check_v1_immutable(v1_events, load_v1_tag_events(), errors)
    except RuntimeError as exc:
        errors.append(f"cannot inherit v1 splits: {exc}")

    all_v2 = v2_public + sealed_events
    for event in public_events + sealed_events:
        _check_event(event, errors)
    if len(v1_events) == 12:
        counts = {t: sum(1 for e in v1_events if e.get("event_type") == t) for t in v2.V1_EXPECTED_TYPE_COUNTS}
        if counts != v2.V1_EXPECTED_TYPE_COUNTS:
            errors.append("v1 composition differs from X.3 (12/7/3/2)")
    else:
        errors.append(f"expected 12 v1 events, found {len(v1_events)}")
    v2.check_h2_for_v1(v1_events, overlay, errors)

    if sealed_events or not args.public_only:
        v2.check_v2_batch(all_v2, errors)
        v2.check_non_overlap(v1_events + all_v2, errors)
        v2.check_region_cap(v1_events + all_v2, errors)
    months = {e["valid_from"][:7] for e in v1_events + all_v2 if isinstance(e.get("valid_from"), str)}
    if len(months) < _MIN_DISTINCT_MONTHS:
        errors.append(f"events span {len(months)} distinct months, need >= {_MIN_DISTINCT_MONTHS}")

    plan_sha = None
    if _PLAN_PATH.is_file():
        plan = yaml.safe_load(_PLAN_PATH.read_text(encoding="utf-8"))
        plan_sha = v2.canonical_sha256(plan)
        if sealed_events:
            v2.check_plan_conformity(plan, all_v2, errors)
        if _RECEIPT_PATH.is_file():
            rec = json.loads(_RECEIPT_PATH.read_text(encoding="utf-8"))
            if rec.get("slot_plan_sha256") != plan_sha:
                errors.append("slot plan sha256 differs from RECEIPT.json")
    elif not args.public_only:
        errors.append("slot plan is absent")

    pub_sha = v2.canonical_sha256(data)
    seal_sha = v2.canonical_sha256(sealed) if sealed else None
    type_counts = {"total": len(public_events) + len(sealed_events)}
    for t in v2.V1_EXPECTED_TYPE_COUNTS:
        type_counts[t] = sum(1 for e in public_events + sealed_events if e.get("event_type") == t)
    split_counts = {"dev": 0, "test": 0, "stress": 0}
    for e in public_events + sealed_events:
        if e.get("split") in split_counts:
            split_counts[e["split"]] += 1
    h2_pool = sum(1 for e in all_v2 if e.get("split") == "test" and e.get("event_type") == "normal")
    manifest = {
        "schema_version": data.get("schema_version"),
        "script_version": data.get("script_version"),
        "canonicalization": "json-sorted-keys-utf8",
        "sha256": pub_sha,
        "public_sha256": pub_sha,
        "sealed_sha256": seal_sha,
        "combined_sha256": v2.combined_sha256(pub_sha, seal_sha) if seal_sha else None,
        "slot_plan_sha256": plan_sha,
        "v1_tag": "event-script-v1",
        "frozen_at": None,
        "git_tag": None,
        "counts": type_counts,
        "split_counts": split_counts,
        "h2_pool_test_normal": h2_pool,
    }

    if args.data_path is not None:
        rows = _load_metric_rows(args.data_path)
        legacy_errs: list[str] = []
        _check_detectability_and_clipping(public_events + sealed_events, rows, legacy_errs)
        errors.extend(_redact_ids(legacy_errs, {e["event_id"] for e in sealed_events}))
        if args.baseline_path is None:
            errors.append("FU.2 gate needs --baseline-path (a no-event render of the same script, row for row)")
        elif all_v2:
            baseline = _load_metric_rows(args.baseline_path)
            bad = v2.baseline_is_clean(rows, baseline)
            if bad:
                errors.append("STOP, FU.2(2) baseline not clean: " + bad)
            else:
                bounds = v2.revenue_bounds(rows)
                gate = []
                for e in all_v2:
                    m = v2.gate_event(e, rows, baseline, bounds)
                    z = _event_z(e, rows)
                    if z is not None and z < v2.V2_MIN_Z_PRESCREEN:
                        errors.append(f"{e['event_id']}: z below the v2 pre-screen {v2.V2_MIN_Z_PRESCREEN}")
                    errors.extend(v2.gate_failures(e["event_id"], m))
                    gate.append((m, z))
                ratios = [m["excess_ratio"] for m, _ in gate if m["excess_ratio"] is not None]
                notes["gate"] = {
                    "events": len(gate),
                    "upper_bound_used": round(bounds["upper"], 3),
                    "upper_raw": round(bounds["raw_upper"], 3),
                    "upper_clean": round(bounds["clean_upper"], 3),
                    "lower_bound_reported": round(bounds["lower"], 3),
                    "min_z": round(min(z for _, z in gate if z is not None), 2),
                    "max_row_share_above_upper": round(max(m["row_share_above_upper"] for m, _ in gate), 4),
                    "max_below_lower_share": round(max(m["below_lower_share"] for m, _ in gate), 4),
                    "events_with_upward_lift": len(ratios),
                    "excluded_rows_baseline_above_upper_total": sum(m["excluded_baseline_above_upper"] for m, _ in gate),
                    "max_excess_ratio": round(max(ratios), 4) if ratios else None,
                }

    if errors:
        print(f"FAIL: {len(errors)} issue(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        if notes:
            print(json.dumps(notes, indent=2), file=sys.stderr)
        return 1

    print("PASS")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if notes:
        print(json.dumps(notes, indent=2))
    if args.freeze:
        from datetime import datetime, timezone

        manifest["frozen_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["git_tag"] = "event-script-v2"
        _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {_MANIFEST_PATH}")
        old = json.loads(_RECEIPT_PATH.read_text(encoding="utf-8"))
        old.update({"status": "frozen", "frozen_at": manifest["frozen_at"], "git_tag": "event-script-v2"})
        _RECEIPT_PATH.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {_RECEIPT_PATH}")
    if args.write_receipt:
        old = json.loads(_RECEIPT_PATH.read_text(encoding="utf-8")) if _RECEIPT_PATH.is_file() else {}
        receipt = {
            "status": "pre-freeze",
            "superseded_slot_plans": old.get("superseded_slot_plans", []),
            "script_version": "2",
            "slot_plan_sha256": plan_sha,
            "sealed_sha256": seal_sha,
            "public_sha256": pub_sha,
            "combined_sha256": manifest["combined_sha256"],
            "counts": {"v2_dev": sum(1 for e in all_v2 if e["split"] == "dev"), "v2_test": h2_pool},
        }
        _RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {_RECEIPT_PATH}")
    if args.write_digests and sealed:
        listing = build_probe_digest_list(data, sealed)
        PROBE_DIGEST_LIST_PATH.write_text(
            json.dumps({"algorithm": "sha256(normalize_query(text))", "digests": listing}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {PROBE_DIGEST_LIST_PATH} ({len(listing)} digests)")
    return 0


def _redact_ids(msgs: list[str], sealed_ids: set[str]) -> list[str]:
    return list(msgs)  # v2 test ids are content-free slot ids; nothing to redact


def _event_z(event: dict, rows: list[dict[str, str]]) -> float | None:
    ds = event.get("data_signal") or {}
    seg = ds.get("segment") or {}
    metric = ds.get("metric")
    seg_rows = [r for r in rows if _segment_matches(r, seg)]
    seg_vals = _numeric_metric_values(seg_rows, metric)
    win = _numeric_metric_values(
        [r for r in seg_rows if event["valid_from"] <= r["date"] <= event["valid_to"]], metric
    )
    if not seg_vals or not win:
        return None
    rel = _robust_rel_std(seg_vals)
    return ds["magnitude"] * _SHAPE_MEAN_WEIGHT.get(ds["shape"], 0.5) * math.sqrt(len(win)) / rel if rel else math.inf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write event_script.manifest.json (X.7 step-5 freeze action only).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to a rendered sales_data.csv — runs the X.7 step-6 "
        "detectability + IQR-clipping gate (附录 AF.1/AF.3). Omit for the "
        "step 4/5 schema-only checks.",
    )
    parser.add_argument("--baseline-path", type=Path, default=None,
                        help="v2: a no-event render of the same script (FU.2(2) baseline).")
    parser.add_argument("--public-only", action="store_true",
                        help="v2: check the public file alone (no sealed file on this machine).")
    parser.add_argument("--write-receipt", action="store_true",
                        help="v2: write eval/sealed/RECEIPT.json (hashes only).")
    parser.add_argument("--freeze", action="store_true",
                        help="v2: write event_script.manifest.json (hash chain) and mark RECEIPT.json frozen.")
    parser.add_argument("--write-digests", action="store_true",
                        help="v2: write the hash-only test probe digest list.")
    args = parser.parse_args()

    text, data = load_event_script_source(
        reason="verify.py structural/manifest verification of the frozen script"
    )
    if str(data.get("script_version")) == "2":
        return _run_v2(args, text, data)
    events = data.get("events", [])

    errors: list[str] = []
    _check_file_hygiene(text, errors)
    for event in events:
        _check_event(event, errors)
    _check_composition(events, errors)
    _check_splits(events, errors)

    if args.data_path is not None:
        rows = _load_metric_rows(args.data_path)
        _check_detectability_and_clipping(events, rows, errors)

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
