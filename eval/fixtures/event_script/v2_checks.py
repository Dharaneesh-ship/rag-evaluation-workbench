"""Event script v2 checks (附录 FR / FS / FU): pure functions used by verify.py.

v2 = the 12 frozen v1 events (splits inherited from git tag event-script-v1,
never recomputed) plus a batch of 18 ``normal`` events with content-free slot
ids ``EVT-V2-NNN``. Slot ``i`` is ``dev`` when ``i % 3 == 1`` else ``test``
(附录 FR.2 plan (b'), ruled in 附录 FS.2). The v2 dev events live in the public
script; the v2 test events live only in the local-only sealed file.

Nothing in this module prints event content. Error strings for v2 events carry
the slot id and the name of the rule that failed, never a probe, a cause, a
segment, a window or a magnitude.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import date, timedelta
from typing import Any, Iterable

V1_FIELDS_ADDED = frozenset({"batch", "exposed", "h2_probes"})

SLOT_ID_RE = re.compile(r"^EVT-V2-(\d{3})$")
YEAR_RE = re.compile(r"(?<!\d)(2022|2023|2024)(?!\d)")

V2_BATCH_SIZE = 18
V2_DEV_COUNT = 6
V2_TEST_COUNT = 12
V2_MAX_EVENTS_PER_REGION = 7  # 附录 FR.2: ceil((30 - 1) / 5) + 1 for B = 18
V2_MIN_IMPLICIT_SHARE = 0.4
V2_MIN_WINDOW_DAYS = 11
#: 附录 FW.3: in each half, upward and downward events are each at least 1/3.
V2_MIN_DIRECTION_SHARE = 1 / 3
V2_MIN_H2_PROBES = 2

#: 附录 FR.5: pre-screen detectability margin for new events (hard gate is 3.0).
V2_MIN_Z_PRESCREEN = 4.5
#: 附录 FU.2 (1) and (2).
GATE_MAX_ROW_SHARE = 0.10
GATE_MAX_EXCESS_RATIO = 0.10

V1_EXPECTED_TYPE_COUNTS = {"normal": 7, "distractor": 3, "unexplained": 2}


# ---------------------------------------------------------------------------
# Hashing (附录 X.5 canonicalization, applied per file)
# ---------------------------------------------------------------------------


def canonical_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def canonical_sha256(data: Any) -> str:
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def combined_sha256(public_sha: str, sealed_sha: str) -> str:
    return hashlib.sha256((public_sha + sealed_sha).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Slot rule
# ---------------------------------------------------------------------------


def slot_of(event_id: Any) -> int | None:
    m = SLOT_ID_RE.match(event_id) if isinstance(event_id, str) else None
    return int(m.group(1)) if m else None


def expected_v2_split(slot: int) -> str:
    return "dev" if slot % 3 == 1 else "test"


# ---------------------------------------------------------------------------
# Years and windows
# ---------------------------------------------------------------------------


def years_in(text: str) -> set[int]:
    return {int(y) for y in YEAR_RE.findall(text)}


def window_years(event: dict) -> set[int]:
    vf = date.fromisoformat(event["valid_from"])
    vt = date.fromisoformat(event["valid_to"])
    return set(range(vf.year, vt.year + 1))


def check_probe_years(label: str, event: dict, field: str, errors: list[str], *, minimum: int) -> None:
    probes = event.get(field)
    if not isinstance(probes, list) or len(probes) < minimum or not all(isinstance(p, str) and p.strip() for p in probes):
        errors.append(f"{label}: {field} must be a list of >= {minimum} non-empty strings")
        return
    try:
        want = window_years(event)
    except (KeyError, ValueError):
        return  # date errors are reported by the base checks
    for n, probe in enumerate(probes, 1):
        got = years_in(probe)
        if not got:
            errors.append(f"{label}: {field}[{n}] states no year")
        elif got != want:
            errors.append(f"{label}: {field}[{n}] states a year set that differs from the event window's years")


def check_leakage_h2(label: str, event: dict, errors: list[str]) -> None:
    facts = (event.get("corpus_evidence") or {}).get("required_facts") or []
    for n, probe in enumerate(event.get("h2_probes") or [], 1):
        for fact in facts:
            if isinstance(fact, str) and fact.strip() and fact.strip() in probe:
                errors.append(f"{label}: h2_probes[{n}] leaks a required_fact verbatim")


def cell_of(event: dict) -> tuple[str | None, date, date] | None:
    """(region or None for all regions, start, end) for events with a data_signal."""
    ds = event.get("data_signal")
    if not ds:
        return None
    return (
        (ds.get("segment") or {}).get("region"),
        date.fromisoformat(event["valid_from"]),
        date.fromisoformat(event["valid_to"]),
    )


def cells_overlap(a: tuple, b: tuple) -> bool:
    (ra, sa, ea), (rb, sb, eb) = a, b
    if ra is not None and rb is not None and ra != rb:
        return False
    return sa <= eb and sb <= ea


# ---------------------------------------------------------------------------
# Structure checks
# ---------------------------------------------------------------------------


def check_v1_immutable(public_v1: list[dict], tag_events: list[dict], errors: list[str]) -> None:
    """v1 events: same ids in the same order, same content except added fields."""
    if [e.get("event_id") for e in public_v1] != [e.get("event_id") for e in tag_events]:
        errors.append("v1 events differ from tag event-script-v1 in ids or order")
        return
    for cur, old in zip(public_v1, tag_events):
        stripped = {k: v for k, v in cur.items() if k not in V1_FIELDS_ADDED}
        if stripped != old:
            errors.append(f"{cur.get('event_id')}: v1 event content changed relative to tag event-script-v1")
        if cur.get("split") != old.get("split"):
            errors.append(f"{cur.get('event_id')}: v1 split must be inherited from tag event-script-v1")
        if cur.get("batch") != "v1":
            errors.append(f"{cur.get('event_id')}: v1 event must carry batch: v1")
        if cur.get("exposed") is not True:
            errors.append(f"{cur.get('event_id')}: v1 event must carry exposed: true")


def check_v2_batch(v2_events: list[dict], errors: list[str]) -> None:
    if len(v2_events) != V2_BATCH_SIZE:
        errors.append(f"v2 batch has {len(v2_events)} events, expected {V2_BATCH_SIZE}")
    slots = []
    for ev in v2_events:
        eid = ev.get("event_id")
        slot = slot_of(eid)
        if slot is None:
            errors.append(f"{eid!r}: v2 event id must match EVT-V2-NNN")
            continue
        slots.append(slot)
        if ev.get("batch") != "v2":
            errors.append(f"{eid}: v2 event must carry batch: v2")
        if ev.get("event_type") != "normal":
            errors.append(f"{eid}: v2 batch carries normal events only")
        if ev.get("split") != expected_v2_split(slot):
            errors.append(f"{eid}: split does not follow the slot rule (i mod 3 == 1 is dev)")
        vf, vt = ev.get("valid_from"), ev.get("valid_to")
        try:
            d0, d1 = date.fromisoformat(vf), date.fromisoformat(vt)
        except (TypeError, ValueError):
            continue
        if d0.year != d1.year:
            errors.append(f"{eid}: window crosses a calendar year")
        if (d1 - d0).days + 1 < V2_MIN_WINDOW_DAYS:
            errors.append(f"{eid}: window shorter than {V2_MIN_WINDOW_DAYS} days")
        ds = ev.get("data_signal") or {}
        if ds.get("metric") != "revenue":
            errors.append(f"{eid}: v2 events use the revenue metric only")
        seg = ds.get("segment") or {}
        if set(seg) != {"region"}:
            errors.append(f"{eid}: v2 events use a region-only segment")
        check_probe_years(eid, ev, "probe_queries", errors, minimum=2)
        check_probe_years(eid, ev, "h2_probes", errors, minimum=V2_MIN_H2_PROBES)
        check_leakage_h2(eid, ev, errors)
    if sorted(slots) != list(range(1, V2_BATCH_SIZE + 1)):
        errors.append("v2 slot numbers must be exactly 1..%d without gaps or repeats" % V2_BATCH_SIZE)

    for half, want in (("dev", V2_DEV_COUNT), ("test", V2_TEST_COUNT)):
        group = [e for e in v2_events if e.get("split") == half]
        if len(group) != want:
            errors.append(f"v2 {half} count = {len(group)}, expected {want}")
        implicit = sum(1 for e in group if "implicit" in ((e.get("corpus_evidence") or {}).get("lexical_divergence") or []))
        if group and implicit < math.ceil(V2_MIN_IMPLICIT_SHARE * len(group)):
            errors.append(f"v2 {half}: implicit-divergence share below {V2_MIN_IMPLICIT_SHARE:.0%}")
        for direction in ("up", "down"):
            k = sum(1 for e in group if (e.get("data_signal") or {}).get("direction") == direction)
            if group and k < math.ceil(V2_MIN_DIRECTION_SHARE * len(group)):
                errors.append(f"v2 {half}: fewer than 1/3 of events go {direction}")
        yrs = set()
        for e in group:
            try:
                yrs |= window_years(e)
            except (KeyError, ValueError):
                pass
        if group and yrs != {2022, 2023, 2024}:
            errors.append(f"v2 {half}: windows do not cover all of 2022, 2023, 2024")


def check_h2_for_v1(
    public_v1: list[dict], overlay: dict[str, list], errors: list[str], *, sealed_present: bool = True
) -> None:
    """Every v1 event needs year-stating h2_probes: public for dev, overlay for test."""
    for ev in public_v1:
        eid = ev.get("event_id")
        merged = dict(ev)
        if ev.get("split") == "test":
            if "h2_probes" in ev:
                errors.append(f"{eid}: v1 test event must not carry h2_probes in the public file (use the sealed overlay)")
            if not sealed_present:
                continue  # the overlay lives in the sealed file, which this machine does not hold
            merged["h2_probes"] = overlay.get(eid)
        check_probe_years(eid, merged, "h2_probes", errors, minimum=V2_MIN_H2_PROBES)
        check_leakage_h2(eid, merged, errors)
    extra = set(overlay) - {e.get("event_id") for e in public_v1 if e.get("split") == "test"}
    if extra:
        errors.append(f"h2_overlay names {len(extra)} id(s) that are not v1 test events")


def check_non_overlap(all_events: list[dict], errors: list[str]) -> None:
    """A v2 event's (region, window) cell may not intersect any other event's."""
    cells = [(ev, cell_of(ev)) for ev in all_events]
    for i, (a, ca) in enumerate(cells):
        if ca is None or a.get("batch") != "v2":
            continue
        for j, (b, cb) in enumerate(cells):
            if i == j or cb is None:
                continue
            if b.get("batch") == "v2" and j < i:
                continue  # report each v2/v2 pair once
            if cells_overlap(ca, cb):
                errors.append(f"{a.get('event_id')}: cell overlaps another event's cell ({b.get('event_id')})")


def check_region_cap(all_events: list[dict], errors: list[str]) -> None:
    counts: dict[str, int] = {}
    for ev in all_events:
        region = ((ev.get("data_signal") or {}).get("segment") or {}).get("region")
        if region:
            counts[region] = counts.get(region, 0) + 1
    for region, n in counts.items():
        if n > V2_MAX_EVENTS_PER_REGION:
            errors.append(f"region {region!r} appears in {n} events, v2 max is {V2_MAX_EVENTS_PER_REGION}")


def check_plan_conformity(plan: dict, v2_events: list[dict], errors: list[str]) -> None:
    """Each v2 event must match its slot's pre-committed structural attributes."""
    slots = (plan or {}).get("slots") or {}
    by_id = {e.get("event_id"): e for e in v2_events}
    if set(slots) != set(by_id):
        errors.append("slot plan and v2 events name different slot ids")
    for sid, want in slots.items():
        ev = by_id.get(sid)
        if ev is None:
            continue
        ds = ev.get("data_signal") or {}
        try:
            d0, d1 = date.fromisoformat(ev["valid_from"]), date.fromisoformat(ev["valid_to"])
        except (KeyError, ValueError):
            continue
        got = {
            "split": ev.get("split"),
            "year": d0.year,
            "region": (ds.get("segment") or {}).get("region"),
            "shape": ds.get("shape"),
            "direction": ds.get("direction"),
            "valid_from": ev["valid_from"],
            "valid_to": ev["valid_to"],
            "magnitude": ds.get("magnitude"),
            "implicit": "implicit" in ((ev.get("corpus_evidence") or {}).get("lexical_divergence") or []),
        }
        for key, val in got.items():
            if want.get(key) != val:
                errors.append(f"{sid}: deviates from the committed slot plan on '{key}'")


# ---------------------------------------------------------------------------
# FU.2 gate
# ---------------------------------------------------------------------------


def _quantile(sorted_vals: list[float], p: float) -> float:
    idx = p * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def iqr_bounds(values: Iterable[float]) -> tuple[float, float]:
    s = sorted(values)
    q1, q3 = _quantile(s, 0.25), _quantile(s, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _to_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def revenue_bounds(rows: list[dict]) -> dict[str, float]:
    """Raw-table bounds and the post-dedup + median-fill bounds (附录 FT.2, FU.2 (3))."""
    raw = [v for v in (_to_float(r.get("revenue")) for r in rows) if v is not None]
    raw_lo, raw_hi = iqr_bounds(raw)
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in rows:
        key = tuple(r.values())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    vals = [_to_float(r.get("revenue")) for r in deduped]
    present = [v for v in vals if v is not None]
    med = statistics.median(present)
    clean_lo, clean_hi = iqr_bounds([med if v is None else v for v in vals])
    return {
        "raw_lower": raw_lo,
        "raw_upper": raw_hi,
        "clean_lower": clean_lo,
        "clean_upper": clean_hi,
        "upper": min(raw_hi, clean_hi),  # the stricter of the two
        "lower": raw_lo,
    }


def baseline_is_clean(rows: list[dict], baseline: list[dict]) -> str | None:
    """Row-for-row alignment between the event render and the no-event render."""
    if len(rows) != len(baseline):
        return "baseline row count differs from the render"
    keys = ("date", "region", "product", "channel", "rep_id")
    for a, b in zip(rows, baseline):
        if any(a.get(k) != b.get(k) for k in keys):
            return "baseline rows are not aligned with the render"
    return None


def gate_event(event: dict, rows: list[dict], baseline: list[dict], bounds: dict[str, float]) -> dict[str, Any]:
    """Per-event FU.2 measurements, with the 附录 FW.2 correction to item (2).

    (1) share of window rows above the upper bound U (all non-blank window rows).
    (2) over window rows whose *baseline* value is <= U only (rows cleaner would
        not clip without the event; this drops the renderer's injected x10
        quality-noise rows): sum(max(0, x_event - U)) / sum(x_event - x_baseline).
        Rows whose baseline already exceeds U are not in the numerator or the
        denominator; their count is reported as ``excluded_baseline_above_upper``.
    Aggregates only; nothing here names content.
    """
    ds = event["data_signal"]
    seg = ds.get("segment") or {}
    vf, vt = event["valid_from"], event["valid_to"]
    upper, lower = bounds["upper"], bounds["lower"]
    n = share_hits = below = excluded = 0
    excess = lift = 0.0
    for r, b in zip(rows, baseline):
        if not (vf <= r["date"] <= vt) or any(r.get(k) != v for k, v in seg.items()):
            continue
        rv, bv = _to_float(r.get("revenue")), _to_float(b.get("revenue"))
        if rv is None:
            continue
        n += 1
        if rv > upper:
            share_hits += 1
        if rv < lower:
            below += 1
        if bv is None:
            continue
        if bv > upper:
            excluded += 1
            continue
        excess += max(0.0, rv - upper)
        lift += rv - bv
    share = share_hits / n if n else math.nan
    if ds.get("direction") == "up" and lift > 0:
        ratio: float | None = excess / lift
    else:
        ratio = None  # no upward lift over the no-event baseline: nothing for an upper clip to erase
    return {
        "n": n,
        "row_share_above_upper": share,
        "below_lower_share": below / n if n else math.nan,
        "excess": excess,
        "lift": lift,
        "excess_ratio": ratio,
        "excluded_baseline_above_upper": excluded,
        "direction": ds.get("direction"),
    }


def gate_failures(label: str, m: dict[str, Any]) -> list[str]:
    out = []
    if not (m["n"] > 0):
        out.append(f"{label}: no non-blank window rows in the render")
        return out
    if m["row_share_above_upper"] > GATE_MAX_ROW_SHARE:
        out.append(f"{label}: FU.2(1) share of window rows above the upper bound exceeds {GATE_MAX_ROW_SHARE:.0%}")
    if m["excess_ratio"] is not None and m["excess_ratio"] > GATE_MAX_EXCESS_RATIO:
        out.append(f"{label}: FU.2(2, corrected FW.2) clipped excess / lift exceeds {GATE_MAX_EXCESS_RATIO:.0%}")
    return out
