"""Tests for event script v2 (附录 FR / FS / FU): checks, gate, digests, suites, renderer.

Synthetic events only. No test-split content appears in this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ES = _ROOT / "eval" / "fixtures" / "event_script"


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2 = _load("v2_checks", _ES / "v2_checks.py")
render = _load("render_v2", _ES / "render_v2.py")
suite = _load("build_suite", _ROOT / "eval" / "fixtures" / "tb5_retrieval" / "build_suite.py")

from rag.eval import split_guard as sg  # noqa: E402


def _v1_tag_present() -> bool:
    try:
        sg.load_v1_tag_events()
        return True
    except RuntimeError:
        return False


def _require_v1_tag() -> None:
    """Skip locally when the tag is missing, but never in CI: CI must fetch tags (fetch-depth: 0)."""
    import os

    if not _v1_tag_present():
        if os.environ.get("CI"):
            pytest.fail("tag event-script-v1 is missing in CI; the checkout must fetch tags (fetch-depth: 0)")
        pytest.skip("tag event-script-v1 not available in this clone")


def _ev(slot: int, *, region="North", vf="2023-02-01", vt="2023-02-20", year_in_probe="2023", **kw):
    e = {
        "event_id": f"EVT-V2-{slot:03d}",
        "event_type": "normal",
        "split": v2.expected_v2_split(slot),
        "batch": "v2",
        "valid_from": vf,
        "valid_to": vt,
        "data_signal": {"metric": "revenue", "segment": {"region": region}, "direction": "down", "magnitude": 0.3, "shape": "step"},
        "corpus_evidence": {"required_facts": ["alpha fact"], "lexical_divergence": ["implicit"]},
        "probe_queries": [f"q one {year_in_probe}", f"q two {year_in_probe}"],
        "h2_probes": [f"h one {year_in_probe}", f"h two {year_in_probe}"],
    }
    e.update(kw)
    return e


# ---------------------------------------------------------------- slot rule


def test_slot_rule_is_one_dev_to_two_test():
    splits = [v2.expected_v2_split(i) for i in range(1, 19)]
    assert splits.count("dev") == 6 and splits.count("test") == 12
    assert [i for i in range(1, 19) if v2.expected_v2_split(i) == "dev"] == [1, 4, 7, 10, 13, 16]


def test_slot_ids_are_content_free_and_strict():
    assert v2.slot_of("EVT-V2-007") == 7
    assert v2.slot_of("EVT-2023-07-EAST-STOCKOUT") is None
    assert v2.slot_of("EVT-V2-7") is None


# ------------------------------------------------------------------- years


def test_probe_must_state_year_matching_window():
    errs: list[str] = []
    v2.check_probe_years("x", _ev(1), "h2_probes", errs, minimum=2)
    assert errs == []
    errs = []
    v2.check_probe_years("x", _ev(1, h2_probes=["no year here", "h 2023"]), "h2_probes", errs, minimum=2)
    assert any("states no year" in e for e in errs)
    errs = []
    v2.check_probe_years("x", _ev(1, h2_probes=["h 2022", "h 2023"]), "h2_probes", errs, minimum=2)
    assert any("differs" in e for e in errs)


def test_year_regex_does_not_match_inside_longer_numbers():
    assert v2.years_in("SKU-20231 and 2023年") == {2023}
    assert v2.years_in("order 120245") == set()


def test_v2_window_may_not_cross_a_year():
    errs: list[str] = []
    v2.check_v2_batch([_ev(1, vf="2023-12-20", vt="2024-01-05")], errs)
    assert any("crosses a calendar year" in e for e in errs)


# ------------------------------------------------------------ batch + cells


def _full_batch():
    evs = []
    years = [2022, 2023, 2024]
    regions = ["North", "South", "East", "West", "Central"]
    for i in range(1, 19):
        y = years[(i // 3) % 3] if i % 3 else years[i % 9 % 3]
        evs.append(_ev(i, region=regions[i % 5], vf=f"{y}-0{1 + i % 8}-01", vt=f"{y}-0{1 + i % 8}-20", year_in_probe=str(y)))
    # cover all three years in each half; mix directions (>= 1/3 each)
    dev = [e for e in evs if e["split"] == "dev"]
    test = [e for e in evs if e["split"] == "test"]
    for grp in (dev, test):
        for k, e in enumerate(grp):
            e["data_signal"]["direction"] = "up" if k % 2 else "down"
            y = years[k % 3]
            e["valid_from"], e["valid_to"] = f"{y}-0{1 + k % 8}-01", f"{y}-0{1 + k % 8}-20"
            e["probe_queries"] = [f"a {y}", f"b {y}"]
            e["h2_probes"] = [f"c {y}", f"d {y}"]
    return evs


def test_full_synthetic_batch_passes():
    errs: list[str] = []
    v2.check_v2_batch(_full_batch(), errs)
    assert errs == []


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda evs: evs.pop(), "expected 18"),
        (lambda evs: evs[0].__setitem__("split", "test"), "slot rule"),
        (lambda evs: evs[0].__setitem__("event_type", "distractor"), "normal events only"),
        (lambda evs: evs[0]["data_signal"].__setitem__("metric", "units"), "revenue metric only"),
        (lambda evs: evs[0]["data_signal"].__setitem__("segment", {}), "region-only"),
        (lambda evs: evs[0].__setitem__("batch", "v1"), "batch: v2"),
        (lambda evs: [e["data_signal"].__setitem__("direction", "down") for e in evs], "fewer than 1/3 of events go up"),
        (lambda evs: [e["data_signal"].__setitem__("direction", "up") for e in evs], "fewer than 1/3 of events go down"),
    ],
)
def test_batch_violations_are_reported(mutate, needle):
    evs = _full_batch()
    mutate(evs)
    errs: list[str] = []
    v2.check_v2_batch(evs, errs)
    assert any(needle in e for e in errs), errs


def test_implicit_share_floor_per_half():
    evs = _full_batch()
    for e in evs:
        e["corpus_evidence"]["lexical_divergence"] = ["synonym"]
    errs: list[str] = []
    v2.check_v2_batch(evs, errs)
    assert sum("implicit-divergence share" in e for e in errs) == 2


def test_non_overlap_and_global_cells():
    a = _ev(1, region="North", vf="2023-02-01", vt="2023-02-20")
    b = _ev(2, region="North", vf="2023-02-15", vt="2023-03-05")
    c = _ev(3, region="South", vf="2023-02-15", vt="2023-03-05")
    glob = {"event_id": "G", "batch": "v1", "valid_from": "2023-03-01", "valid_to": "2023-03-31",
            "data_signal": {"segment": {}}}
    errs: list[str] = []
    v2.check_non_overlap([a, b, c], errs)
    assert len(errs) == 1 and "EVT-V2-002" in errs[0] or "EVT-V2-001" in errs[0]
    errs = []
    v2.check_non_overlap([c, glob], errs)
    assert len(errs) == 1


def test_region_cap():
    evs = [_ev(i, region="West") for i in range(1, 9)]
    errs: list[str] = []
    v2.check_region_cap(evs, errs)
    assert errs and "max is 7" in errs[0]


def test_plan_conformity_detects_any_structural_drift():
    ev = _ev(1)
    ds = ev["data_signal"]
    plan = {"slots": {ev["event_id"]: {
        "split": ev["split"], "year": 2023, "region": "North", "shape": "step", "direction": "down",
        "valid_from": ev["valid_from"], "valid_to": ev["valid_to"], "magnitude": ds["magnitude"], "implicit": True}}}
    errs: list[str] = []
    v2.check_plan_conformity(plan, [ev], errs)
    assert errs == []
    ev["data_signal"]["magnitude"] = 0.5
    v2.check_plan_conformity(plan, [ev], errs)
    assert any("magnitude" in e for e in errs)


# --------------------------------------------------------------- v1 intact


def test_v1_immutable_allows_only_added_fields():
    old = [{"event_id": "A", "split": "dev", "probe_queries": ["p"]}]
    cur = [{"event_id": "A", "split": "dev", "probe_queries": ["p"], "batch": "v1", "exposed": True, "h2_probes": ["x 2023"]}]
    errs: list[str] = []
    v2.check_v1_immutable(cur, old, errs)
    assert errs == []
    cur[0]["probe_queries"] = ["p changed"]
    v2.check_v1_immutable(cur, old, errs)
    assert any("content changed" in e for e in errs)


def test_v1_test_events_may_not_carry_public_h2_probes():
    ev = [{"event_id": "A", "split": "test", "valid_from": "2023-01-01", "valid_to": "2023-01-09", "h2_probes": ["a 2023", "b 2023"]}]
    errs: list[str] = []
    v2.check_h2_for_v1(ev, {}, errs)
    assert any("sealed overlay" in e for e in errs)


def test_real_v1_events_match_the_tag():
    _require_v1_tag()
    tag_events = sg.load_v1_tag_events()
    pub = sg.load_event_script_document(reason="test: v1 immutability")["events"]
    errs: list[str] = []
    v2.check_v1_immutable([e for e in pub if e.get("batch") == "v1"], tag_events, errs)
    assert errs == []


# -------------------------------------------------------------------- gate


def _rows(values, region="North", start_day=1):
    return [
        {"date": f"2023-02-{start_day + i:02d}", "region": region, "product": "P", "channel": "C", "rep_id": str(i),
         "revenue": "" if v is None else str(v)}
        for i, v in enumerate(values)
    ]


def test_bounds_take_the_stricter_of_raw_and_clean():
    rows = _rows([100.0] * 20 + [None] * 5 + [500.0] * 3)
    b = v2.revenue_bounds(rows)
    assert b["upper"] == min(b["raw_upper"], b["clean_upper"])


def test_dedup_changes_the_clean_bound():
    base = _rows([100, 110, 120, 130, 140, 150, 160, 170])
    dup = base + [dict(base[7])] * 6  # exact duplicates of one row
    assert v2.revenue_bounds(dup)["raw_upper"] != v2.revenue_bounds(dup)["clean_upper"]


def test_baseline_alignment_is_enforced():
    r = _rows([1.0, 2.0])
    assert v2.baseline_is_clean(r, r) is None
    assert "count" in v2.baseline_is_clean(r, r[:1])
    other = _rows([1.0, 2.0])
    other[1]["region"] = "South"
    assert "aligned" in v2.baseline_is_clean(r, other)


def _ds(direction):
    return {"valid_from": "2023-02-01", "valid_to": "2023-02-28",
            "data_signal": {"metric": "revenue", "segment": {"region": "North"}, "direction": direction, "magnitude": 0.3, "shape": "step"}}


def test_gate_down_event_has_no_lift_to_clip():
    base = _rows([100.0] * 10)
    rows = _rows([70.0] * 10)
    m = v2.gate_event(_ds("down"), rows, base, {"upper": 120.0, "lower": 50.0})
    assert m["excess_ratio"] is None and m["row_share_above_upper"] == 0 and v2.gate_failures("e", m) == []


def test_gate_up_event_fails_when_event_pushes_rows_over_the_bound():
    base = _rows([100.0] * 10)
    rows = _rows([130.0] * 10)
    m = v2.gate_event(_ds("up"), rows, base, {"upper": 110.0, "lower": 50.0})
    assert m["row_share_above_upper"] == 1.0
    assert m["excess_ratio"] == pytest.approx(20 / 30)
    assert len(v2.gate_failures("e", m)) == 2


def test_gate_up_event_passes_when_under_the_bound():
    base = _rows([100.0] * 10)
    rows = _rows([108.0] * 10)
    m = v2.gate_event(_ds("up"), rows, base, {"upper": 120.0, "lower": 50.0})
    assert v2.gate_failures("e", m) == []


def test_gate_excludes_rows_whose_baseline_already_exceeds_the_bound():
    """附录 FW.2: injected x10 outlier rows must not count against the event."""
    base = _rows([100.0] * 9 + [1000.0])
    rows = _rows([108.0] * 9 + [10800.0])  # the outlier row is clipped with or without the event
    m = v2.gate_event(_ds("up"), rows, base, {"upper": 120.0, "lower": 50.0})
    assert m["excluded_baseline_above_upper"] == 1
    assert m["excess"] == 0 and m["lift"] == pytest.approx(72.0)
    assert m["excess_ratio"] == 0
    assert m["row_share_above_upper"] == pytest.approx(0.1)  # item (1) still sees every row


def test_gate_counts_only_window_segment_rows():
    base = _rows([100.0] * 4) + _rows([100.0] * 4, region="South")
    rows = _rows([100.0] * 4) + _rows([999.0] * 4, region="South")
    m = v2.gate_event(_ds("up"), rows, base, {"upper": 120.0, "lower": 50.0})
    assert m["n"] == 4 and m["row_share_above_upper"] == 0


# -------------------------------------------------------------- split_guard


def test_digest_list_covers_events_that_are_not_in_the_public_file(tmp_path):
    secret = "a synthetic sealed probe 2023"
    script = tmp_path / "es.yaml"
    script.write_text('schema_version: "1"\nscript_version: "2"\nevents: []\n', encoding="utf-8")
    (tmp_path / sg.PROBE_DIGEST_LIST_PATH.name).write_text(
        json.dumps({"digests": [sg.digest_of(secret)]}), encoding="utf-8")
    sg.clear_cache()
    assert sg.is_test_probe_query(secret, path=script)
    assert sg.is_test_probe_query("  A SYNTHETIC sealed probe 2023 ", path=script)
    assert not sg.is_test_probe_query("some other query", path=script)
    sg.clear_cache()


def test_h2_probes_of_test_events_are_recognised_and_dev_ones_are_not(tmp_path):
    script = tmp_path / "es.yaml"
    script.write_text(
        'events:\n'
        '  - {event_id: T, split: test, probe_queries: ["tq 2023"], h2_probes: ["th 2023"]}\n'
        '  - {event_id: D, split: dev, probe_queries: ["dq 2023"], h2_probes: ["dh 2023"]}\n',
        encoding="utf-8")
    sg.clear_cache()
    assert sg.is_test_probe_query("th 2023", path=script)
    assert sg.is_test_probe_query("tq 2023", path=script)
    assert not sg.is_test_probe_query("dh 2023", path=script)
    sg.clear_cache()


def test_metadata_accessor_strips_h2_probes():
    sg.clear_cache()
    for ev in sg.load_event_metadata():
        assert "h2_probes" not in ev and "probe_queries" not in ev


def test_build_digest_list_is_text_free_and_includes_overlay():
    pub = {"events": [{"split": "test", "probe_queries": ["p1"], "h2_probes": ["h1"]}, {"split": "dev", "probe_queries": ["pd"]}]}
    sealed = {"events": [{"split": "test", "probe_queries": ["s1"], "h2_probes": ["s2"]}], "h2_overlay": {"X": ["o1"]}}
    lst = sg.build_probe_digest_list(pub, sealed)
    assert lst == sorted(sg.digest_of(t) for t in ["p1", "h1", "s1", "s2", "o1"])
    assert all(len(d) == 64 for d in lst)


def test_sealed_loader_requires_a_reason_and_tolerates_absence(tmp_path):
    with pytest.raises(ValueError):
        sg.load_sealed_source(reason=" ")
    assert sg.load_sealed_source(reason="t", path=tmp_path / "missing.yaml") == ("", {})


# ------------------------------------------------------------- build_suite


def _suite_events():
    ce = {"relevance": "relevant"}
    return [
        {"event_id": "A", "event_type": "normal", "split": "test", "batch": "v1", "corpus_evidence": ce,
         "business_description": "d", "probe_queries": ["qa1", "qa2"]},
        {"event_id": "B", "event_type": "normal", "split": "dev", "batch": "v1", "corpus_evidence": ce,
         "business_description": "d", "probe_queries": ["qb1"]},
        {"event_id": "EVT-V2-001", "event_type": "normal", "split": "dev", "batch": "v2", "corpus_evidence": ce,
         "business_description": "d", "probe_queries": ["qv1 2023"], "h2_probes": ["h 2023"]},
        {"event_id": "EVT-V2-002", "event_type": "normal", "split": "test", "batch": "v2", "corpus_evidence": ce,
         "business_description": "d", "probe_queries": ["leak 2023"]},
    ]


def test_v2_never_reaches_the_retrieval_test_suite():
    chunks = {"A": ["c1"], "B": ["c2"], "EVT-V2-001": ["c3"], "EVT-V2-002": ["c4"]}
    suites, missing = suite.build_suites(_suite_events(), chunks)
    assert [e["query_id"] for e in suites["test"]] == ["A#1", "A#2"]
    assert [e["query_id"] for e in suites["dev"]] == ["B#1", "EVT-V2-001#1"]
    assert missing == []
    assert all("h 2023" != e["query"] for s in suites.values() for e in s)


def test_v1_only_flag_reproduces_the_v1_suites():
    chunks = {"A": ["c1"], "B": ["c2"]}
    suites, missing = suite.build_suites(_suite_events(), chunks, v1_only=True)
    assert [e["query_id"] for e in suites["dev"]] == ["B#1"] and missing == []


def test_uningested_v2_dev_event_is_reported_not_skipped():
    _, missing = suite.build_suites(_suite_events(), {"A": ["c1"], "B": ["c2"]})
    assert missing == ["EVT-V2-001"]


def test_committed_test_json_is_unchanged():
    p = _ROOT / "eval" / "fixtures" / "tb5_retrieval" / "test.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest().startswith("78885927")


# ---------------------------------------------------------------- renderer


def test_assemble_events_modes():
    pub = [{"event_id": "V1A", "batch": "v1"}, {"event_id": "EVT-V2-004", "batch": "v2"}]
    sealed = [{"event_id": "EVT-V2-002", "batch": "v2"}]
    ids = lambda m: [e["event_id"] for e in render.assemble_events(m, pub, sealed)]
    assert ids("baseline") == []
    assert ids("v1-only") == ["V1A"]
    assert ids("public") == ["V1A", "EVT-V2-004"]
    assert ids("v2") == ["V1A", "EVT-V2-002", "EVT-V2-004"]
    with pytest.raises(SystemExit):
        render.assemble_events("v2", pub, [])


def test_renderer_refuses_to_write_into_other_repositories():
    for root in render.FORBIDDEN_OUT_ROOTS:
        with pytest.raises(SystemExit):
            render.refuse_forbidden_out_dir(root / "data" / "demo")
    render.refuse_forbidden_out_dir(Path("/tmp/somewhere/else"))


# ------------------------------------------------------------ whole-script


def test_verify_public_only_runs_on_this_checkout():
    _require_v1_tag()
    res = subprocess.run(
        [sys.executable, "-B", str(_ES / "verify.py"), "--public-only"],
        capture_output=True, text=True, cwd=_ROOT,
    )
    assert res.returncode == 0, res.stderr[-2000:]


def test_verify_fails_on_a_tampered_v1_event():
    _require_v1_tag()
    # run verify's checks directly on a mutated copy of the v1 list
    pub = sg.load_event_script_document(reason="test: tamper check")["events"]
    v1 = [dict(e) for e in pub if e.get("batch") == "v1"]
    v1[0]["business_description"] = "changed"
    errs: list[str] = []
    v2.check_v1_immutable(v1, sg.load_v1_tag_events(), errs)
    assert errs


# ------------------------------------------------- clean base table (附录 GP)


def _write_csv(path, rows):
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "region", "revenue", "units"])
        w.writeheader()
        w.writerows(rows)


def test_clean_consistency_accepts_blank_x10_and_duplicates(tmp_path):
    clean = [
        {"date": "2023-01-01", "region": "N", "revenue": "10.5", "units": "1"},
        {"date": "2023-01-02", "region": "S", "revenue": "20.0", "units": "2"},
        {"date": "2023-01-03", "region": "E", "revenue": "30.0", "units": "3"},
    ]
    dirty = [dict(r) for r in clean]
    dirty[0]["revenue"] = ""            # 3% blanking
    dirty[1]["revenue"] = "200.0"       # 1% x10
    dirty.append(dict(clean[2], revenue="300.0"))  # duplicate row that was also x10'd
    _write_csv(tmp_path / "c.csv", clean)
    _write_csv(tmp_path / "d.csv", dirty)
    out = render.check_clean_consistency(tmp_path / "c.csv", tmp_path / "d.csv")
    assert out["consistent"] and out["head_blank_restored"] == 1 and out["head_x10_restored"] == 1
    assert out["appended_are_duplicates_of_clean_rows"] == 1 and out["clean_blank_revenue_cells"] == 0


def test_clean_consistency_flags_anything_else(tmp_path):
    clean = [{"date": "2023-01-01", "region": "N", "revenue": "10.0", "units": "1"}]
    dirty = [dict(clean[0], revenue="11.0"), {"date": "2023-02-02", "region": "N", "revenue": "10.0", "units": "1"}]
    _write_csv(tmp_path / "c.csv", clean)
    _write_csv(tmp_path / "d.csv", dirty)
    out = render.check_clean_consistency(tmp_path / "c.csv", tmp_path / "d.csv")
    assert not out["consistent"] and out["head_other_revenue_mismatches"] == 1 and out["appended_not_matched"] == 1


def test_pinned_renderer_hash_is_enforced(tmp_path):
    repo = render.DEFAULT_MAEDA_REPO
    if not (repo / ".git").exists():
        pytest.skip("MAEDA repository not available")
    with pytest.raises(SystemExit):
        render.materialize_pinned_renderer(repo, render.PINNED_RENDERER_REF, "0" * 64, tmp_path / "x")
    d, commit = render.materialize_pinned_renderer(
        repo, render.PINNED_RENDERER_REF, render.PINNED_RENDERER_SHA256, tmp_path / "y"
    )
    assert (d / "generate_demo_data.py").is_file() and commit.startswith(render.PINNED_RENDERER_REF)


def test_v1_test_events_need_no_overlay_when_the_sealed_file_is_absent():
    ev = [{"event_id": "A", "split": "test", "valid_from": "2023-01-01", "valid_to": "2023-01-09"}]
    errs: list[str] = []
    v2.check_h2_for_v1(ev, {}, errs, sealed_present=False)
    assert errs == []
    v2.check_h2_for_v1(ev, {}, errs, sealed_present=True)
    assert errs
