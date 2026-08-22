"""Tests for rag.cli.eval's test-split reveal guard — 附录 CI.4
(ECOSYSTEM_INTEGRATION_PLAN.md, rag-side counterpart of MAEDA's 附录 CC.1
``scripts/measure_noise.py --split`` fix).

Before ``filter_by_split``/``warn_if_test_split_involved`` existed,
``eval/fixtures/tb5_retrieval/test.json`` was a plain, unprotected JSON
file — ``rag.cli.eval --suite-file .../test.json`` could be run with zero
warning and zero refusal, the only "protection" being a docstring sentence
in ``build_suite.py`` that no code path ever checked (附录 CH.6). These
tests lock the fix the same way MAEDA's
``tests/unit/test_measure_noise_script.py`` locks ``--split``'s default:
inspecting the parser's own source, not trusting a comment.

The guard is content-based (each entry's own ``"split"`` field, written by
``build_suite.py``), not filename-based, so it is checked here via
``--suite-file`` pointed at an on-disk file whose *content* happens to
carry ``"split": "test"`` — this is deliberately the same bypass shape as
MAEDA's ``test_warns_even_for_explicit_cases_that_happen_to_include_test``
(an explicit file/case selection that happens to include test-split data
must still be flagged, not just the ``--split test`` selector path).

No real pipeline/LLM/embedding calls anywhere in this file.
"""

from __future__ import annotations

import inspect
import json

from rag.cli.eval import (
    _build_parser,
    filter_by_split,
    main,
    warn_if_test_split_involved,
)


# ---------------------------------------------------------------------------
# --split default (the actual fix)
# ---------------------------------------------------------------------------


def test_split_argument_defaults_to_dev():
    """Locks the CLI default by inspecting the parser's own source — same
    technique as MAEDA's test_split_argument_defaults_to_dev — so this
    fails loudly if the default is ever changed back to unfiltered/"all"
    without touching this test too."""
    src = inspect.getsource(_build_parser)
    assert '"--split",' in src
    assert 'choices=["dev", "test", "all"]' in src
    assert 'default="dev"' in src


# ---------------------------------------------------------------------------
# filter_by_split
# ---------------------------------------------------------------------------


def _entries():
    return [
        {"query_id": "D1", "query": "dev query 1", "split": "dev"},
        {"query_id": "D2", "query": "dev query 2", "split": "dev"},
        {"query_id": "T1", "query": "test query 1", "split": "test"},
        {"query_id": "T2", "query": "test query 2", "split": "test"},
    ]


def test_default_split_excludes_test_entries():
    """The exact scenario CI.4 found dangerous: a default invocation must
    not touch test-split entries."""
    selected = filter_by_split(_entries(), "dev")
    assert {e["query_id"] for e in selected} == {"D1", "D2"}


def test_split_test_is_opt_in_only():
    selected = filter_by_split(_entries(), "test")
    assert {e["query_id"] for e in selected} == {"T1", "T2"}


def test_split_all_returns_everything_but_is_never_the_default():
    selected = filter_by_split(_entries(), "all")
    assert {e["query_id"] for e in selected} == {"D1", "D2", "T1", "T2"}


def test_entries_without_split_field_pass_through_as_non_test():
    """example_queries.json / failure_cases.json (and any other non-TB5
    suite) have no 'split' field at all — the default filter must not
    silently eat them."""
    entries = [{"query_id": "X1", "query": "unrelated suite entry"}]
    selected = filter_by_split(entries, "dev")
    assert {e["query_id"] for e in selected} == {"X1"}
    # ...but they never count as "test" either, even under --split test.
    assert filter_by_split(entries, "test") == []


# ---------------------------------------------------------------------------
# warn_if_test_split_involved (附录 CI.4: "明确的、读得懂的警告")
# ---------------------------------------------------------------------------


def test_warns_when_test_split_selected(capsys):
    """附录 CZ: the banner moved from stdout to stderr when the guard moved
    into the shared ``rag/`` package (``rag.eval.split_guard._emit_banner``).
    A human at a terminal sees stderr exactly as they saw stdout, so 硬约束 1
    is unaffected; the MCP stdio JSON-RPC stream is now untouched by
    construction. Asserting ``out == ""`` here makes stdout-cleanliness a
    *behavioural* property of the guard, not only an AST rule in
    tests/test_mcp_stdio_safety.py."""
    warn_if_test_split_involved([{"query_id": "T1", "split": "test"}])
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "TEST-SPLIT" in captured.err
    assert "T1" in captured.err
    assert captured.out == ""


def test_no_warning_when_only_dev_selected(capsys):
    warn_if_test_split_involved([{"query_id": "D1", "split": "dev"}])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_no_warning_for_entries_without_split_field(capsys):
    warn_if_test_split_involved([{"query_id": "X1"}])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# main() end-to-end — bypass coverage (附录 CI.4's explicit requirement)
# ---------------------------------------------------------------------------


def test_main_suite_file_pointed_straight_at_test_split_is_excluded_by_default(
    tmp_path, capsys
):
    """附录 CI.4's named bypass: '绕过 split 参数、直接指定文件...' — a
    --suite-file argument pointed directly at a file that IS the frozen
    test split (content-wise, regardless of what it's named on disk) must
    still be excluded by the default --split dev, not silently run."""
    suite_file = tmp_path / "not_named_test_at_all.json"
    suite_file.write_text(
        json.dumps(
            [
                {"query_id": "T1", "query": "q1", "split": "test", "expected_sources": ["c1"]},
                {"query_id": "T2", "query": "q2", "split": "test", "expected_sources": ["c2"]},
            ]
        )
    )

    rc = main(["--suite", "example_queries", "--suite-file", str(suite_file)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "test-split" in captured.out
    assert "nothing to evaluate" in captured.out
    # Must not print the "empty suite" message meant for a genuinely empty
    # file — this is a *filtered-to-empty* suite, a materially different
    # situation the operator needs to be told apart from "the file is empty".
    assert "suite is empty" not in captured.out


def test_main_explicit_split_test_warns_and_does_not_silently_run(tmp_path, capsys):
    """Bypass coverage's other half: even the explicit opt-in path prints
    the warning banner naming the test ids, per CI.4/硬约束 1 — this is
    the last point before a real run where a human is still watching."""
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps(
            [{"query_id": "T1", "query": "q1", "split": "test", "expected_sources": ["c1"]}]
        )
    )
    db_path = tmp_path / "test.db"
    db_path.touch()

    from unittest.mock import patch

    mock_results = [
        {
            "query_id": "T1",
            "query": "q1",
            "retrieved": ["c1"],
            "relevant": ["c1"],
            "candidates": [],
            "expected_behavior": "answer",
            "query_latency_ms": 1.0,
            "error": None,
        }
    ]
    with patch("rag.cli.eval._run_queries", return_value=mock_results):
        rc = main([
            "--suite", "example_queries",
            "--suite-file", str(suite_file),
            "--db", str(db_path),
            "--split", "test",
        ])

    assert rc == 0
    captured = capsys.readouterr()
    # 附录 CZ: banner on stderr, results on stdout. Both land on the same
    # terminal for the human this warning is for; only the JSON-RPC stream
    # of an MCP process can tell them apart, which is the point.
    assert "WARNING" in captured.err
    assert "TEST-SPLIT" in captured.err
    assert "T1" in captured.err
    assert "WARNING" not in captured.out


def test_main_default_split_on_mixed_suite_only_evaluates_dev(tmp_path, capsys):
    """A suite containing both dev- and test-marked entries (not today's
    actual dev.json/test.json layout, but the filter must be correct for
    it) only evaluates the dev-marked ones under the default."""
    suite_file = tmp_path / "mixed.json"
    suite_file.write_text(
        json.dumps(
            [
                {"query_id": "D1", "query": "dev q", "split": "dev", "expected_sources": ["c1"]},
                {"query_id": "T1", "query": "test q", "split": "test", "expected_sources": ["c2"]},
            ]
        )
    )
    db_path = tmp_path / "test.db"
    db_path.touch()

    from unittest.mock import patch

    def fake_run_queries(entries, *args, **kwargs):
        # Assert the filter already happened before _run_queries is called.
        assert {e["query_id"] for e in entries} == {"D1"}
        return [
            {
                "query_id": "D1",
                "query": "dev q",
                "retrieved": ["c1"],
                "relevant": ["c1"],
                "candidates": [],
                "expected_behavior": "answer",
                "query_latency_ms": 1.0,
                "error": None,
            }
        ]

    with patch("rag.cli.eval._run_queries", side_effect=fake_run_queries):
        rc = main([
            "--suite", "example_queries",
            "--suite-file", str(suite_file),
            "--db", str(db_path),
        ])

    assert rc == 0
    captured = capsys.readouterr()
    assert "T1" not in captured.out.split("WARNING")[0]  # no warning banner printed
    assert "WARNING" not in captured.out
