"""The test-split reveal guard as a *mechanism* — 附录 CP.4 / 附录 CZ.

附录 CI.4 built the first version of this guard and wired it into
``rag.cli.eval``. 附录 CP.4 then showed that a repo-wide grep found
``filter_by_split`` called from exactly one place — that file — while the
probe scripts of 附录 CJ.3 and 附录 CO read the frozen event script directly
with ``yaml.safe_load`` and never came near it. 附录 CO.0 is the recorded
test-split observation that walked through that hole.

``tests/test_eval_cli_split.py`` covers the CI.4 half: the CLI's ``--split``
default and its ``--suite-file`` bypass. **This file covers the half that
matters for CP.4: the paths that never touch ``rag.cli.eval`` at all.** Every
test here reaches a test-split probe query the way a hand-rolled probe script
would — raw YAML, a copied and renamed fixture, the derived ``test.json``, a
re-wrapped string — and asserts that it still fails closed at execution.

What this file deliberately does *not* do:

* It never asserts on, logs, or parametrises the probe query text itself.
  Recognition is by digest, and a test that printed the string into a pytest
  report to prove the string is protected would be self-defeating.
* It does not run retrieval. The pipeline's collaborators are mocks; the point
  is *where the refusal happens*, not what retrieval would have returned.

No real pipeline/LLM/embedding calls anywhere in this file.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from rag.eval import split_guard
from rag.eval.split_guard import (
    EVENT_SCRIPT_PATH,
    REVEAL_ENV_VAR,
    assert_not_test_probe_query,
    declare_test_split_reveal,
    is_test_probe_query,
    load_event_script,
    load_event_metadata,
)

# Aliased on import: pytest's collection heuristics would otherwise try to
# collect ``TestSplitRevealError`` as a test class and
# ``test_probe_query_digests`` as a test function purely because of how they
# are spelled. The library names are right for the library (it guards the
# *test split*); the aliases are local noise suppression, not disagreement.
from rag.eval.split_guard import TestSplitRevealError as RevealError  # noqa: E402
from rag.eval.split_guard import test_probe_query_digests as _probe_digests  # noqa: E402
from rag.pipelines.query_pipeline import QueryPipeline

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TB5_DIR = _REPO_ROOT / "eval" / "fixtures" / "tb5_retrieval"


# ---------------------------------------------------------------------------
# Helpers — each one reproduces a *bypass* shape, not the sanctioned path
# ---------------------------------------------------------------------------


def _test_probe_query_via_raw_yaml(path: Path | None = None) -> str:
    """Get a test-split probe query exactly the way 附录 CJ.3/CO did.

    ``yaml.safe_load`` over the frozen file, hand-rolled split check, no
    import from ``rag.eval``. If the guard only worked for callers who
    cooperate, this function is what would defeat it.
    """
    data = yaml.safe_load((path or EVENT_SCRIPT_PATH).read_text(encoding="utf-8"))
    for event in data["events"]:
        if event.get("split") == "test":
            for query in event.get("probe_queries") or []:
                if isinstance(query, str) and query.strip():
                    return query
    pytest.skip("frozen event script holds no test-split probe query")


def _dev_probe_query() -> str:
    for event in load_event_script("dev"):
        for query in event.get("probe_queries") or []:
            if isinstance(query, str) and query.strip():
                return query
    pytest.skip("frozen event script holds no dev-split probe query")


def _mock_pipeline() -> tuple[QueryPipeline, Mock, Mock]:
    keyword_index, trace_store = Mock(), Mock()
    keyword_index.search.return_value = []
    trace_store.save_run.return_value = "run-1"
    return (
        QueryPipeline(keyword_index=keyword_index, trace_store=trace_store),
        keyword_index,
        trace_store,
    )


@pytest.fixture(autouse=True)
def _no_ambient_reveal(monkeypatch):
    """Fail closed regardless of the developer's shell.

    ``REVEAL_ENV_VAR`` in the ambient environment (or in a ``.env`` that some
    entry point loaded) would otherwise turn every assertion in this file into
    a silent pass — the exact failure mode the guard exists to prevent.
    """
    monkeypatch.delenv(REVEAL_ENV_VAR, raising=False)
    assert split_guard.reveal_declared() is None


# ---------------------------------------------------------------------------
# The CP.4 requirement: paths that bypass rag.cli.eval are blocked too
# ---------------------------------------------------------------------------


class TestBypassPathsFailClosed:
    def test_raw_yaml_read_then_query_is_refused_at_execution(self):
        """The 附录 CO.0 shape, end to end: read the frozen YAML directly,
        hand the string to the pipeline, get refused.

        This is the single load-bearing test of 附录 CP.4. It uses the real
        frozen fixture (not a synthetic one) because the claim being locked is
        that *the actual test split* is protected, not that the mechanism is
        capable of protecting something.
        """
        query = _test_probe_query_via_raw_yaml()
        pipeline, keyword_index, trace_store = _mock_pipeline()

        with pytest.raises(RevealError):
            pipeline.query(query)

        # Refused *before* anything happened, not after the fact: no trace row
        # (附录 CO.0's evidence was 7 stray trace rows) and no index touched.
        trace_store.save_run.assert_not_called()
        keyword_index.search.assert_not_called()

    def test_copied_and_renamed_fixture_is_recognised_by_content(self, tmp_path):
        """附录 CK.6's criterion: the check keys off entry content, not the
        file's name or location. Copy the frozen script somewhere innocuous,
        read the copy, and the query is still refused."""
        decoy = tmp_path / "notes" / "scratch_queries.yaml"
        decoy.parent.mkdir()
        decoy.write_bytes(EVENT_SCRIPT_PATH.read_bytes())

        query = _test_probe_query_via_raw_yaml(decoy)
        pipeline, _, _ = _mock_pipeline()

        with pytest.raises(RevealError):
            pipeline.query(query)

    def test_every_derived_test_json_query_is_recognised(self):
        """``test.json`` is a *derived* artefact — a second door onto the same
        strings. Every one of its queries must be recognised, or the guard is
        only protecting the YAML."""
        entries = json.loads((_TB5_DIR / "test.json").read_text(encoding="utf-8"))
        assert entries, "test.json is empty — this test would vacuously pass"
        unrecognised = [
            e.get("query_id", "?") for e in entries if not is_test_probe_query(e.get("query", ""))
        ]
        assert unrecognised == []

    def test_no_dev_split_query_is_falsely_recognised(self):
        """The other direction: a guard that blocked dev queries too would be
        abandoned within a day, and abandoning it is the real failure mode."""
        entries = json.loads((_TB5_DIR / "dev.json").read_text(encoding="utf-8"))
        assert entries
        false_positives = [
            e.get("query_id", "?") for e in entries if is_test_probe_query(e.get("query", ""))
        ]
        assert false_positives == []

    def test_dev_query_runs_normally_through_the_same_funnel(self):
        """Positive control for the test above, at the pipeline rather than
        the predicate: a dev-split probe query passes the guard and reaches
        the trace store."""
        pipeline, keyword_index, trace_store = _mock_pipeline()
        pipeline.query(_dev_probe_query())
        assert trace_store.save_run.called
        assert keyword_index.search.called

    def test_rewrapped_query_text_is_still_refused(self):
        """Copy-paste damage — re-wrapped lines, collapsed indentation, a
        changed case — must not launder a test query past the guard. This is
        forgiveness, not adversary-resistance: see ``normalize_query``."""
        query = _test_probe_query_via_raw_yaml()
        mangled = "  ".join(f"\n\t{part.upper()} " for part in query.split(" "))
        pipeline, _, _ = _mock_pipeline()

        with pytest.raises(RevealError):
            pipeline.query(mangled)

    def test_cli_eval_run_queries_refuses_before_constructing_a_pipeline(self, monkeypatch):
        """The cheap, early half of the check. ``rag.cli.eval._run_queries``
        must refuse a test-split batch *before* it builds indexes or an
        embedding provider — a refusal that first spends money is not much of
        a refusal."""
        from rag.cli import eval as cli_eval

        built = Mock()
        monkeypatch.setattr("rag.app.mcp_server.wiring.build_query_pipeline", built)
        entries = [{"query_id": "T1", "query": _test_probe_query_via_raw_yaml()}]

        with pytest.raises(RevealError):
            cli_eval._run_queries(
                entries, Path("nonexistent.db"), Path("nonexistent_index"), top_k=5
            )

        built.assert_not_called()


# ---------------------------------------------------------------------------
# The declaration — the one door that opens, and what it costs to open it
# ---------------------------------------------------------------------------


class TestRevealDeclaration:
    def test_context_manager_opens_the_gate_only_inside_the_block(self):
        query = _test_probe_query_via_raw_yaml()
        pipeline, _, trace_store = _mock_pipeline()

        with declare_test_split_reveal("unit test: predetermined reveal"):
            pipeline.query(query)
        assert trace_store.save_run.called

        with pytest.raises(RevealError):
            pipeline.query(query)

    def test_declaration_is_restored_even_when_the_block_raises(self):
        query = _test_probe_query_via_raw_yaml()
        with pytest.raises(RuntimeError):
            with declare_test_split_reveal("unit test"):
                raise RuntimeError("boom")
        with pytest.raises(RevealError):
            assert_not_test_probe_query(query)

    def test_empty_reason_is_rejected(self):
        with pytest.raises(ValueError):
            with declare_test_split_reveal("   "):
                pass

    def test_env_var_opens_the_gate_for_an_out_of_process_driver(self, monkeypatch, capsys):
        """MAEDA's stage-4 harness drives this repository from outside, so the
        declaration has to be reachable without importing anything.

        附录 CZ: it must not be the *quiet* route. A ``.env`` line picked up by
        ``load_dotenv(override=False)`` would otherwise grant an ambient,
        invisible reveal permission for the whole process."""
        query = _test_probe_query_via_raw_yaml()
        monkeypatch.setenv(REVEAL_ENV_VAR, "1")
        split_guard.clear_cache()  # re-arm the once-per-process banner
        assert_not_test_probe_query(query)  # must not raise
        assert "REVEAL DECLARED" in capsys.readouterr().err

    def test_env_var_banner_is_emitted_once_not_per_query(self, monkeypatch, capsys):
        query = _test_probe_query_via_raw_yaml()
        monkeypatch.setenv(REVEAL_ENV_VAR, "1")
        split_guard.clear_cache()
        assert_not_test_probe_query(query)
        capsys.readouterr()
        assert_not_test_probe_query(query)
        assert capsys.readouterr().err == ""

    def test_env_var_set_to_zero_does_not_open_the_gate(self, monkeypatch):
        query = _test_probe_query_via_raw_yaml()
        monkeypatch.setenv(REVEAL_ENV_VAR, "0")
        with pytest.raises(RevealError):
            assert_not_test_probe_query(query)

    def test_reveal_banner_goes_to_stderr_not_stdout(self, capsys):
        with declare_test_split_reveal("unit test: banner stream"):
            pass
        captured = capsys.readouterr()
        assert "REVEAL DECLARED" in captured.err
        assert captured.out == ""


# ---------------------------------------------------------------------------
# What the refusal itself must not do
# ---------------------------------------------------------------------------


class TestRefusalDoesNotLeak:
    def test_error_message_never_contains_the_query_text(self):
        """Blocking a reveal by writing the protected string into a traceback,
        a CI log and a pytest report would spread it further than letting it
        run. The message carries digest prefixes only."""
        query = _test_probe_query_via_raw_yaml()
        with pytest.raises(RevealError) as exc_info:
            assert_not_test_probe_query(query, context="unit test")
        message = str(exc_info.value)
        assert query not in message
        for word in query.split():
            if len(word) > 4:
                assert word not in message
        assert "硬约束 1" in message
        assert REVEAL_ENV_VAR in message

    def test_metadata_accessor_exposes_splits_without_exposing_queries(self):
        """附录 CR.5: reading *which* events exist and which split each is in
        is not a reveal, and the guard could not build its digest set
        otherwise. ``load_event_metadata`` makes that the easy path."""
        events = load_event_metadata()
        assert events
        assert any(e.get("split") == "test" for e in events)
        for event in events:
            assert "probe_queries" not in event
            assert isinstance(event["probe_query_count"], int)


# ---------------------------------------------------------------------------
# Default-safe loading, and the structural rule that keeps it the only door
# ---------------------------------------------------------------------------


class TestDefaultSafeLoading:
    def test_load_event_script_defaults_to_dev_only(self):
        events = load_event_script()
        assert events
        assert {e.get("split") for e in events} == {"dev"}

    def test_reading_test_split_requires_naming_it(self):
        assert {e.get("split") for e in load_event_script("test")} == {"test"}
        assert {e.get("split") for e in load_event_script("all")} == {"dev", "test"}

    def test_invalid_split_selector_is_rejected(self):
        with pytest.raises(ValueError):
            load_event_script("prod")

    def test_digest_set_is_non_empty(self):
        """A guard whose digest set silently became empty — renamed fixture,
        moved repo root, changed field name — would pass every 'dev query is
        not blocked' test in this file while protecting nothing."""
        assert len(_probe_digests()) > 0


class TestNoSecondDirectReader:
    """The structural half of 附录 CP.4.

    Every test above proves the *execution* guard fires. None of them can stop
    a future script from re-inventing ``yaml.safe_load(event_script.yaml)`` and
    reading test-split text without declaring anything — reading is not
    blocked, by design (附录 CR.5). What this test does is make that act
    visible: exactly one module in this repository may name the frozen file,
    so a second reader shows up as a failing test rather than as a habit.
    """

    _ALLOWED = {
        Path("rag/eval/split_guard.py"),
        Path("tests/test_split_guard.py"),
    }

    def test_only_split_guard_names_the_frozen_event_script(self):
        offenders: dict[str, list[int]] = {}
        for py_file in _REPO_ROOT.rglob("*.py"):
            rel = py_file.relative_to(_REPO_ROOT)
            if "__pycache__" in rel.parts or ".venv" in rel.parts or rel in self._ALLOWED:
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            lines = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (
                    node.value == "event_script.yaml"
                    or node.value.endswith("/event_script.yaml")
                )
            ]
            if lines:
                offenders[str(rel)] = lines

        assert not offenders, (
            "These modules name the frozen event script directly instead of "
            "going through rag.eval.split_guard, which is how 附录 CO.0 "
            f"happened: {offenders}. If a new caller genuinely needs every "
            "split, call load_event_script_document(reason=...) — it takes a "
            "declaration and prints a banner. If it needs dev, call "
            "load_event_script()."
        )
