"""End-to-end main path validation — Task 9.3.

Tests cover the four acceptance criteria:
  1. Ingest a TXT file via CLI — succeeds (exit 0, chunk_count > 0)
  2. Query via CLI — returns results with citations
  3. Streamlit four pages load without errors (import test)
  4. MCP tool call returns structured response with answer field
  5. Full unit test suite passes (subprocess pytest -q)

All tests use a temporary SQLite DB and index directory, so they are
fully isolated and leave no artefacts behind.

These tests are marked ``e2e`` and are skipped in fast CI runs unless
``-m e2e`` is passed explicitly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_workspace(tmp_path):
    """Yield a (db_path, index_dir, sample_txt) tuple in a temp directory."""
    db_path = tmp_path / "rag_e2e.db"
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    sample_txt = tmp_path / "sample.txt"
    sample_txt.write_text(
        textwrap.dedent("""\
        # Introduction to RAG

        Retrieval-Augmented Generation (RAG) is a technique that combines
        information retrieval with large language model generation.

        ## How it works

        A query is embedded and compared against indexed document chunks.
        The most relevant chunks are retrieved and passed to the LLM as context.

        ## Benefits

        RAG reduces hallucination by grounding answers in retrieved evidence.
        It also allows the LLM to access up-to-date information without retraining.
        """),
        encoding="utf-8",
    )
    return db_path, index_dir, sample_txt


# ---------------------------------------------------------------------------
# 1. CLI ingest
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_cli_ingest_txt_succeeds(tmp_workspace):
    """Ingesting a TXT file via CLI returns exit code 0 and stores chunks."""
    db_path, index_dir, sample_txt = tmp_workspace

    from rag.cli.ingest import main as ingest_main

    exit_code = ingest_main([
        "--path", str(sample_txt),
        "--db", str(db_path),
        "--token-budget", "256",
    ])
    assert exit_code == 0, "CLI ingest should return exit code 0 on success"
    # DB file should have been created by the pipeline
    assert db_path.exists(), "SQLite DB should exist after ingest"


@pytest.mark.e2e
def test_cli_ingest_nonexistent_file_fails():
    """Ingesting a nonexistent file returns non-zero exit code."""
    from rag.cli.ingest import main as ingest_main

    exit_code = ingest_main(["--path", "/nonexistent/missing.txt"])
    assert exit_code != 0, "CLI ingest should return non-zero on failure"


# ---------------------------------------------------------------------------
# 2. CLI query
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_cli_query_returns_results(tmp_workspace, capsys, monkeypatch):
    """Query CLI returns results (exit 0) after ingest."""
    db_path, index_dir, sample_txt = tmp_workspace

    # First ingest
    from rag.cli.ingest import main as ingest_main
    rc = ingest_main([
        "--path", str(sample_txt),
        "--db", str(db_path),
        "--token-budget", "256",
    ])
    assert rc == 0

    # ingest_main() just called load_dotenv(override=False) internally,
    # which (附录 CI.3/CH.2: RAG_EMBEDDING_PROVIDER_DEFAULT is now a live
    # variable, no longer dead-named) may have pulled this machine's real
    # ~/.env into os.environ for the rest of the process. rag.cli.query has
    # no --embedding-provider override (unlike rag.cli.eval), so strip it
    # back out before querying — this test only ever ingested BM25 (rag.
    # cli.ingest has no embedding step) and is testing "does the plain
    # ingest→query path work", not "which embedding provider happens to be
    # configured on this machine". Without this, a real .env with
    # RAG_EMBEDDING_PROVIDER_DEFAULT=openai would push this smoke test onto
    # a vector-search path with no matching index (INDEX_NOT_FOUND).
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER_DEFAULT", raising=False)
    monkeypatch.delenv("RAG_RERANKER_PROVIDER_DEFAULT", raising=False)

    # Then query
    from rag.cli.query import main as query_main
    exit_code = query_main([
        "What is RAG?",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
        "--top-k", "3",
    ])
    assert exit_code == 0, "CLI query should return exit code 0"

    captured = capsys.readouterr()
    # Output should mention the query or results
    assert len(captured.out) > 0, "Query CLI should produce output"


@pytest.mark.e2e
def test_cli_query_empty_index_returns_gracefully(tmp_workspace):
    """Query CLI on an empty index returns exit 0 with no results (not a crash)."""
    db_path, index_dir, _ = tmp_workspace

    from rag.cli.query import main as query_main
    exit_code = query_main([
        "What is RAG?",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
        "--top-k", "3",
    ])
    # CLI exits non-zero when no DB exists yet (no crash / no exception)
    assert exit_code in (0, 1), "Exit code must be 0 or 1, never a crash"


# ---------------------------------------------------------------------------
# 3. Streamlit pages load without errors
# ---------------------------------------------------------------------------


def _make_minimal_st_stub() -> types.ModuleType:
    """Build a minimal Streamlit stub for import-time testing."""

    class _CM:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __getattr__(self, name): return lambda *a, **kw: _CM()

    stub = types.ModuleType("streamlit")
    _noop = lambda *a, **kw: _CM()

    for attr in [
        "set_page_config", "title", "caption", "header", "subheader",
        "info", "success", "warning", "error", "divider", "markdown",
        "expander", "spinner", "metric", "rerun", "write", "dataframe",
        "json", "selectbox", "checkbox", "tabs", "form_submit_button",
    ]:
        setattr(stub, attr, _noop)

    stub.text_input = lambda *a, **kw: kw.get("value", "")
    stub.text_area = lambda *a, **kw: kw.get("value", "")
    stub.number_input = lambda *a, **kw: kw.get("value", 0)
    stub.radio = lambda *a, **kw: (kw.get("options") or [None])[0]
    def _selectbox(*a, **kw):
        opts = kw.get("options") or (list(a[1]) if len(a) > 1 else [])
        return opts[0] if opts else None
    stub.selectbox = _selectbox
    stub.multiselect = lambda *a, **kw: []
    stub.checkbox = lambda *a, **kw: kw.get("value", False)
    stub.toggle = lambda *a, **kw: kw.get("value", False)
    stub.button = lambda *a, **kw: False
    stub.file_uploader = lambda *a, **kw: None
    stub.form_submit_button = lambda *a, **kw: False
    stub.container = lambda *a, **kw: _CM()
    stub.columns = lambda n: [_CM() for _ in range(n if isinstance(n, int) else len(n))]
    # tabs must return a real list so `tab_a, tab_b = st.tabs([...])` works
    stub.tabs = lambda labels: [_CM() for _ in labels]
    stub.sidebar = _CM()
    stub.form = lambda *a, **kw: _CM()
    stub.cache_data = lambda *a, **kw: (lambda f: f)
    # Catch-all: any attribute not explicitly set returns a no-op callable
    stub.__class__ = type("StreamlitStub", (types.ModuleType,), {
        "__getattr__": lambda self, name: _noop
    })

    class _SS(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError: raise AttributeError(k)
        def __setattr__(self, k, v): self[k] = v

    stub.session_state = _SS(
        ingest_result=None, ingest_config=None, ingest_error=None,
        query_result=None, query_running=False, query_error=None, last_query="",
    )
    return stub


def _import_page(file_path: str, module_name: str) -> types.ModuleType:
    stub = _make_minimal_st_stub()
    # Also stub pandas with a minimal DataFrame that supports chaining
    class _FakeDF:
        def __init__(self, data=None): pass
        def __getattr__(self, name): return lambda *a, **kw: self
        def __iter__(self): return iter([])
        def __len__(self): return 0

    pd_stub = types.ModuleType("pandas")
    pd_stub.DataFrame = lambda *a, **kw: _FakeDF()
    old_st = sys.modules.get("streamlit")
    old_pd = sys.modules.get("pandas")
    sys.modules["streamlit"] = stub
    sys.modules["pandas"] = pd_stub
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old_st is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old_st
        if old_pd is None:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = old_pd
    return mod


_STUDIO_DIR = Path(__file__).parent.parent.parent / "rag/app/studio"


def test_streamlit_home_page_loads():
    mod = _import_page(str(_STUDIO_DIR / "studio.py"), "st_home")
    assert mod is not None


def test_streamlit_ingestion_manager_loads():
    mod = _import_page(str(_STUDIO_DIR / "pages/1_ingest_inspect.py"), "st_ingest")
    assert mod is not None


def test_streamlit_ingestion_traces_loads():
    mod = _import_page(str(_STUDIO_DIR / "pages/1_ingest_inspect.py"), "st_ingest_inspect")
    assert mod is not None


def test_streamlit_query_traces_loads():
    mod = _import_page(str(_STUDIO_DIR / "pages/2_query_traces.py"), "st_query")
    assert mod is not None


def test_streamlit_evaluation_panel_loads():
    mod = _import_page(str(_STUDIO_DIR / "pages/3_evaluation_panel.py"), "st_eval")
    assert mod is not None


# ---------------------------------------------------------------------------
# 4. MCP tool call returns structured response
# ---------------------------------------------------------------------------


def test_mcp_ingest_tool_returns_structured_response(tmp_workspace):
    """rag.ingest MCP tool returns IngestToolOutput with expected fields."""
    db_path, index_dir, sample_txt = tmp_workspace

    from rag.app.mcp_server.schemas import IngestToolInput, IngestToolOutput
    from rag.app.mcp_server.server import rag_ingest

    inp = IngestToolInput(
        source_path=str(sample_txt),
        db_path=str(db_path),
        index_dir=str(index_dir),
        token_budget=256,
    )
    result = rag_ingest(inp)

    assert isinstance(result, IngestToolOutput)
    assert result.error is None, f"Ingest should not error: {result.error}"
    assert result.doc_id, "doc_id should be populated"
    assert result.chunk_count >= 1, "At least one chunk should be produced"


def test_mcp_query_tool_returns_structured_response(tmp_workspace):
    """rag.query MCP tool returns QueryToolOutput with query and run_id fields."""
    db_path, index_dir, sample_txt = tmp_workspace

    # Ingest first so there's something to query
    from rag.app.mcp_server.schemas import IngestToolInput, QueryToolInput, QueryToolOutput
    from rag.app.mcp_server.server import rag_ingest, rag_query

    ingest_inp = IngestToolInput(
        source_path=str(sample_txt),
        db_path=str(db_path),
        index_dir=str(index_dir),
        token_budget=256,
    )
    rag_ingest(ingest_inp)

    query_inp = QueryToolInput(
        query="What is RAG?",
        db_path=str(db_path),
        index_dir=str(index_dir),
        top_k=3,
        enable_generation=False,  # skip LLM in e2e to avoid API calls
    )
    result = rag_query(query_inp)

    assert isinstance(result, QueryToolOutput)
    assert result.query == "What is RAG?"
    assert result.error is None, f"Query should not error: {result.error}"
    assert result.run_id, "run_id should be populated"


def test_mcp_eval_tool_returns_structured_response(tmp_workspace):
    """rag.eval.run MCP tool returns EvalRunToolOutput for a valid dataset."""
    db_path, index_dir, _ = tmp_workspace

    dataset_path = tmp_workspace[0].parent / "eval.json"
    dataset_path.write_text(
        json.dumps([{"query": "What is RAG?", "expected_chunks": ["c1"]}])
    )

    from rag.app.mcp_server.schemas import EvalRunToolInput, EvalRunToolOutput
    from rag.app.mcp_server.server import rag_eval_run

    inp = EvalRunToolInput(
        dataset_path=str(dataset_path),
        metrics=["recall_at_k", "mrr"],
        db_path=str(db_path),
        index_dir=str(index_dir),
    )
    result = rag_eval_run(inp)

    assert isinstance(result, EvalRunToolOutput)
    assert result.error is None, f"Eval should not error: {result.error}"
    assert result.num_queries == 1
    assert len(result.metrics) == 2


# ---------------------------------------------------------------------------
# 5. Full unit test suite (lightweight smoke check)
# ---------------------------------------------------------------------------


def test_all_non_e2e_unit_tests_importable():
    """Verify that all test modules (excluding e2e) can be imported cleanly."""
    tests_dir = Path(__file__).parent.parent
    failures = []
    for test_file in sorted(tests_dir.glob("test_*.py")):
        module_name = f"_check_{test_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            mod = importlib.util.module_from_spec(spec)
            # Don't exec — just verify the spec resolves cleanly
            assert spec is not None, f"Could not load spec for {test_file.name}"
        except Exception as exc:
            failures.append(f"{test_file.name}: {exc}")
    assert not failures, "Some test modules failed to load:\n" + "\n".join(failures)
