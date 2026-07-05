"""Tests for IndexManager — index startup loader (Task 4.7)."""

import tempfile
from pathlib import Path

import pytest

from rag.core.contracts.chunk import Chunk
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.indexes.bm25_local import BM25LocalIndex
from rag.infra.indexes.faiss_local import FaissLocalIndex
from rag.infra.indexes.index_manager import IndexManager
from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema as init_doc_schema
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore, init_schema as init_trace_schema
from rag.pipelines.ingest_pipeline import IngestPipeline

_DIM = 8


# ---------------------------------------------------------------------------
# Stub embedding provider
# ---------------------------------------------------------------------------


class StubEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def dim(self) -> int:
        return _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        return [[float(len(t) % _DIM == i) for i in range(_DIM)] for t in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stores(db_path: str):
    init_doc_schema(db_path)
    init_trace_schema(db_path)
    return SQLiteDocStore(db_path), SQLiteTraceStore(db_path)


def _make_txt(tmp_path: Path) -> str:
    p = tmp_path / "sample.txt"
    p.write_text(
        "First paragraph about machine learning.\n\n"
        "Second paragraph about neural networks.\n\n"
        "Third paragraph about transformers and attention.\n",
        encoding="utf-8",
    )
    return str(p)


# ---------------------------------------------------------------------------
# First-run (no files on disk)
# ---------------------------------------------------------------------------


def test_fresh_start_creates_empty_indexes(tmp_path):
    """IndexManager with no saved files should give working empty indexes."""
    manager = IndexManager(tmp_path / "indexes")
    assert isinstance(manager.bm25, BM25LocalIndex)
    assert isinstance(manager.faiss, FaissLocalIndex)
    # Empty indexes return no results
    assert manager.bm25.search("query", top_k=5) == []
    assert manager.faiss.search([0.0] * _DIM, top_k=5) == []


# ---------------------------------------------------------------------------
# Save → reload cycle (simulates process restart)
# ---------------------------------------------------------------------------


def test_save_and_reload_restores_bm25(tmp_path):
    """BM25 state saved after ingest must be present after simulated restart."""
    index_dir = str(tmp_path / "indexes")
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    txt = _make_txt(tmp_path)

    # --- Process 1: ingest ---
    manager1 = IndexManager(index_dir)
    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=manager1.bm25,
        vector_index=manager1.faiss,
    )
    result = pipeline.ingest(txt)
    assert result.error is None
    manager1.save()

    # --- Process 2: restart (fresh IndexManager pointing at same dir) ---
    manager2 = IndexManager(index_dir)
    candidates = manager2.bm25.search("machine learning", top_k=5)
    assert len(candidates) > 0


def test_save_and_reload_restores_faiss(tmp_path):
    """FAISS state saved after ingest must be present after simulated restart."""
    index_dir = str(tmp_path / "indexes")
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    txt = _make_txt(tmp_path)

    # --- Process 1: ingest ---
    manager1 = IndexManager(index_dir)
    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=manager1.bm25,
        vector_index=manager1.faiss,
    )
    pipeline.ingest(txt)
    manager1.save()

    # --- Process 2: restart ---
    manager2 = IndexManager(index_dir)
    query = [0.0] * _DIM
    candidates = manager2.faiss.search(query, top_k=5)
    assert len(candidates) > 0


# ---------------------------------------------------------------------------
# Reload method
# ---------------------------------------------------------------------------


def test_reload_picks_up_new_data(tmp_path):
    """reload() must replace in-memory state with updated on-disk indexes."""
    index_dir = str(tmp_path / "indexes")
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    txt = _make_txt(tmp_path)

    manager = IndexManager(index_dir)

    # Ingest via separate index references, save
    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=manager.bm25,
        vector_index=manager.faiss,
    )
    pipeline.ingest(txt)
    manager.save()

    # Simulate external update: reload from disk
    manager.reload()
    assert manager.bm25.search("transformers", top_k=5)


# ---------------------------------------------------------------------------
# Corrupt / partial file handling
# ---------------------------------------------------------------------------


def test_corrupted_bm25_file_starts_empty(tmp_path):
    """A corrupt bm25.pkl must not crash — IndexManager falls back to empty."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "bm25.pkl").write_bytes(b"not valid pickle data")

    manager = IndexManager(str(index_dir))  # must not raise
    assert manager.bm25.search("anything", top_k=5) == []


# ---------------------------------------------------------------------------
# exclusive_session() — concurrent-write safety
# ---------------------------------------------------------------------------


def _ingest_one_doc(mgr: IndexManager, doc_store, trace_store, tmp_path, name: str) -> None:
    """Ingest a single, distinctly-named document against mgr's live indexes."""
    p = tmp_path / f"{name}.md"
    p.write_text(
        f"# {name}\n\nContent about {name} with enough words to clear the gate."
    )
    pipeline = IngestPipeline(
        doc_store, trace_store, keyword_index=mgr.bm25, vector_index=None,
    )
    result = pipeline.ingest(str(p))
    assert result.status == "ok", result.error


def _has_chunk_mentioning(index: BM25LocalIndex, name: str) -> bool:
    """True if any stored chunk's text mentions ``name``.

    Checks the definitive stored chunk list rather than ``.search()`` —
    BM25LocalIndex.search() always returns up to top_k results regardless of
    relevance score (no zero/negative-score filtering), so an empty corpus
    aside, "search returns nothing" is not a valid proxy for "content is
    absent" once the corpus is non-empty.
    """
    return any(name in c.display_text for c in index._chunks)


class TestExclusiveSessionCorrectness:
    def test_baseline_without_locking_loses_data(self, tmp_path):
        """Proves the bug exists: two managers loaded before either saves,
        mutated independently, sequential save — the second overwrites the
        first's addition. This is the exact failure exclusive_session() is
        built to prevent; compare against test_exclusive_session_prevents_lost_update.
        """
        index_dir = str(tmp_path / "indexes")
        db = str(tmp_path / "rag.db")
        doc_store, trace_store = _make_stores(db)

        # Both "sessions" load the same (empty) initial snapshot before
        # either one saves — the classic lost-update setup.
        mgr_a = IndexManager(index_dir)
        mgr_b = IndexManager(index_dir)

        _ingest_one_doc(mgr_a, doc_store, trace_store, tmp_path, "alpha")
        mgr_a.save()

        _ingest_one_doc(mgr_b, doc_store, trace_store, tmp_path, "beta")
        mgr_b.save()  # overwrites mgr_a's save — alpha is now gone from disk

        reloaded = IndexManager(index_dir)
        assert _has_chunk_mentioning(reloaded.bm25, "beta")
        assert not _has_chunk_mentioning(reloaded.bm25, "alpha")  # lost

    def test_exclusive_session_prevents_lost_update(self, tmp_path):
        """Same setup as the baseline above, but both writes go through
        exclusive_session() — neither is lost."""
        index_dir = str(tmp_path / "indexes")
        db = str(tmp_path / "rag.db")
        doc_store, trace_store = _make_stores(db)

        mgr_a = IndexManager(index_dir)
        mgr_b = IndexManager(index_dir)  # constructed before mgr_a's session runs

        with mgr_a.exclusive_session():
            _ingest_one_doc(mgr_a, doc_store, trace_store, tmp_path, "alpha")

        # mgr_b's initial load (at construction, above) is now stale — its
        # session must reload fresh and see alpha before beta is added.
        with mgr_b.exclusive_session():
            assert _has_chunk_mentioning(mgr_b.bm25, "alpha")  # reload saw it
            _ingest_one_doc(mgr_b, doc_store, trace_store, tmp_path, "beta")

        reloaded = IndexManager(index_dir)
        assert _has_chunk_mentioning(reloaded.bm25, "alpha")
        assert _has_chunk_mentioning(reloaded.bm25, "beta")

    def test_concurrent_sessions_block_and_both_survive(self, tmp_path):
        """Two threads entering exclusive_session() concurrently must
        serialize — not race — and the second must see the first's write
        after its own reload."""
        import threading
        import time

        index_dir = str(tmp_path / "indexes")
        db = str(tmp_path / "rag.db")
        doc_store, trace_store = _make_stores(db)
        errors: list[Exception] = []

        def worker(name: str, delay: float) -> None:
            try:
                mgr = IndexManager(index_dir)
                with mgr.exclusive_session():
                    time.sleep(delay)  # hold the lock briefly, simulating real work
                    _ingest_one_doc(mgr, doc_store, trace_store, tmp_path, name)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("alpha", 0.2))
        t2 = threading.Thread(target=worker, args=("beta", 0.0))
        t1.start()
        time.sleep(0.05)  # ensure t1 acquires the lock first
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        reloaded = IndexManager(index_dir)
        assert _has_chunk_mentioning(reloaded.bm25, "alpha")
        assert _has_chunk_mentioning(reloaded.bm25, "beta")

    def test_exception_inside_session_does_not_save_partial_state(self, tmp_path):
        index_dir = str(tmp_path / "indexes")
        mgr = IndexManager(index_dir)

        with pytest.raises(RuntimeError):
            with mgr.exclusive_session():
                mgr.bm25._chunks.append(
                    Chunk(
                        doc_id="d1",
                        stable_text="x",
                        display_text="x",
                        chunk_signature="deadbeef",
                    )
                )
                raise RuntimeError("simulated failure mid-ingest")

        # Nothing was saved — a fresh manager sees no trace of the aborted write.
        reloaded = IndexManager(index_dir)
        assert reloaded.bm25._chunks == []

    def test_exception_inside_session_still_releases_lock(self, tmp_path):
        index_dir = str(tmp_path / "indexes")
        mgr = IndexManager(index_dir)

        with pytest.raises(RuntimeError):
            with mgr.exclusive_session():
                raise RuntimeError("simulated failure")

        # Lock must be free — a subsequent session must not time out.
        mgr2 = IndexManager(index_dir)
        with mgr2.exclusive_session(timeout=1.0):
            pass  # must not raise IndexLockTimeoutError

    def test_timeout_raises_clear_error_when_lock_held_elsewhere(self, tmp_path):
        from filelock import FileLock
        from rag.infra.indexes.index_manager import IndexLockTimeoutError, _LOCK_FILENAME

        index_dir = tmp_path / "indexes"
        index_dir.mkdir()
        external_lock = FileLock(str(index_dir / _LOCK_FILENAME))
        external_lock.acquire()
        try:
            mgr = IndexManager(str(index_dir))
            with pytest.raises(IndexLockTimeoutError):
                with mgr.exclusive_session(timeout=0.2):
                    pass  # pragma: no cover - should never reach here
        finally:
            external_lock.release()
