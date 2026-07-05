"""Tests for FaissLocalIndex."""

import tempfile

import numpy as np
import pytest

from rag.core.contracts.candidate import RetrievalSource
from rag.core.contracts.chunk import Chunk
from rag.infra.indexes.faiss_local import FaissLocalIndex


def _make_chunk(
    chunk_id: str,
    embedding: list[float],
    doc_id: str = "doc1",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        stable_text=f"stable text for {chunk_id}",
        display_text=f"display text for {chunk_id}",
        chunk_signature=chunk_id,
        embedding=embedding,
    )


# Three simple 3-dim vectors that are easy to reason about
_VEC_A = [1.0, 0.0, 0.0]
_VEC_B = [0.0, 1.0, 0.0]
_VEC_C = [0.0, 0.0, 1.0]


@pytest.fixture()
def three_chunks() -> list[Chunk]:
    return [
        _make_chunk("chunk-a", _VEC_A),
        _make_chunk("chunk-b", _VEC_B),
        _make_chunk("chunk-c", _VEC_C),
    ]


@pytest.fixture()
def loaded_index(three_chunks: list[Chunk]) -> FaissLocalIndex:
    idx = FaissLocalIndex()
    idx.add(three_chunks)
    return idx


# ── Basic search ───────────────────────────────────────────────────────────────


def test_search_returns_correct_chunk(loaded_index: FaissLocalIndex) -> None:
    """Query closest to VEC_A should return chunk-a first."""
    results = loaded_index.search(_VEC_A, top_k=3)
    assert results[0].chunk_id == "chunk-a"


def test_search_returns_candidate_objects(loaded_index: FaissLocalIndex) -> None:
    """Results must be Candidate instances with VECTOR source."""
    results = loaded_index.search(_VEC_A, top_k=1)
    assert len(results) == 1
    assert results[0].retrieval_source == RetrievalSource.VECTOR
    assert results[0].vector_score is not None


def test_search_top_k_limits_results(loaded_index: FaissLocalIndex) -> None:
    results = loaded_index.search(_VEC_A, top_k=2)
    assert len(results) == 2


def test_search_empty_index_returns_empty() -> None:
    idx = FaissLocalIndex()
    results = idx.search(_VEC_A, top_k=5)
    assert results == []


# ── Removal ───────────────────────────────────────────────────────────────────


def test_remove_chunk_not_in_results(loaded_index: FaissLocalIndex) -> None:
    """After removing chunk-a, a query for VEC_A should not return chunk-a."""
    loaded_index.remove("chunk-a")
    results = loaded_index.search(_VEC_A, top_k=3)
    ids = [r.chunk_id for r in results]
    assert "chunk-a" not in ids


def test_remove_unknown_chunk_no_error(loaded_index: FaissLocalIndex) -> None:
    """Removing a non-existent chunk_id must not raise."""
    loaded_index.remove("nonexistent-id")  # should not raise


def test_remove_all_chunks_returns_empty(loaded_index: FaissLocalIndex) -> None:
    for cid in ["chunk-a", "chunk-b", "chunk-c"]:
        loaded_index.remove(cid)
    assert loaded_index.search(_VEC_A, top_k=3) == []


# ── Persistence ───────────────────────────────────────────────────────────────


def test_save_and_load(loaded_index: FaissLocalIndex, three_chunks: list[Chunk]) -> None:
    """Save then reload; results must match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded_index.save(tmpdir)

        idx2 = FaissLocalIndex()
        idx2.load(tmpdir)

        results = idx2.search(_VEC_B, top_k=1)
        assert results[0].chunk_id == "chunk-b"


def test_load_preserves_metadata(
    loaded_index: FaissLocalIndex, three_chunks: list[Chunk]
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded_index.save(tmpdir)
        idx2 = FaissLocalIndex()
        idx2.load(tmpdir)

        results = idx2.search(_VEC_C, top_k=1)
        assert results[0].doc_id == "doc1"
        assert results[0].stable_text == "stable text for chunk-c"


def test_load_missing_index_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = FaissLocalIndex()
        with pytest.raises(FileNotFoundError):
            idx.load(tmpdir)


# ── Error handling ─────────────────────────────────────────────────────────────


def test_add_chunk_without_embedding_raises() -> None:
    idx = FaissLocalIndex()
    chunk = Chunk(
        chunk_id="no-embed",
        doc_id="doc1",
        stable_text="text",
        display_text="text",
        chunk_signature="sig",
        embedding=None,
    )
    with pytest.raises(ValueError, match="no embedding"):
        idx.add([chunk])


def test_add_chunk_without_chunk_id_raises() -> None:
    idx = FaissLocalIndex()
    chunk = Chunk(
        chunk_id=None,
        doc_id="doc1",
        stable_text="text",
        display_text="text",
        chunk_signature="sig",
        embedding=[1.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="chunk_id"):
        idx.add([chunk])


# ── Deferred rebuild (batch ingest throughput) ─────────────────────────────────


def test_add_with_rebuild_false_not_searchable_until_rebuild(three_chunks):
    idx = FaissLocalIndex()
    idx.add(three_chunks, rebuild=False)
    assert idx.dimension == 0  # index not built yet
    assert idx.search(_VEC_A, top_k=3) == []

    idx.rebuild()
    assert idx.dimension == 3
    results = idx.search(_VEC_A, top_k=3)
    assert len(results) == 3


def test_multiple_deferred_adds_then_single_rebuild(three_chunks):
    idx = FaissLocalIndex()
    for chunk in three_chunks:
        idx.add([chunk], rebuild=False)
    assert idx.dimension == 0

    idx.rebuild()
    results = idx.search(_VEC_A, top_k=1)
    assert results[0].chunk_id == "chunk-a"


def test_add_default_rebuild_true_matches_prior_behavior(three_chunks):
    idx = FaissLocalIndex()
    idx.add(three_chunks)  # rebuild defaults to True
    assert idx.dimension == 3
    assert len(idx.search(_VEC_A, top_k=3)) == 3


def test_deferred_add_does_not_rebuild_per_call(three_chunks):
    idx = FaissLocalIndex()
    call_count = {"n": 0}
    original_rebuild = idx._rebuild

    def counting_rebuild():
        call_count["n"] += 1
        original_rebuild()

    idx._rebuild = counting_rebuild
    for chunk in three_chunks:
        idx.add([chunk], rebuild=False)
    assert call_count["n"] == 0

    idx.rebuild()
    assert call_count["n"] == 1
