"""Integration tests for the minimal ingest pipeline."""

import tempfile
from pathlib import Path

import pytest

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.indexes.bm25_local import BM25LocalIndex
from rag.infra.indexes.faiss_local import FaissLocalIndex
from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema as init_doc_schema
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore, init_schema as init_trace_schema
from rag.pipelines.ingest_pipeline import IngestPipeline


# ---------------------------------------------------------------------------
# Stub embedding provider (no external API required)
# ---------------------------------------------------------------------------

_DIM = 8


class StubEmbeddingProvider(BaseEmbeddingProvider):
    """Returns deterministic unit vectors (no API call)."""

    @property
    def dim(self) -> int:
        return _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        # Each text gets a unique-ish vector based on length
        return [[float(len(t) % _DIM == i) for i in range(_DIM)] for t in texts]

SAMPLE_MD = Path(__file__).parent.parent / "tests" / "fixtures" / "sample.md"
SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def _make_stores(db_path: str):
    init_doc_schema(db_path)
    init_trace_schema(db_path)
    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)
    return doc_store, trace_store


def _make_pipeline(db_path: str) -> IngestPipeline:
    doc_store, trace_store = _make_stores(db_path)
    return IngestPipeline(doc_store, trace_store)


@pytest.fixture()
def sample_md(tmp_path: Path) -> str:
    p = tmp_path / "sample.md"
    p.write_text(
        "# Introduction\n\nThis is the first paragraph.\n\n"
        "## Section One\n\nContent of section one.\n\n"
        "## Section Two\n\nContent of section two with more details.\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def sample_txt(tmp_path: Path) -> str:
    p = tmp_path / "sample.txt"
    p.write_text(
        "First paragraph of text.\n\n"
        "Second paragraph of text.\n\n"
        "Third paragraph of text.\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def sample_html(tmp_path: Path) -> str:
    p = tmp_path / "sample.html"
    p.write_text(
        "<!DOCTYPE html><html><head><title>Test</title></head>"
        "<body><article>"
        "<h1>Article Title</h1>"
        "<p>First paragraph with meaningful content.</p>"
        "<p>Second paragraph with more content here.</p>"
        "</article></body></html>",
        encoding="utf-8",
    )
    return str(p)


# ---------------------------------------------------------------------------
# Markdown ingestion
# ---------------------------------------------------------------------------


def test_ingest_markdown_succeeds(sample_md, tmp_path):
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest(sample_md)
    assert result.error is None
    assert result.doc_id
    assert result.block_count > 0
    assert result.chunk_count > 0


def test_ingest_markdown_stores_document(sample_md, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_md)
    doc = doc_store.get_document(result.doc_id)
    assert doc is not None
    assert doc.source_path == result.source_path


def test_ingest_markdown_stores_text_blocks(sample_md, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_md)
    blocks = doc_store.get_text_blocks(result.doc_id)
    assert len(blocks) > 0


def test_ingest_markdown_stores_chunks(sample_md, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_md)
    chunks = doc_store.get_chunks(result.doc_id)
    assert len(chunks) > 0


def test_ingest_markdown_writes_run(sample_md, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_md)
    assert result.run_id
    runs = trace_store.list_runs(run_type="ingest")
    assert len(runs) >= 1


# ---------------------------------------------------------------------------
# TXT ingestion
# ---------------------------------------------------------------------------


def test_ingest_txt_succeeds(sample_txt, tmp_path):
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest(sample_txt)
    assert result.error is None
    assert result.block_count > 0


# ---------------------------------------------------------------------------
# HTML ingestion
# ---------------------------------------------------------------------------


def test_ingest_html_succeeds(sample_html, tmp_path):
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest(sample_html)
    assert result.error is None


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------


def test_ingest_pdf_succeeds(tmp_path):
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest(str(SAMPLE_PDF))
    assert result.error is None
    assert result.block_count > 0
    assert result.chunk_count > 0


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


def test_ingest_result_has_elapsed_ms(sample_md, tmp_path):
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest(sample_md)
    assert result.elapsed_ms > 0


def test_ingest_nonexistent_file_returns_error(tmp_path):
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest("/nonexistent/path/file.txt")
    assert result.error is not None


# ---------------------------------------------------------------------------
# Embedding + index integration
# ---------------------------------------------------------------------------


def test_ingest_embeds_chunks_into_faiss(sample_txt, tmp_path):
    """After ingest, FAISS index should hold searchable vectors."""
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    provider = StubEmbeddingProvider()
    faiss_idx = FaissLocalIndex()

    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=provider,
        vector_index=faiss_idx,
    )
    result = pipeline.ingest(sample_txt)

    assert result.error is None
    assert result.chunk_count > 0
    # FAISS index must now contain the ingested vectors
    query = [0.0] * _DIM
    candidates = faiss_idx.search(query, top_k=result.chunk_count)
    assert len(candidates) == result.chunk_count


def test_ingest_updates_bm25_index(sample_txt, tmp_path):
    """After ingest, BM25 index should return relevant results."""
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    provider = StubEmbeddingProvider()
    bm25_idx = BM25LocalIndex()

    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=provider,
        keyword_index=bm25_idx,
    )
    result = pipeline.ingest(sample_txt)

    assert result.error is None
    candidates = bm25_idx.search("paragraph", top_k=5)
    assert len(candidates) > 0


def test_ingest_without_embedding_provider_skips_indexing(sample_txt, tmp_path):
    """Pipeline without an embedding provider must still succeed."""
    db = str(tmp_path / "rag.db")
    pipeline = _make_pipeline(db)
    result = pipeline.ingest(sample_txt)
    assert result.error is None
    assert result.embed_tokens == 0


def test_ingest_faiss_searchable_after_save_reload(sample_txt, tmp_path):
    """FAISS index persisted to disk and reloaded must still return results."""
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    provider = StubEmbeddingProvider()
    faiss_idx = FaissLocalIndex()

    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=provider,
        vector_index=faiss_idx,
    )
    pipeline.ingest(sample_txt)

    # Persist and reload into a fresh index
    index_dir = str(tmp_path / "indexes")
    faiss_idx.save(index_dir)

    faiss_idx2 = FaissLocalIndex()
    faiss_idx2.load(index_dir)

    query = [0.0] * _DIM
    candidates = faiss_idx2.search(query, top_k=5)
    assert len(candidates) > 0


# ---------------------------------------------------------------------------
# Collection tagging / isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_txt_other(tmp_path: Path) -> str:
    p = tmp_path / "sample_other.txt"
    p.write_text(
        "First paragraph about zebras.\n\n"
        "Second paragraph about zebras.\n\n"
        "Third paragraph about zebras.\n",
        encoding="utf-8",
    )
    return str(p)


def test_ingest_with_collection_stamps_chunk_metadata(sample_txt, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_txt, collection="wake_apparel")
    chunks = doc_store.get_chunks(result.doc_id)
    assert chunks
    assert all(c.metadata.get("collection") == "wake_apparel" for c in chunks)


def test_ingest_without_collection_leaves_chunk_metadata_unset(sample_txt, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_txt)
    chunks = doc_store.get_chunks(result.doc_id)
    assert chunks
    assert all("collection" not in c.metadata for c in chunks)


def test_ingest_with_collection_stamps_document_metadata(sample_txt, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    result = pipeline.ingest(sample_txt, collection="wake_apparel")
    doc = doc_store.get_document(result.doc_id)
    assert doc.metadata.get("collection") == "wake_apparel"


def test_list_documents_reports_collection(sample_txt, sample_txt_other, tmp_path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    pipeline = IngestPipeline(doc_store, trace_store)
    r1 = pipeline.ingest(sample_txt, collection="wake_apparel")
    r2 = pipeline.ingest(sample_txt_other)

    docs = {d["doc_id"]: d for d in doc_store.list_documents()}
    assert docs[r1.doc_id]["collection"] == "wake_apparel"
    assert docs[r2.doc_id]["collection"] is None


def test_bm25_search_scoped_to_collection_excludes_other_collections(
    sample_txt, sample_txt_other, tmp_path
):
    """End-to-end: a collection-scoped query must not surface chunks from an
    unrelated document ingested without that collection tag — this is the
    isolation guarantee the 'collection' field exists to provide."""
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    bm25_idx = BM25LocalIndex()
    pipeline = IngestPipeline(doc_store, trace_store, keyword_index=bm25_idx)

    r1 = pipeline.ingest(sample_txt, collection="wake_apparel")
    r2 = pipeline.ingest(sample_txt_other)  # unrelated doc, no collection tag

    # Unscoped search sees both documents' chunks.
    unscoped = bm25_idx.search("paragraph", top_k=10)
    assert len(unscoped) == r1.chunk_count + r2.chunk_count

    # Scoped search only sees the tagged document's chunks.
    scoped = bm25_idx.search("paragraph", top_k=10, collection="wake_apparel")
    assert len(scoped) == r1.chunk_count
    assert all(c.metadata.get("collection") == "wake_apparel" for c in scoped)
    assert all(c.doc_id == r1.doc_id for c in scoped)
