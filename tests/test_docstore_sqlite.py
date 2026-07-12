"""Integration tests for SQLiteDocStore write/read methods."""

import pytest

from rag.core.contracts.chunk import Chunk
from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType
from rag.core.contracts.text_block import TextBlock
from rag.infra.stores.docstore_sqlite import SQLiteDocStore


@pytest.fixture
def store(tmp_path):
    return SQLiteDocStore(tmp_path / "test.sqlite")


@pytest.fixture
def sample_doc():
    return Document(
        doc_id="doc-001",
        source_path="/tmp/sample.pdf",
        mime_type="application/pdf",
        metadata={"author": "Alice", "pages": 10},
    )


@pytest.fixture
def sample_blocks():
    return [
        TextBlock(
            block_id="blk-001",
            doc_id="doc-001",
            block_type=BlockType.HEADING,
            text="Introduction",
            block_hash="hash-001",
            page=1,
            sequence=0,
            section_path=[],
        ),
        TextBlock(
            block_id="blk-002",
            doc_id="doc-001",
            block_type=BlockType.PARAGRAPH,
            text="This document covers the deployment architecture.",
            block_hash="hash-002",
            page=1,
            sequence=1,
            section_path=["Introduction"],
        ),
    ]


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chk-001",
            doc_id="doc-001",
            stable_text="introduction this document covers the deployment architecture",
            display_text="Introduction\nThis document covers the deployment architecture.",
            chunk_signature="sig-001",
            block_hashes=["hash-001", "hash-002"],
            token_count=12,
            metadata={"page_start": 1},
        ),
    ]


@pytest.mark.integration
class TestDocumentCRUD:
    def test_save_and_get_document(self, store, sample_doc):
        store.save_document(sample_doc)
        retrieved = store.get_document("doc-001")
        assert retrieved is not None
        assert retrieved.doc_id == "doc-001"
        assert retrieved.source_path == "/tmp/sample.pdf"
        assert retrieved.mime_type == "application/pdf"
        assert retrieved.metadata["author"] == "Alice"

    def test_document_exists(self, store, sample_doc):
        assert not store.document_exists("doc-001")
        store.save_document(sample_doc)
        assert store.document_exists("doc-001")

    def test_get_nonexistent_document_returns_none(self, store):
        assert store.get_document("missing") is None

    def test_delete_document(self, store, sample_doc):
        store.save_document(sample_doc)
        store.delete_document("doc-001")
        assert not store.document_exists("doc-001")

    def test_save_document_upsert(self, store, sample_doc):
        store.save_document(sample_doc)
        updated = Document(
            doc_id="doc-001",
            source_path="/tmp/updated.pdf",
            mime_type="application/pdf",
        )
        store.save_document(updated)
        retrieved = store.get_document("doc-001")
        assert retrieved.source_path == "/tmp/updated.pdf"

    def test_list_documents_exposes_collection_from_metadata(self, store, sample_doc):
        store.save_document(sample_doc)  # metadata has no "collection" key
        tagged = Document(
            doc_id="doc-002",
            source_path="/tmp/tagged.pdf",
            metadata={"collection": "wake_apparel"},
        )
        store.save_document(tagged)

        docs = {d["doc_id"]: d for d in store.list_documents()}
        assert docs["doc-001"]["collection"] is None
        assert docs["doc-002"]["collection"] == "wake_apparel"


@pytest.mark.integration
class TestTextBlockCRUD:
    def test_save_and_get_text_blocks(self, store, sample_doc, sample_blocks):
        store.save_document(sample_doc)
        store.save_text_blocks(sample_blocks)
        blocks = store.get_text_blocks("doc-001")
        assert len(blocks) == 2
        assert blocks[0].sequence == 0
        assert blocks[1].sequence == 1
        assert blocks[0].block_type == BlockType.HEADING
        assert blocks[0].section_path == []
        assert blocks[1].section_path == ["Introduction"]

    def test_get_text_blocks_empty(self, store, sample_doc):
        store.save_document(sample_doc)
        assert store.get_text_blocks("doc-001") == []

    def test_blocks_deleted_with_document(self, store, sample_doc, sample_blocks):
        store.save_document(sample_doc)
        store.save_text_blocks(sample_blocks)
        store.delete_document("doc-001")
        assert store.get_text_blocks("doc-001") == []


@pytest.mark.integration
class TestChunkCRUD:
    def test_save_and_get_chunks(self, store, sample_doc, sample_chunks):
        store.save_document(sample_doc)
        store.save_chunks(sample_chunks)
        chunks = store.get_chunks("doc-001")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chk-001"
        assert chunks[0].token_count == 12
        assert chunks[0].block_hashes == ["hash-001", "hash-002"]
        assert chunks[0].metadata["page_start"] == 1

    def test_get_chunk_by_id(self, store, sample_doc, sample_chunks):
        store.save_document(sample_doc)
        store.save_chunks(sample_chunks)
        chunk = store.get_chunk_by_id("chk-001")
        assert chunk is not None
        assert chunk.chunk_signature == "sig-001"

    def test_get_chunk_by_id_missing(self, store):
        assert store.get_chunk_by_id("nonexistent") is None

    def test_get_chunks_for_doc_id(self, store, sample_doc, sample_chunks):
        store.save_document(sample_doc)
        store.save_chunks(sample_chunks)
        assert len(store.get_chunks("doc-001")) == 1
        assert store.get_chunks("other-doc") == []

    def test_chunks_deleted_with_document(self, store, sample_doc, sample_chunks):
        store.save_document(sample_doc)
        store.save_chunks(sample_chunks)
        store.delete_document("doc-001")
        assert store.get_chunks("doc-001") == []
