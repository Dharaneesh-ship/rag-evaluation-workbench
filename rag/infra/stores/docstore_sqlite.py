"""SQLite implementation of the DocStore — schema creation and CRUD."""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from rag.core.contracts.chunk import Chunk
from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType
from rag.core.contracts.text_block import TextBlock
from rag.core.interfaces.doc_store import BaseDocStore


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    mime_type       TEXT NOT NULL DEFAULT '',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    fingerprint     TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_TEXT_BLOCKS = """
CREATE TABLE IF NOT EXISTS text_blocks (
    block_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL,
    block_type        TEXT NOT NULL DEFAULT 'paragraph',
    text              TEXT NOT NULL,
    block_hash        TEXT NOT NULL,
    page              INTEGER,
    sequence          INTEGER NOT NULL,
    section_path_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    stable_text         TEXT NOT NULL,
    display_text        TEXT NOT NULL,
    chunk_signature     TEXT NOT NULL,
    block_hashes_json   TEXT NOT NULL DEFAULT '[]',
    token_count         INTEGER NOT NULL DEFAULT 0,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
"""

_DDL_CONNECTOR_STATE = """
CREATE TABLE IF NOT EXISTS connector_state (
    connector_name  TEXT PRIMARY KEY,
    cursor          TEXT NOT NULL DEFAULT '',
    last_sync_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_INDEXES = [
    # text_blocks
    "CREATE INDEX IF NOT EXISTS idx_text_blocks_doc_id ON text_blocks(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_text_blocks_block_hash ON text_blocks(block_hash);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_text_blocks_doc_seq ON text_blocks(doc_id, sequence);",
    # chunks
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_signature ON chunks(chunk_signature);",
]


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_schema(db_path: str | Path) -> None:
    """Create the DocStore schema in a SQLite database.

    Safe to call on an existing database — all statements use
    ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS``.

    Args:
        db_path: Path to the SQLite database file. The file and any
            parent directories are created if they do not exist.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute(_DDL_DOCUMENTS)
            conn.execute(_DDL_TEXT_BLOCKS)
            conn.execute(_DDL_CHUNKS)
            conn.execute(_DDL_CONNECTOR_STATE)
            for ddl in _DDL_INDEXES:
                conn.execute(ddl)
            conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to initialise DocStore schema at {db_path}: {exc}") from exc


def get_tables(db_path: str | Path) -> list[str]:
    """Return names of all user tables in the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Sorted list of table names.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
    return [row[0] for row in rows]


def get_indexes(db_path: str | Path) -> list[str]:
    """Return names of all indexes in the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Sorted list of index names.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;"
        ).fetchall()
    return [row[0] for row in rows]


# ── SQLiteDocStore ─────────────────────────────────────────────────────────────

class SQLiteDocStore(BaseDocStore):
    """SQLite-backed document store.

    Initialises the schema on construction and exposes CRUD methods for
    documents, text blocks, and chunks.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        init_schema(self._db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Documents ──────────────────────────────────────────────────────────────

    def save_document(self, document: Document) -> str:
        """Persist a Document and return its doc_id.

        Args:
            document: The parsed Document to store.

        Returns:
            The ``doc_id`` of the stored document.
        """
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                        (doc_id, source_path, mime_type, metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        document.doc_id,
                        document.source_path,
                        document.mime_type,
                        json.dumps(document.metadata),
                    ),
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to save document {document.doc_id}: {exc}") from exc
        return document.doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a Document by its doc_id.

        Args:
            doc_id: Unique document identifier.

        Returns:
            The Document if found, or None.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT doc_id, source_path, mime_type, metadata_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return Document(
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            mime_type=row["mime_type"],
            metadata=json.loads(row["metadata_json"]),
        )

    def document_exists(self, doc_id: str) -> bool:
        """Check whether a document with the given doc_id exists.

        Args:
            doc_id: Unique document identifier.

        Returns:
            True if the document exists in the store.
        """
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
        return count > 0

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all associated blocks and chunks.

        Args:
            doc_id: Unique document identifier to delete.
        """
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to delete document {doc_id}: {exc}") from exc

    # ── Text blocks ────────────────────────────────────────────────────────────

    def save_text_blocks(self, blocks: list[TextBlock]) -> None:
        """Persist a list of TextBlocks.

        Args:
            blocks: TextBlocks to store.
        """
        rows = []
        for block in blocks:
            block_id = block.block_id or str(uuid.uuid4())
            rows.append((
                block_id,
                block.doc_id,
                block.block_type.value,
                block.text,
                block.block_hash,
                block.page,
                block.sequence,
                json.dumps(block.section_path),
            ))
        try:
            with self._conn() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO text_blocks
                        (block_id, doc_id, block_type, text, block_hash,
                         page, sequence, section_path_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to save text blocks: {exc}") from exc

    def get_text_blocks(self, doc_id: str) -> list[TextBlock]:
        """Retrieve all TextBlocks for a document, ordered by sequence.

        Args:
            doc_id: Parent document identifier.

        Returns:
            Ordered list of TextBlocks.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT block_id, doc_id, block_type, text, block_hash,
                       page, sequence, section_path_json
                FROM text_blocks WHERE doc_id = ? ORDER BY sequence
                """,
                (doc_id,),
            ).fetchall()
        return [
            TextBlock(
                block_id=row["block_id"],
                doc_id=row["doc_id"],
                block_type=BlockType(row["block_type"]),
                text=row["text"],
                block_hash=row["block_hash"],
                page=row["page"],
                sequence=row["sequence"],
                section_path=json.loads(row["section_path_json"]),
            )
            for row in rows
        ]

    def get_prev_doc_id_for_source(self, source_path: str) -> Optional[str]:
        """Return the doc_id of the most recently stored version of a source path.

        Used by Task 11.5 (stale chunk removal) to identify the previous
        document version before its chunks are purged from the indexes.

        Args:
            source_path: Absolute path to the source file.

        Returns:
            doc_id string if a previous version exists, otherwise None.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE source_path = ? ORDER BY rowid DESC LIMIT 1",
                (source_path,),
            ).fetchone()
        return row["doc_id"] if row else None

    def get_prev_blocks_for_source(self, source_path: str) -> list[TextBlock]:
        """Fetch blocks for the most recently stored version of a source path.

        Used by the incremental ingestion block diff logic (Task 11.2) to
        compare the previous document version against the newly ingested one.

        Args:
            source_path: Absolute path to the source file.

        Returns:
            Ordered list of TextBlocks for the previous version, or empty list
            if no previous version exists.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE source_path = ? ORDER BY rowid DESC LIMIT 1",
                (source_path,),
            ).fetchone()
        if row is None:
            return []
        return self.get_text_blocks(row["doc_id"])

    # ── Chunks ─────────────────────────────────────────────────────────────────

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist a list of Chunks.

        Args:
            chunks: Chunks to store.
        """
        rows = []
        for chunk in chunks:
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            rows.append((
                chunk_id,
                chunk.doc_id,
                chunk.stable_text,
                chunk.display_text,
                chunk.chunk_signature,
                json.dumps(chunk.block_hashes),
                chunk.token_count,
                json.dumps(chunk.metadata),
            ))
        try:
            with self._conn() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks
                        (chunk_id, doc_id, stable_text, display_text,
                         chunk_signature, block_hashes_json, token_count, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to save chunks: {exc}") from exc

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Retrieve all Chunks for a document.

        Args:
            doc_id: Parent document identifier.

        Returns:
            List of Chunks for the document.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, stable_text, display_text,
                       chunk_signature, block_hashes_json, token_count, metadata_json
                FROM chunks WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single Chunk by its chunk_id.

        Args:
            chunk_id: Unique chunk identifier.

        Returns:
            The Chunk if found, or None.
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT chunk_id, doc_id, stable_text, display_text,
                       chunk_signature, block_hashes_json, token_count, metadata_json
                FROM chunks WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def list_documents(self) -> list[dict]:
        """Return a summary list of all ingested documents.

        Returns:
            List of dicts with keys: doc_id, source_path, mime_type,
            collection (from metadata_json, or None if never set).
            Ordered by rowid descending (most recently ingested first).
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id, source_path, mime_type, metadata_json FROM documents ORDER BY rowid DESC"
            ).fetchall()
        return [
            {
                "doc_id": r["doc_id"],
                "source_path": r["source_path"],
                "mime_type": r["mime_type"],
                "collection": json.loads(r["metadata_json"]).get("collection"),
            }
            for r in rows
        ]

    def get_all_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        """Retrieve chunks, optionally filtered to a single document.

        Args:
            doc_id: If provided, only return chunks for this document.
                    If None, return all chunks across all documents.

        Returns:
            List of Chunks ordered by doc_id then rowid.
        """
        with self._conn() as conn:
            if doc_id:
                rows = conn.execute(
                    """SELECT chunk_id, doc_id, stable_text, display_text,
                              chunk_signature, block_hashes_json, token_count, metadata_json
                       FROM chunks WHERE doc_id = ? ORDER BY rowid""",
                    (doc_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT chunk_id, doc_id, stable_text, display_text,
                              chunk_signature, block_hashes_json, token_count, metadata_json
                       FROM chunks ORDER BY doc_id, rowid"""
                ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    # ------------------------------------------------------------------
    # Connector state (cursor persistence) — Task 15.1
    # ------------------------------------------------------------------

    def save_connector_cursor(self, connector_name: str, cursor: str) -> None:
        """Persist the sync cursor for a named connector.

        Uses an upsert so the first call creates the row and subsequent
        calls update it.

        Args:
            connector_name: Stable connector identifier (primary key).
            cursor: Opaque cursor string returned by the connector.
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO connector_state (connector_name, cursor, last_sync_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(connector_name) DO UPDATE SET
                    cursor = excluded.cursor,
                    last_sync_at = excluded.last_sync_at
                """,
                (connector_name, cursor),
            )
            conn.commit()

    def load_connector_cursor(self, connector_name: str) -> str:
        """Load the last persisted cursor for a named connector.

        Args:
            connector_name: Stable connector identifier.

        Returns:
            Cursor string, or ``""`` if no cursor has been saved yet.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT cursor FROM connector_state WHERE connector_name = ?",
                (connector_name,),
            ).fetchone()
        return row["cursor"] if row else ""

    def get_connector_state(self, connector_name: str) -> Optional[dict]:
        """Return the full connector state row as a dict.

        Args:
            connector_name: Stable connector identifier.

        Returns:
            Dict with ``connector_name``, ``cursor``, and ``last_sync_at``
            keys, or ``None`` if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT connector_name, cursor, last_sync_at FROM connector_state WHERE connector_name = ?",
                (connector_name,),
            ).fetchone()
        if row is None:
            return None
        return {
            "connector_name": row["connector_name"],
            "cursor": row["cursor"],
            "last_sync_at": row["last_sync_at"],
        }

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            stable_text=row["stable_text"],
            display_text=row["display_text"],
            chunk_signature=row["chunk_signature"],
            block_hashes=json.loads(row["block_hashes_json"]),
            token_count=row["token_count"],
            metadata=json.loads(row["metadata_json"]),
        )
