"""Local FAISS vector index with disk persistence and chunk-level deletion."""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.chunk import Chunk
from rag.core.interfaces.vector_index import BaseVectorIndex

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "faiss.index"
_MAPPING_FILENAME = "faiss_mapping.pkl"


class FaissLocalIndex(BaseVectorIndex):
    """FAISS flat L2 vector index with in-memory chunk_id → vector mapping.

    FAISS does not natively support deletion. This implementation maintains
    a ``chunk_id → vector`` mapping in memory. Removal filters the mapping
    and rebuilds the FAISS index from the remaining vectors, which is
    acceptable for V1 corpus sizes.

    Persistence writes two files to a given directory:
    - ``faiss.index``       — the serialised FAISS index
    - ``faiss_mapping.pkl`` — the chunk_id order list and chunk metadata

    Usage::

        index = FaissLocalIndex()
        index.add(chunks)                          # chunks with embeddings
        candidates = index.search(query_vec, 5)
        index.save("/path/to/dir")

        index2 = FaissLocalIndex()
        index2.load("/path/to/dir")
        candidates2 = index2.search(query_vec, 5)
    """

    def __init__(self) -> None:
        # chunk_id → (vector, Chunk) for in-memory management
        self._store: dict[str, tuple[np.ndarray, Chunk]] = {}
        self._index: Optional[faiss.Index] = None
        # ordered list of chunk_ids mirroring FAISS internal row order
        self._id_order: list[str] = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild the FAISS flat index from the current in-memory store."""
        if not self._store:
            self._index = None
            self._id_order = []
            logger.debug("FAISS index cleared (no vectors remaining).")
            return

        self._id_order = list(self._store.keys())
        matrix = np.array(
            [self._store[cid][0] for cid in self._id_order], dtype="float32"
        )
        dim = matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(matrix)  # type: ignore[arg-type]
        self._index = index
        logger.debug("FAISS index rebuilt with %d vectors (dim=%d).", len(self._id_order), dim)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Vector dimension of the current index, or 0 if the index is empty."""
        if self._index is None:
            return 0
        return self._index.d

    # ── BaseVectorIndex ───────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk], rebuild: bool = True) -> None:
        """Add chunks (with pre-computed embeddings) to the index.

        Args:
            chunks: Chunks whose ``embedding`` field has been populated.
            rebuild: Rebuild the FAISS index immediately. Pass ``False``
                when adding many batches in a loop and call :meth:`rebuild`
                once afterwards — rebuilding reconstructs the full index
                from every vector added so far, so doing it on every call
                makes a batch of N documents O(N^2) instead of O(N).

        Raises:
            ValueError: If any chunk has a None or empty embedding or chunk_id.
        """
        added = 0
        for chunk in chunks:
            if chunk.embedding is None or len(chunk.embedding) == 0:
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' has no embedding. "
                    "Embed chunks before adding to the FAISS index."
                )
            if not chunk.chunk_id:
                raise ValueError("All chunks must have a non-empty chunk_id.")

            vec = np.array(chunk.embedding, dtype="float32")
            self._store[chunk.chunk_id] = (vec, chunk)
            added += 1

        if added and rebuild:
            self._rebuild()

    def rebuild(self) -> None:
        """Rebuild the FAISS index from all chunks added so far.

        Call this once after a batch of ``add(..., rebuild=False)`` calls.
        """
        self._rebuild()

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        collection: str | None = None,
    ) -> list[Candidate]:
        """Return top-k nearest chunks by L2 distance.

        Scores are returned as negative L2 distance so that higher is better,
        matching the convention used by ``vector_score`` across the pipeline.

        Args:
            query_vector: Dense query embedding.
            top_k: Maximum number of results.
            collection: Optional collection name to filter results.  When set,
                FAISS is queried with an overscan (``top_k * 5``) and the
                returned candidates are filtered to the requested collection
                before taking ``top_k``.  Falls back to unfiltered results if
                the filtered set is empty.

        Returns:
            Candidates ordered by ascending L2 distance (descending similarity).
            Returns an empty list if the index is empty.
        """
        if self._index is None or not self._id_order:
            return []

        q = np.array([query_vector], dtype="float32")
        # Overscan when filtering by collection so small collections aren't
        # entirely drowned out by a larger one ranked above them.
        fetch_k = min(top_k * 5 if collection else top_k, len(self._id_order))
        distances, indices = self._index.search(q, fetch_k)  # type: ignore[arg-type]

        candidates: list[Candidate] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                # FAISS returns -1 for padded results when k > index size
                continue
            chunk_id = self._id_order[idx]
            _, chunk = self._store[chunk_id]
            if collection and chunk.metadata.get("collection") != collection:
                continue  # skip chunks outside the requested collection
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    display_text=chunk.display_text,
                    stable_text=chunk.stable_text,
                    vector_score=float(-dist),  # negate so higher = more similar
                    retrieval_source=RetrievalSource.VECTOR,
                    metadata=chunk.metadata,
                )
            )
            if len(candidates) == top_k:
                break

        if not candidates and collection:
            logger.warning(
                "FAISS collection filter '%s' matched 0 results — "
                "falling back to unfiltered search.",
                collection,
            )
            return self.search(query_vector, top_k, collection=None)

        return candidates

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index by its chunk_id.

        Triggers a full index rebuild from the remaining vectors.

        Args:
            chunk_id: ID of the chunk to remove.
        """
        if chunk_id not in self._store:
            logger.warning("chunk_id '%s' not found in FAISS index.", chunk_id)
            return

        del self._store[chunk_id]
        self._rebuild()
        logger.debug("Removed chunk '%s' from FAISS index.", chunk_id)

    def save(self, path: str) -> None:
        """Persist the FAISS index and chunk_id mapping to disk.

        Writes:
        - ``faiss.index``       — serialised FAISS index bytes
        - ``faiss_mapping.pkl`` — (id_order, store metadata) for reconstruction

        Args:
            path: Directory where index files are written.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_path = dir_path / _INDEX_FILENAME
        mapping_path = dir_path / _MAPPING_FILENAME

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(index_path))
        else:
            # Write an empty marker so load() can detect empty state
            index_path.write_bytes(b"")

        # Save mapping: id_order + stripped chunk metadata (no large vectors)
        mapping = {
            "id_order": self._id_order,
            "chunks": {cid: chunk for cid, (_, chunk) in self._store.items()},
            "vectors": {cid: vec.tolist() for cid, (vec, _) in self._store.items()},
        }
        with mapping_path.open("wb") as f:
            pickle.dump(mapping, f)

        logger.debug(
            "FAISS index saved to '%s' (%d vectors).", dir_path, len(self._id_order)
        )

    def load(self, path: str) -> None:
        """Load a previously persisted index from disk.

        Args:
            path: Directory containing ``faiss.index`` and ``faiss_mapping.pkl``.

        Raises:
            FileNotFoundError: If either required file is missing.
        """
        dir_path = Path(path)
        index_path = dir_path / _INDEX_FILENAME
        mapping_path = dir_path / _MAPPING_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file not found at '{index_path}'. "
                "Run add() and save() first."
            )
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"FAISS mapping file not found at '{mapping_path}'. "
                "Run add() and save() first."
            )

        # Load mapping
        with mapping_path.open("rb") as f:
            mapping = pickle.load(f)

        chunks: dict[str, Chunk] = mapping["chunks"]
        vectors: dict[str, list[float]] = mapping["vectors"]

        # Reconstruct in-memory store
        self._store = {
            cid: (np.array(vectors[cid], dtype="float32"), chunks[cid])
            for cid in mapping["id_order"]
            if cid in chunks and cid in vectors
        }

        # Load FAISS index (empty file means zero vectors)
        index_bytes = index_path.read_bytes()
        if index_bytes and self._store:
            self._index = faiss.read_index(str(index_path))
            self._id_order = mapping["id_order"]
        else:
            self._index = None
            self._id_order = []

        logger.debug(
            "FAISS index loaded from '%s' (%d vectors).", dir_path, len(self._id_order)
        )
