"""Abstract base class for keyword index plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.candidate import Candidate
from rag.core.contracts.chunk import Chunk


class BaseKeywordIndex(ABC):
    """Interface that all keyword index plugins must implement.

    Keyword indexes support lexical retrieval using term-frequency methods
    such as BM25 or TF-IDF. Implementations include rank-bm25 and OpenSearch.

    The index is populated during ingestion (``add``) and queried during
    retrieval (``search``). Implementations must support persistence.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], rebuild: bool = True) -> None:
        """Add chunks to the keyword index using their ``stable_text``.

        Args:
            chunks: Chunks to index. The ``stable_text`` field is used
                for tokenisation and indexing.
            rebuild: Rebuild the searchable index immediately after adding.
                Batch callers adding many chunks in a loop should pass
                ``False`` and call :meth:`rebuild` once at the end —
                rebuilding after every single add is O(total chunks) per
                call, making a naive per-document loop O(N^2) overall.
        """

    @abstractmethod
    def rebuild(self) -> None:
        """Rebuild the searchable index from all chunks added so far.

        Only needed when chunks were added with ``rebuild=False``; a no-op
        (but safe to call) if the index is already up to date.
        """

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[Candidate]:
        """Retrieve the top-k most relevant chunks by keyword match.

        Args:
            query: Raw query string (tokenised internally).
            top_k: Number of candidates to return.

        Returns:
            List of Candidates ordered by descending BM25 score,
            with ``bm25_score`` populated and ``retrieval_source`` set
            to ``RetrievalSource.BM25``.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: Directory path where index files are written.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously persisted index from disk.

        Args:
            path: Directory path containing the saved index files.
        """
