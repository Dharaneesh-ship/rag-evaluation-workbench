"""Index startup loader — initialises or reloads BM25 and FAISS indexes from disk."""

import logging
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout as _FileLockTimeout

from rag.infra.indexes.bm25_local import BM25LocalIndex, _INDEX_FILENAME as _BM25_FILE
from rag.infra.indexes.faiss_local import FaissLocalIndex, _INDEX_FILENAME as _FAISS_FILE
from rag.infra.indexes.manifest import IndexManifest

logger = logging.getLogger(__name__)

_LOCK_FILENAME = ".index.lock"


class IndexLockTimeoutError(TimeoutError):
    """Raised when ``exclusive_session()`` cannot acquire the index lock in time.

    Means another process holds the lock — almost always a concurrent
    ingest/sync already in progress against the same ``index_dir``.
    """


class IndexManager:
    """Manages lifecycle of BM25 and FAISS indexes with disk persistence.

    On construction, ``IndexManager`` checks whether saved index files exist
    at ``index_dir``:

    - If both ``faiss.index`` / ``faiss_mapping.pkl`` and ``bm25.pkl`` are
      found, they are loaded into memory immediately.
    - If the directory is absent or the files do not exist, fresh empty
      indexes are created (first-run case).

    After ingestion the caller should call :meth:`save` to persist the
    updated indexes so they survive process restarts.

    Usage::

        manager = IndexManager("/data/indexes")
        # manager.bm25 and manager.faiss are ready to use

        # After ingestion:
        manager.save()

    Args:
        index_dir: Directory where index files are stored.
    """

    def __init__(self, index_dir: str | Path) -> None:
        self._index_dir = Path(index_dir)
        self.bm25: BM25LocalIndex = BM25LocalIndex()
        self.faiss: FaissLocalIndex = FaissLocalIndex()
        self.manifest: IndexManifest = IndexManifest.empty()
        self._load_if_exists()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        """Load indexes from disk if saved files are present."""
        bm25_path = self._index_dir / _BM25_FILE
        faiss_path = self._index_dir / _FAISS_FILE

        if bm25_path.exists():
            try:
                self.bm25.load(str(self._index_dir))
                logger.info("BM25 index loaded from '%s'.", self._index_dir)
            except Exception as exc:
                logger.warning(
                    "Failed to load BM25 index from '%s': %s. Starting empty.",
                    self._index_dir,
                    exc,
                )
                self.bm25 = BM25LocalIndex()
        else:
            logger.info(
                "No BM25 index found at '%s'. Starting with empty index.", self._index_dir
            )

        if faiss_path.exists():
            try:
                self.faiss.load(str(self._index_dir))
                logger.info("FAISS index loaded from '%s'.", self._index_dir)
            except Exception as exc:
                logger.warning(
                    "Failed to load FAISS index from '%s': %s. Starting empty.",
                    self._index_dir,
                    exc,
                )
                self.faiss = FaissLocalIndex()
        else:
            logger.info(
                "No FAISS index found at '%s'. Starting with empty index.", self._index_dir
            )

        self.manifest = IndexManifest.load(self._index_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_embedding_config(
        self,
        provider: str | None,
        model: str | None,
        dimension: int | None,
    ) -> None:
        """Record the embedding configuration that produced this index's vectors.

        Called once vectors have actually been added during an ingest — see
        ``rag.app.mcp_server.retrieval_config`` for how ``provider`` is
        resolved from a request's ``embedding_provider`` enum. Persisted by
        the next :meth:`save`.

        Args:
            provider: Concrete provider key (e.g. ``"openai"``), or None —
                a no-op, since a BM25-only ingest doesn't change what
                vectors (if any) already exist in this index_dir.
            model: Embedding model identifier.
            dimension: Vector dimensionality.
        """
        if provider is None:
            return
        self.manifest = IndexManifest(
            embedding_provider=provider,
            embedding_model=model,
            vector_dimension=dimension,
        )

    def save(self) -> None:
        """Persist both indexes and the manifest to disk.

        Creates ``index_dir`` if it does not yet exist.
        """
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(self._index_dir))
        self.faiss.save(str(self._index_dir))
        self.manifest.save(self._index_dir)
        logger.info("BM25, FAISS indexes and manifest saved to '%s'.", self._index_dir)

    def reload(self) -> None:
        """Reload both indexes from disk, replacing current in-memory state.

        Useful after an external process has updated the index files.
        """
        self.bm25 = BM25LocalIndex()
        self.faiss = FaissLocalIndex()
        self._load_if_exists()
        logger.info("Indexes reloaded from '%s'.", self._index_dir)

    @contextmanager
    def exclusive_session(self, timeout: float = 60.0):
        """Cross-process-safe load → mutate → save critical section.

        Without this, two concurrent writers (two ingest sessions, or an
        ingest racing a connector sync) each load a stale in-memory snapshot,
        mutate it independently, and the second one to finish silently
        overwrites the first one's additions — the whole-file dump-on-save
        pattern has no way to detect or merge concurrent changes. This
        acquires an exclusive file lock, reloads fresh from disk *inside*
        the lock (discarding any snapshot loaded before the lock was held,
        which may already be stale), yields ``self`` for the caller to
        mutate, and saves on clean exit.

        Usage — construct the pipeline *inside* the ``with`` block, not
        before it, so it references the freshly-reloaded ``bm25``/``faiss``
        objects rather than a pre-lock snapshot that ``reload()`` replaces::

            mgr = IndexManager(index_dir)
            with mgr.exclusive_session():
                pipeline = IngestPipeline(
                    doc_store, trace_store,
                    keyword_index=mgr.bm25, vector_index=mgr.faiss,
                )
                pipeline.ingest(path)
            # saved automatically on clean exit

        Args:
            timeout: Seconds to wait for the lock before giving up.
                Defaults to 60.

        Yields:
            ``self`` (for symmetry with the usage pattern above; callers
            typically just read ``mgr.bm25``/``mgr.faiss`` directly).

        Raises:
            IndexLockTimeoutError: If the lock isn't acquired within
                ``timeout`` seconds — another process is holding it.
        """
        self._index_dir.mkdir(parents=True, exist_ok=True)
        lock_path = str(self._index_dir / _LOCK_FILENAME)
        lock = FileLock(lock_path, timeout=timeout)
        try:
            lock.acquire()
        except _FileLockTimeout as exc:
            raise IndexLockTimeoutError(
                f"Could not acquire index lock for '{self._index_dir}' within "
                f"{timeout}s — another ingest or sync is likely in progress "
                f"against the same index directory."
            ) from exc

        try:
            self.reload()
            yield self
            self.save()
        finally:
            lock.release()
