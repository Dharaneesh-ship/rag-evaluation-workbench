"""Shared pytest fixtures and configuration for the RAG framework test suite."""

import pytest


# ── Marker registration ────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "unit: fast, isolated unit tests")
    config.addinivalue_line("markers", "integration: tests that touch real infrastructure (db, disk)")
    config.addinivalue_line("markers", "e2e: end-to-end pipeline tests")


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_retrieval_config_env(monkeypatch):
    """Prevent a developer's real ``.env`` from leaking retrieval-tier
    config across test files for the rest of the pytest process.

    附录 CI.3/CH.2: ``RAG_EMBEDDING_PROVIDER_DEFAULT``/
    ``RAG_RERANKER_PROVIDER_DEFAULT`` used to sit in ``.env`` under dead
    names (``EMBEDDING_PROVIDER``/``RERANK_PROVIDER``) that no code ever
    read, so ``load_dotenv()`` picking up ``.env`` was a no-op for them.
    Now that the names are fixed and these variables are actually wired up
    (CH.2's fix), several tests (``tests/test_cli_ingest.py``, ``tests/
    test_ingest_hardening.py``, ``tests/e2e/test_main_path.py``) exercise
    ``rag.cli.ingest``'s ``load_dotenv(override=False)`` call, which writes
    straight into ``os.environ`` with no teardown — so a real repo-local
    ``.env`` with ``RAG_EMBEDDING_PROVIDER_DEFAULT=openai`` silently
    flipped every *other* test module's assumed BM25-only
    ``"server_default"`` for the remainder of the same pytest run,
    depending on file-collection order. Individual tests that want a
    specific value already set it explicitly via ``monkeypatch`` (see
    ``tests/test_retrieval_config.py``); this fixture just guarantees every
    test starts from a clean slate instead of whatever an unrelated
    earlier test happened to load from disk.
    """
    for var in (
        "RAG_EMBEDDING_PROVIDER_DEFAULT",
        "RAG_RERANKER_PROVIDER_DEFAULT",
        "RAG_INDEX_COMPAT_MODE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal in-memory config dict suitable for unit tests.

    Returns:
        Dictionary with default settings for all major subsystems.
    """
    return {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dim": 1536,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        "retrieval": {
            "top_k": 5,
            "bm25_top_k": 20,
            "vector_top_k": 20,
        },
        "rerank": {
            "enabled": True,
            "provider": "voyage",
        },
        "storage": {
            "data_dir": "/tmp/rag_test",
        },
    }
