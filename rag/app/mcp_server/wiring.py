"""MCP server wiring — builds pipeline objects from tool input schemas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_ingest_pipeline(
    db_path: str,
    index_dir: str,
    token_budget: int = 512,
    embedding_provider: Optional[str] = "server_default",
    embedding_model: str = "text-embedding-3-small",
    vector_dimension: int = 1536,
    index_mgr=None,
):
    """Instantiate a configured IngestPipeline plus the IndexManager it mutates.

    Args:
        db_path: SQLite database path.
        index_dir: Directory for BM25 and FAISS index files.
        token_budget: Chunk token budget.
        embedding_provider: ``"server_default"``, ``"none"``, an explicit
            provider key, or None (legacy — treated as ``"server_default"``).
            Resolved via ``rag.app.mcp_server.retrieval_config`` — the same
            resolution used by ``retrieve``/``retrieve_with_metadata`` so an
            ingest and a later query never disagree about what "default"
            means (R2's "统一...配置解析" requirement).
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        index_mgr: Optional pre-built ``IndexManager``. Pass one you've
            already entered via ``exclusive_session()`` so the pipeline is
            wired to the freshly-reloaded ``bm25``/``faiss`` objects rather
            than a pre-lock snapshot — required for concurrent-write
            safety. If None, a fresh (unlocked) ``IndexManager`` is
            constructed here, matching the previous behavior.

    Returns:
        ``(pipeline, mgr)`` — the configured ``IngestPipeline`` and the
        ``IndexManager`` whose ``bm25``/``faiss`` objects it was wired with
        (``index_mgr`` itself, if provided). Callers must call ``mgr.save()``
        (or exit the ``exclusive_session()`` that owns it) — constructing a
        *different* fresh ``IndexManager`` to save would reload a stale,
        pre-ingest snapshot instead of the one this pipeline actually
        mutated. Saving also persists the index manifest recorded here.

    Raises:
        IndexConfigMismatchError: This index_dir already holds vectors built
            with a different embedding provider/model/dimension — mixing
            embedding spaces in one FAISS index isn't a legitimate degrade
            path (see ``rag.app.mcp_server.index_validation``).
        EmbeddingProviderUnavailableError: The resolved provider's
            dependency (API key, package) is unavailable.
    """
    from rag.app.mcp_server.index_validation import validate_ingest_embedding_config
    from rag.app.mcp_server.retrieval_config import (
        build_embedding_provider,
        resolve_embedding_config,
    )
    from rag.infra.indexes.index_manager import IndexManager
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)

    # Keyword (BM25) indexing needs no vectors, so the IndexManager is built
    # regardless of embedding_provider — a BM25-only ingest (no embedding
    # provider configured) must still get a keyword index wired in.
    mgr = index_mgr if index_mgr is not None else IndexManager(index_dir=index_dir)

    resolved = resolve_embedding_config(embedding_provider, embedding_model, vector_dimension)
    validate_ingest_embedding_config(mgr, resolved)
    embed_provider = build_embedding_provider(resolved)
    vec_index = mgr.faiss if embed_provider is not None else None
    if embed_provider is not None:
        mgr.record_embedding_config(resolved.provider, resolved.model, resolved.dimension)

    pipeline = IngestPipeline(
        doc_store=doc_store,
        trace_store=trace_store,
        token_budget=token_budget,
        embedding_provider=embed_provider,
        vector_index=vec_index,
        keyword_index=mgr.bm25,
    )
    return pipeline, mgr


def build_query_pipeline(
    db_path: str,
    index_dir: str,
    top_k: int = 10,
    context_top_k: int = 6,
    token_budget: int = 2048,
    embedding_provider: Optional[str] = "server_default",
    embedding_model: str = "text-embedding-3-small",
    vector_dimension: int = 1536,
    llm_model: str = "gpt-4o-mini",
    enable_generation: bool = True,
    reranker_provider: Optional[str] = "server_default",
    reranker_model: str = "",
):
    """Instantiate and return a configured QueryPipeline.

    Args:
        db_path: SQLite database path.
        index_dir: Directory for BM25 and FAISS index files.
        top_k: Candidates to retrieve per index.
        context_top_k: Chunks to pack into the LLM context window.
        token_budget: Token budget for context packing.
        embedding_provider: ``"server_default"``, ``"none"``, an explicit
            provider key, or None (legacy — treated as ``"server_default"``).
            Resolved via ``rag.app.mcp_server.retrieval_config``, including
            the index-compatibility check (R2) — an incompatible or
            unavailable config raises rather than silently degrading.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        llm_model: LLM model identifier for generation.
        enable_generation: Whether to wire the LLM composer.
        reranker_provider: ``"server_default"``, ``"none"``, an explicit
            reranker key, or None (legacy — treated as ``"server_default"``).
        reranker_model: Model identifier for the reranker, or ``""`` to use
            the resolved provider's own default.

    Returns:
        Configured ``QueryPipeline`` instance.

    Raises:
        IndexNotFoundError: Vector search was resolved but no compatible
            index exists (strict mode only — see ``RAG_INDEX_COMPAT_MODE``).
        IndexConfigMismatchError: Vector search was resolved but the index
            was built with a different provider/model/dimension (strict
            mode only).
        EmbeddingProviderUnavailableError: The resolved embedding provider's
            dependency (API key, package) is unavailable.
        RerankerUnavailableError: The resolved reranker's dependency (API
            key, package) is unavailable.
    """
    from rag.app.mcp_server.index_validation import validate_index_compatibility
    from rag.app.mcp_server.retrieval_config import (
        build_embedding_provider,
        build_reranker,
        resolve_embedding_config,
        resolve_reranker_config,
    )
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.infra.indexes.index_manager import IndexManager
    from rag.pipelines.query_pipeline import QueryPipeline

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    trace_store = SQLiteTraceStore(db_path)
    mgr = IndexManager(index_dir=index_dir)

    resolved_embed = resolve_embedding_config(embedding_provider, embedding_model, vector_dimension)
    validation = validate_index_compatibility(mgr, resolved_embed)
    embed_provider = None
    vec_index = None
    if validation.use_vector_search:
        embed_provider = build_embedding_provider(resolved_embed)
        vec_index = mgr.faiss

    composer = None
    if enable_generation:
        from rag.infra.llm.openai_llm_client import OpenAILLMClient
        from rag.infra.generation.answer_composer_basic import BasicAnswerComposer
        llm = OpenAILLMClient(model=llm_model)
        composer = BasicAnswerComposer(
            llm_client=llm,
            top_k=context_top_k,
            token_budget=token_budget,
        )

    resolved_rerank = resolve_reranker_config(reranker_provider, reranker_model)
    reranker = build_reranker(resolved_rerank)

    return QueryPipeline(
        keyword_index=mgr.bm25,
        trace_store=trace_store,
        vector_index=vec_index,
        embedding_provider=embed_provider,
        answer_composer=composer,
        reranker=reranker,
        top_k=top_k,
    )
