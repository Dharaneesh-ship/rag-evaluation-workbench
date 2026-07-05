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
    embedding_provider: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    vector_dimension: int = 1536,
):
    """Instantiate a configured IngestPipeline plus the IndexManager it mutates.

    Args:
        db_path: SQLite database path.
        index_dir: Directory for BM25 and FAISS index files.
        token_budget: Chunk token budget.
        embedding_provider: Provider key (``"openai"``) or None to skip embedding.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.

    Returns:
        ``(pipeline, mgr)`` — the configured ``IngestPipeline`` and the
        ``IndexManager`` whose ``bm25``/``faiss`` objects it was wired with.
        Callers must call ``mgr.save()`` on *this* object after ingesting —
        constructing a fresh ``IndexManager`` at save time reloads a stale,
        pre-ingest snapshot from disk instead of the one this pipeline
        actually mutated.
    """
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
    mgr = IndexManager(index_dir=index_dir)

    embed_provider = None
    vec_index = None
    if embedding_provider == "openai":
        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider

        embed_provider = OpenAIEmbeddingProvider(
            model=embedding_model,
            dim=vector_dimension,
        )
        vec_index = mgr.faiss

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
    embedding_provider: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    vector_dimension: int = 1536,
    llm_model: str = "gpt-4o-mini",
    enable_generation: bool = True,
    reranker_provider: Optional[str] = None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    """Instantiate and return a configured QueryPipeline.

    Args:
        db_path: SQLite database path.
        index_dir: Directory for BM25 and FAISS index files.
        top_k: Candidates to retrieve per index.
        context_top_k: Chunks to pack into the LLM context window.
        token_budget: Token budget for context packing.
        embedding_provider: Provider key (``"openai"``) or None.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        llm_model: LLM model identifier for generation.
        enable_generation: Whether to wire the LLM composer.

    Returns:
        Configured ``QueryPipeline`` instance.
    """
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.infra.indexes.index_manager import IndexManager
    from rag.pipelines.query_pipeline import QueryPipeline

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    trace_store = SQLiteTraceStore(db_path)
    mgr = IndexManager(index_dir=index_dir)

    embed_provider = None
    vec_index = None
    if embedding_provider == "openai":
        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
        embed_provider = OpenAIEmbeddingProvider(
            model=embedding_model,
            dimensions=vector_dimension,
        )
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

    reranker = None
    if reranker_provider == "crossencoder":
        from rag.infra.rerank.crossencoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model=reranker_model)
    elif reranker_provider == "voyage":
        from rag.infra.rerank.voyage_rerank import VoyageReranker
        reranker = VoyageReranker(model=reranker_model)

    return QueryPipeline(
        keyword_index=mgr.bm25,
        trace_store=trace_store,
        vector_index=vec_index,
        embedding_provider=embed_provider,
        answer_composer=composer,
        reranker=reranker,
        top_k=top_k,
    )
