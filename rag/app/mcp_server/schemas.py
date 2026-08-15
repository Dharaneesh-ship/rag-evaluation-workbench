"""MCP tool schemas — Pydantic v2 models for rag.ingest, rag.query, rag.eval.run, rag.sync_source,
retrieve, retrieve_with_metadata, list_collections."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Shared retrieval config enums (定案 #7)
#
# provider and model are independent knobs: "server_default" defers to the
# service's configured default (see rag.app.mcp_server.retrieval_config),
# "none" explicitly disables that stage, and any other literal is an
# explicit request that overrides the server default. Old clients that omit
# the field, or send an explicit ``null``, are normalized to
# "server_default" below rather than rejected.
# ---------------------------------------------------------------------------

EmbeddingProviderEnum = Literal["server_default", "none", "openai"]
RerankerProviderEnum = Literal["server_default", "none", "crossencoder", "voyage"]


def _server_default_if_none(v: Any) -> Any:
    """Normalize an explicit ``null``/omitted enum value to ``"server_default"``."""
    return v if v is not None else "server_default"


# ---------------------------------------------------------------------------
# rag.ingest
# ---------------------------------------------------------------------------


class IngestToolInput(BaseModel):
    """Input schema for the ``rag.ingest`` MCP tool.

    Attributes:
        source_path: Absolute or relative path to the file to ingest.
        collection: Target collection name. Defaults to ``"default"``.
        token_budget: Maximum tokens per chunk. Defaults to 512.
        embedding_provider: ``"server_default"`` (use the service's
            configured default), ``"none"`` (BM25 only), or an explicit
            provider key (e.g. ``"openai"``). Defaults to ``"server_default"``.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        db_path: Path to the SQLite database file.
        index_dir: Directory for BM25 and FAISS index files.
    """

    source_path: str = Field(..., min_length=1, description="Path to the file to ingest")
    collection: str = Field(default="default", min_length=1)
    token_budget: int = Field(default=512, ge=64, le=4096)
    embedding_provider: EmbeddingProviderEnum = Field(default="server_default")
    embedding_model: str = Field(default="text-embedding-3-small")
    vector_dimension: int = Field(default=1536, ge=1, le=8192)
    db_path: str = Field(default="data/rag.db")
    index_dir: str = Field(default="data/indexes")

    _normalize_embedding_provider = field_validator("embedding_provider", mode="before")(
        _server_default_if_none
    )

    @field_validator("source_path")
    @classmethod
    def source_path_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_path must not be blank")
        return v


class IngestToolOutput(BaseModel):
    """Output schema for the ``rag.ingest`` MCP tool.

    Attributes:
        doc_id: Identifier of the stored document.
        source_path: Absolute path that was ingested.
        block_count: Number of TextBlocks produced.
        chunk_count: Number of Chunks stored.
        embed_tokens: Token count used for embedding, or 0 if skipped.
        elapsed_ms: Total pipeline wall-clock time in milliseconds.
        skipped: True if the document was already up-to-date.
        error: Error message if ingestion failed, or None on success.
        run_id: TraceStore run identifier.
    """

    doc_id: str
    source_path: str
    block_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embed_tokens: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(ge=0.0)
    skipped: bool = False
    error: Optional[str] = None
    run_id: str = ""


# ---------------------------------------------------------------------------
# rag.query
# ---------------------------------------------------------------------------


class QueryToolInput(BaseModel):
    """Input schema for the ``rag.query`` MCP tool.

    Attributes:
        query: Natural-language question to answer.
        top_k: Number of candidates to retrieve per index. Defaults to 10.
        context_top_k: Chunks to pack into the LLM context window. Defaults to 6.
        token_budget: Token budget for context packing. Defaults to 2048.
        collection: Collection to query against. Defaults to ``"default"``.
        embedding_provider: ``"server_default"``, ``"none"``, or an explicit
            provider key. Defaults to ``"server_default"``.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        reranker_provider: ``"server_default"``, ``"none"``, or an explicit
            reranker key (``"crossencoder"``/``"voyage"``). Defaults to
            ``"server_default"``.
        reranker_model: Model identifier for the reranker, or ``""`` to use
            the resolved provider's own default model.
        llm_model: LLM model identifier for answer generation.
        enable_generation: Whether to run LLM generation. Defaults to True.
        db_path: Path to the SQLite database file.
        index_dir: Directory for BM25 and FAISS index files.
    """

    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=10, ge=1, le=100)
    context_top_k: int = Field(default=6, ge=1, le=20)
    token_budget: int = Field(default=2048, ge=64, le=8192)
    collection: str = Field(default="default", min_length=1)
    embedding_provider: EmbeddingProviderEnum = Field(default="server_default")
    embedding_model: str = Field(default="text-embedding-3-small")
    vector_dimension: int = Field(default=1536, ge=1, le=8192)
    reranker_provider: RerankerProviderEnum = Field(default="server_default")
    reranker_model: str = Field(default="")
    llm_model: str = Field(default="gpt-4o-mini")
    enable_generation: bool = Field(default=True)
    db_path: str = Field(default="data/rag.db")
    index_dir: str = Field(default="data/indexes")

    _normalize_embedding_provider = field_validator("embedding_provider", mode="before")(
        _server_default_if_none
    )
    _normalize_reranker_provider = field_validator("reranker_provider", mode="before")(
        _server_default_if_none
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class CitationOutput(BaseModel):
    """A single source citation in the query response.

    Attributes:
        ref_number: 1-based inline reference number, e.g. ``[1]``.
        chunk_id: Identifier of the supporting chunk.
        doc_id: Parent document identifier.
        source_label: Human-readable source description.
        page: Page number within the source document, if applicable.
        display_text: Short excerpt from the chunk.
    """

    ref_number: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    source_label: str
    page: Optional[int] = None
    display_text: str = ""


class QueryToolOutput(BaseModel):
    """Output schema for the ``rag.query`` MCP tool.

    Attributes:
        query: The original query string.
        answer: Generated answer text (empty string when generation is disabled).
        citations: Ordered list of source citations.
        abstained: True when the model declined to answer.
        candidate_count: Number of candidates retrieved and ranked.
        prompt_tokens: LLM prompt token count (0 if generation disabled).
        completion_tokens: LLM completion token count.
        total_tokens: Total LLM token count.
        generation_latency_ms: Generation wall-clock time in ms.
        elapsed_ms: Total query pipeline wall-clock time in ms.
        error: Error message if the query failed, or None on success.
        run_id: TraceStore run identifier.
    """

    query: str
    answer: str = ""
    citations: list[CitationOutput] = Field(default_factory=list)
    abstained: bool = False
    candidate_count: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    generation_latency_ms: float = Field(default=0.0, ge=0.0)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None
    run_id: str = ""


# ---------------------------------------------------------------------------
# rag.eval.run
# ---------------------------------------------------------------------------


class EvalRunToolInput(BaseModel):
    """Input schema for the ``rag.eval.run`` MCP tool.

    Attributes:
        dataset_path: Path to the evaluation dataset file (JSON or JSONL).
            Each record must contain ``query`` and ``expected_chunks`` (list of
            chunk_ids) and optionally ``expected_answer``.
        metrics: List of metric names to compute. Valid values:
            ``recall_at_k``, ``mrr``, ``ndcg_at_k``,
            ``faithfulness``, ``answer_relevance``.
        top_k: Retrieval top-k used during evaluation. Defaults to 10.
        collection: Collection to evaluate against. Defaults to ``"default"``.
        db_path: Path to the SQLite database file.
        index_dir: Directory for BM25 and FAISS index files.
        output_path: Optional path to write the evaluation report JSON.
    """

    dataset_path: str = Field(..., min_length=1)
    metrics: list[str] = Field(
        default_factory=lambda: ["recall_at_k", "mrr"],
        min_length=1,
    )
    top_k: int = Field(default=10, ge=1, le=100)
    collection: str = Field(default="default")
    db_path: str = Field(default="data/rag.db")
    index_dir: str = Field(default="data/indexes")
    output_path: Optional[str] = None

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        valid = {"recall_at_k", "mrr", "ndcg_at_k", "faithfulness", "answer_relevance"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(
                f"Unknown metric(s): {invalid}. Valid options: {valid}"
            )
        return v


class MetricResult(BaseModel):
    """Result for a single evaluation metric.

    Attributes:
        metric: Metric name.
        value: Computed metric value (typically in [0, 1]).
        num_samples: Number of samples used to compute this metric.
    """

    metric: str
    value: float = Field(ge=0.0)
    num_samples: int = Field(ge=0)


class EvalRunToolOutput(BaseModel):
    """Output schema for the ``rag.eval.run`` MCP tool.

    Attributes:
        dataset_path: Path to the evaluation dataset that was used.
        metrics: List of computed metric results.
        num_queries: Total number of queries evaluated.
        elapsed_ms: Total evaluation wall-clock time in milliseconds.
        output_path: Path where the evaluation report was written, if any.
        error: Error message if evaluation failed, or None on success.
    """

    dataset_path: str
    metrics: list[MetricResult] = Field(default_factory=list)
    num_queries: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    output_path: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# rag.sync_source
# ---------------------------------------------------------------------------


class SyncSourceToolInput(BaseModel):
    """Input schema for the ``rag.sync_source`` MCP tool.

    Attributes:
        connector: Connector name to sync.  Must match one of the registered
            ``connector_name`` values: ``"email"``, ``"slack"``, ``"notion"``,
            ``"google_docs"``.
        since_cursor: Optional override cursor.  If omitted, the last persisted
            cursor is loaded from the DocStore.
        db_path: Path to the SQLite database file.
        index_dir: Directory for BM25 and FAISS index files.
        token_budget: Maximum tokens per ingested chunk.
        embedding_provider: ``"server_default"``, ``"none"``, or an explicit
            provider key. Defaults to ``"server_default"``.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
    """

    connector: str = Field(
        ...,
        description="Connector name: email | slack | notion | google_docs",
    )
    since_cursor: Optional[str] = Field(
        default=None,
        description="Optional cursor override (ISO timestamp or UID). "
        "Defaults to the last persisted cursor.",
    )
    db_path: str = Field(default="data/rag.db")
    index_dir: str = Field(default="data/indexes")
    token_budget: int = Field(default=512, ge=64, le=4096)
    embedding_provider: EmbeddingProviderEnum = Field(default="server_default")
    embedding_model: str = Field(default="text-embedding-3-small")
    vector_dimension: int = Field(default=1536, ge=1, le=8192)

    _normalize_embedding_provider = field_validator("embedding_provider", mode="before")(
        _server_default_if_none
    )

    @field_validator("connector")
    @classmethod
    def validate_connector(cls, v: str) -> str:
        valid = {"email", "slack", "notion", "google_docs"}
        if v not in valid:
            raise ValueError(f"Unknown connector '{v}'. Valid options: {valid}")
        return v


class SyncSourceToolOutput(BaseModel):
    """Output schema for the ``rag.sync_source`` MCP tool.

    Attributes:
        connector: Connector name that was synced.
        fetched: Total artifacts returned by the connector.
        ingested: Artifacts successfully ingested.
        skipped: Artifacts skipped (no content / already up-to-date).
        failed: Artifacts that failed during ingest.
        cursor_before: Cursor at the start of the sync run.
        cursor_after: New cursor persisted after the sync run.
        elapsed_ms: Total wall-clock time in milliseconds.
        run_id: TraceStore run identifier.
        error: Error message if the sync failed, or None on success.
    """

    connector: str
    fetched: int = Field(default=0, ge=0)
    ingested: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cursor_before: str = ""
    cursor_after: str = ""
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    run_id: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# retrieve / retrieve_with_metadata
# ---------------------------------------------------------------------------


class RetrieveInput(BaseModel):
    """Input schema for the ``retrieve`` and ``retrieve_with_metadata`` MCP tools.

    Attributes:
        query: Natural-language search query.
        top_k: Maximum number of chunks to return. Defaults to 5.
        collection: Optional collection name to scope retrieval, e.g. ``"resumes"``
            or ``"knowledge_base"``. When omitted, all collections are searched.
        db_path: Path to the SQLite database file.
        index_dir: Directory for BM25 and FAISS index files.
        embedding_provider: ``"server_default"``, ``"none"``, or an explicit
            provider key. Defaults to ``"server_default"``.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        reranker_provider: ``"server_default"``, ``"none"``, or an explicit
            reranker key (``"crossencoder"``/``"voyage"``). Defaults to
            ``"server_default"``.
        reranker_model: Model identifier for the reranker, or ``""`` to use
            the resolved provider's own default model.
    """

    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of chunks to return")
    collection: Optional[str] = Field(
        default=None,
        description="Scope retrieval to a collection, e.g. 'resumes' or 'knowledge_base'",
    )
    db_path: str = Field(default="data/rag.db")
    index_dir: str = Field(default="data/indexes")
    embedding_provider: EmbeddingProviderEnum = Field(default="server_default")
    embedding_model: str = Field(default="text-embedding-3-small")
    vector_dimension: int = Field(default=1536, ge=1, le=8192)
    reranker_provider: RerankerProviderEnum = Field(default="server_default")
    reranker_model: str = Field(default="")

    _normalize_embedding_provider = field_validator("embedding_provider", mode="before")(
        _server_default_if_none
    )
    _normalize_reranker_provider = field_validator("reranker_provider", mode="before")(
        _server_default_if_none
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class ChunkResult(BaseModel):
    """A single retrieved chunk for the ``retrieve`` tool.

    Attributes:
        content: Raw text content of the chunk.
        score: RRF fusion score (higher = more relevant).
        metadata: Arbitrary chunk metadata dict (doc_id, source_path, page, etc.).
    """

    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveOutput(BaseModel):
    """Output schema for the ``retrieve`` MCP tool.

    Internal failures are raised as real MCP tool errors (``isError=True``
    with a stable error-code prefix — see ``rag.core.errors``), not encoded
    in this schema. A ``RetrieveOutput`` is only ever returned on success,
    including the legal-zero-hits case (``chunks=[]``).

    Attributes:
        chunks: Ranked list of retrieved chunks.
        retrieval_mode: What actually ran: ``"bm25_only"``, ``"hybrid"``, or
            ``"hybrid_rerank"`` — the resolved config, not just "enabled".
        embedding_provider: Resolved embedding provider, or None if BM25-only.
        embedding_model: Resolved embedding model, or None if BM25-only.
        reranker_provider: Resolved reranker provider, or None if no rerank.
        degraded_reason: Set only when an incompatible/unavailable vector
            config was silently downgraded to BM25-only under
            ``RAG_INDEX_COMPAT_MODE=degraded``. None when strict (the
            default) or when nothing needed to degrade.
        error: Deprecated migration-period field, retained for MAEDA's dual-
            track check (定案 #13) — always None; failures now raise instead.
    """

    chunks: list[ChunkResult] = Field(default_factory=list)
    retrieval_mode: str = ""
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_provider: Optional[str] = None
    degraded_reason: Optional[str] = None
    error: Optional[str] = None


class ChunkResultWithMetadata(BaseModel):
    """A single retrieved chunk for the ``retrieve_with_metadata`` tool.

    Attributes:
        content: Raw text content of the chunk.
        score: RRF fusion score (higher = more relevant).
        source_file: Original file name or URL of the source document.
        page: Page number within the source document, if applicable.
        chunk_id: Unique identifier of this chunk.
    """

    content: str
    score: float
    source_file: str
    page: Optional[int] = None
    chunk_id: str


class RetrieveWithMetadataOutput(BaseModel):
    """Output schema for the ``retrieve_with_metadata`` MCP tool.

    See ``RetrieveOutput`` for the error-handling contract — internal
    failures raise real MCP tool errors instead of being encoded here.

    Attributes:
        chunks: Ranked list of retrieved chunks with explicit metadata fields.
        retrieval_mode: What actually ran: ``"bm25_only"``, ``"hybrid"``, or
            ``"hybrid_rerank"``.
        embedding_provider: Resolved embedding provider, or None if BM25-only.
        embedding_model: Resolved embedding model, or None if BM25-only.
        reranker_provider: Resolved reranker provider, or None if no rerank.
        degraded_reason: Set only when an incompatible/unavailable vector
            config was silently downgraded to BM25-only under
            ``RAG_INDEX_COMPAT_MODE=degraded``.
        error: Deprecated migration-period field — always None; failures
            now raise instead.
    """

    chunks: list[ChunkResultWithMetadata] = Field(default_factory=list)
    retrieval_mode: str = ""
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_provider: Optional[str] = None
    degraded_reason: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


class CollectionInfo(BaseModel):
    """Metadata for a single collection.

    Attributes:
        name: Collection name.
        doc_count: Number of documents in this collection.
        description: Human-readable description or empty string.
    """

    name: str
    doc_count: int = Field(ge=0)
    description: str = ""


class ListCollectionsOutput(BaseModel):
    """Output schema for the ``list_collections`` MCP tool.

    Attributes:
        collections: List of available collections with their metadata.
        error: Error message if the call failed, or None on success.
    """

    collections: list[CollectionInfo] = Field(default_factory=list)
    error: Optional[str] = None
