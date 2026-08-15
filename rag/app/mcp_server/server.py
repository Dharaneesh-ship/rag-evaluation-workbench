"""MCP server — exposes rag.ingest, rag.query, rag.eval.run, and rag.sync_source tools.

stdout is reserved exclusively for the MCP JSON-RPC protocol stream — any
stray text written to it (a debug ``print()``, a verbose third-party
library) corrupts the protocol from the client's perspective. See
``_harden_stdio()`` below.
"""

from __future__ import annotations

import builtins
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

from mcp.server.fastmcp import FastMCP

from rag.core.errors import InvalidCollectionError, RagMCPError, RetrievalInternalError
from rag.app.mcp_server.schemas import (
    CitationOutput,
    ChunkResult,
    ChunkResultWithMetadata,
    CollectionInfo,
    EvalRunToolInput,
    EvalRunToolOutput,
    IngestToolInput,
    IngestToolOutput,
    ListCollectionsOutput,
    MetricResult,
    QueryToolInput,
    QueryToolOutput,
    RetrieveInput,
    RetrieveOutput,
    RetrieveWithMetadataOutput,
    SyncSourceToolInput,
    SyncSourceToolOutput,
)
from rag.app.mcp_server.wiring import build_ingest_pipeline, build_query_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="rag-framework",
    instructions=(
        "RAG Framework MCP server. Provides tools for document ingestion, "
        "grounded question answering, and retrieval evaluation."
    ),
    # host/port only matter for the "sse"/"streamable-http" transports below —
    # ignored entirely under the default "stdio" transport. MCP_HOST defaults
    # to loopback-only; set to 0.0.0.0 (as docker-compose.yml does) so the
    # container's published port is actually reachable from outside it.
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    # 8002 matches the ecosystem-wide port allocation (rag-framework=8002,
    # data-cleaner=8001) that MAEDA's settings.py assumes by default — see
    # ECOSYSTEM_INTEGRATION_PLAN.md 定案 #1.
    port=int(os.environ.get("MCP_PORT", "8002")),
)


# ---------------------------------------------------------------------------
# rag.ingest
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag.ingest",
    description=(
        "Ingest a document into the RAG corpus. "
        "Parses, cleans, chunks, and optionally embeds the file, "
        "then stores it in the DocStore and indexes."
    ),
)
def rag_ingest(input: IngestToolInput) -> IngestToolOutput:
    """Ingest a single document file.

    Args:
        input: Validated IngestToolInput payload.

    Returns:
        IngestToolOutput with doc_id, chunk_count, elapsed_ms, and error.
    """
    logger.info("rag.ingest called: source_path=%s", input.source_path)
    try:
        from rag.infra.indexes.index_manager import IndexManager

        index_mgr = IndexManager(index_dir=input.index_dir)
        # Locks the load-mutate-save cycle against a concurrent ingest/sync
        # on the same index_dir; reload happens fresh inside the lock, save
        # happens automatically on clean exit.
        with index_mgr.exclusive_session():
            pipeline, _ = build_ingest_pipeline(
                db_path=input.db_path,
                index_dir=input.index_dir,
                token_budget=input.token_budget,
                embedding_provider=input.embedding_provider,
                embedding_model=input.embedding_model,
                vector_dimension=input.vector_dimension,
                index_mgr=index_mgr,
            )
            result = pipeline.ingest(input.source_path, collection=input.collection)

        return IngestToolOutput(
            doc_id=result.doc_id,
            source_path=result.source_path,
            block_count=result.block_count,
            chunk_count=result.chunk_count,
            embed_tokens=result.embed_tokens,
            elapsed_ms=result.elapsed_ms,
            skipped=result.skipped,
            error=result.error,
            run_id=result.run_id,
        )
    except Exception as exc:
        logger.exception("rag.ingest failed: %s", exc)
        return IngestToolOutput(
            doc_id="",
            source_path=input.source_path,
            block_count=0,
            chunk_count=0,
            elapsed_ms=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# rag.query
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag.query",
    description=(
        "Query the RAG corpus with a natural-language question. "
        "Retrieves relevant chunks via BM25 and optional vector search, "
        "applies RRF fusion, and optionally generates a grounded answer."
    ),
)
def rag_query(input: QueryToolInput) -> QueryToolOutput:
    """Answer a question against the ingested corpus.

    Args:
        input: Validated QueryToolInput payload.

    Returns:
        QueryToolOutput with answer text, citations, and generation stats.
    """
    logger.info("rag.query called: query=%r", input.query)
    try:
        pipeline = build_query_pipeline(
            db_path=input.db_path,
            index_dir=input.index_dir,
            top_k=input.top_k,
            context_top_k=input.context_top_k,
            token_budget=input.token_budget,
            embedding_provider=input.embedding_provider,
            embedding_model=input.embedding_model,
            vector_dimension=input.vector_dimension,
            llm_model=input.llm_model,
            enable_generation=input.enable_generation,
        )
        result = pipeline.query(input.query, collection=input.collection)

        # Extract answer fields
        answer_text = ""
        abstained = False
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        generation_latency_ms = 0.0

        if result.answer:
            answer_text = result.answer.text
            abstained = result.answer.abstained

        if result.answer_trace:
            at = result.answer_trace
            prompt_tokens = at.prompt_tokens
            completion_tokens = at.completion_tokens
            total_tokens = at.total_tokens
            generation_latency_ms = at.total_latency_ms

        # Build citation outputs
        citations_out: list[CitationOutput] = []
        source_citations = result.answer.citations if result.answer else result.citations
        for cit in source_citations:
            citations_out.append(
                CitationOutput(
                    ref_number=cit.ref_number,
                    chunk_id=cit.chunk_id,
                    doc_id=cit.doc_id,
                    source_label=cit.source_label,
                    page=cit.page,
                    display_text=cit.display_text,
                )
            )

        return QueryToolOutput(
            query=input.query,
            answer=answer_text,
            citations=citations_out,
            abstained=abstained,
            candidate_count=len(result.candidates),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generation_latency_ms=generation_latency_ms,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
            run_id=result.run_id,
        )
    except Exception as exc:
        logger.exception("rag.query failed: %s", exc)
        return QueryToolOutput(
            query=input.query,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# rag.eval.run
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag.eval.run",
    description=(
        "Run a retrieval evaluation suite against the RAG corpus. "
        "Computes Recall@k, MRR, NDCG@k, and optional generation metrics."
    ),
)
def rag_eval_run(input: EvalRunToolInput) -> EvalRunToolOutput:
    """Run evaluation metrics over a labelled dataset.

    Args:
        input: Validated EvalRunToolInput payload.

    Returns:
        EvalRunToolOutput with per-metric results and elapsed time.

    Note:
        Full evaluation implementation is delivered in Phase 10.
        This stub returns a structured placeholder response.
    """
    logger.info("rag.eval.run called: dataset=%s metrics=%s", input.dataset_path, input.metrics)
    try:
        import json
        import time
        from pathlib import Path

        start = time.monotonic()

        dataset_path = Path(input.dataset_path)
        if not dataset_path.exists():
            return EvalRunToolOutput(
                dataset_path=input.dataset_path,
                error=f"Dataset file not found: {input.dataset_path}",
            )

        # Load dataset records
        records = []
        with dataset_path.open() as f:
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                # JSONL
                records = [json.loads(line) for line in content.splitlines() if line.strip()]

        num_queries = len(records)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Phase 10 will compute real metrics — return placeholder zeros for now
        metric_results = [
            MetricResult(metric=m, value=0.0, num_samples=num_queries)
            for m in input.metrics
        ]

        output = EvalRunToolOutput(
            dataset_path=input.dataset_path,
            metrics=metric_results,
            num_queries=num_queries,
            elapsed_ms=elapsed_ms,
        )

        # Write report if requested
        if input.output_path:
            Path(input.output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(input.output_path).write_text(output.model_dump_json(indent=2))
            output.output_path = input.output_path

        return output

    except Exception as exc:
        logger.exception("rag.eval.run failed: %s", exc)
        return EvalRunToolOutput(
            dataset_path=input.dataset_path,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# rag.sync_source
# ---------------------------------------------------------------------------

#: Registry mapping connector names to their import paths
_CONNECTOR_REGISTRY: dict[str, str] = {
    "email": "rag.infra.connectors.email_connector.EmailConnector",
    "slack": "rag.infra.connectors.slack_connector.SlackConnector",
    "notion": "rag.infra.connectors.notion_connector.NotionConnector",
    "google_docs": "rag.infra.connectors.google_docs_connector.GoogleDocsConnector",
}


def _load_connector(connector_name: str):
    """Dynamically import and instantiate a connector by name."""
    import importlib

    dotted = _CONNECTOR_REGISTRY[connector_name]
    module_path, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


@mcp.tool(
    name="rag.sync_source",
    description=(
        "Sync an external source (email, slack, notion, google_docs) into the RAG corpus. "
        "Fetches new/updated items since the last cursor, ingests them, and persists "
        "the updated cursor for the next run."
    ),
)
def rag_sync_source(input: SyncSourceToolInput) -> SyncSourceToolOutput:
    """Trigger a connector sync and ingest cycle.

    Args:
        input: Validated SyncSourceToolInput payload.

    Returns:
        SyncSourceToolOutput with fetched/ingested/skipped/failed counts and cursor.
    """
    logger.info("rag.sync_source called: connector=%s", input.connector)
    try:
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.connector_sync_pipeline import ConnectorSyncPipeline

        from rag.infra.indexes.index_manager import IndexManager

        connector = _load_connector(input.connector)
        doc_store = SQLiteDocStore(input.db_path)
        trace_store = SQLiteTraceStore(input.db_path)

        index_mgr = IndexManager(index_dir=input.index_dir)
        with index_mgr.exclusive_session():
            ingest_pipeline, _ = build_ingest_pipeline(
                db_path=input.db_path,
                index_dir=input.index_dir,
                token_budget=input.token_budget,
                embedding_provider=input.embedding_provider,
                embedding_model=input.embedding_model,
                vector_dimension=input.vector_dimension,
                index_mgr=index_mgr,
            )
            pipeline = ConnectorSyncPipeline(
                connector=connector,
                ingest_pipeline=ingest_pipeline,
                doc_store=doc_store,
                trace_store=trace_store,
            )
            result = pipeline.run(since_cursor=input.since_cursor)

        return SyncSourceToolOutput(
            connector=input.connector,
            fetched=result.fetched,
            ingested=result.ingested,
            skipped=result.skipped,
            failed=result.failed,
            cursor_before=result.cursor_before,
            cursor_after=result.cursor_after,
            elapsed_ms=result.elapsed_ms,
            run_id=result.run_id,
            error=result.error,
        )

    except Exception as exc:
        logger.exception("rag.sync_source failed: %s", exc)
        return SyncSourceToolOutput(
            connector=input.connector,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


def _known_collections(db_path: str) -> set[str]:
    """Return the set of collection names actually tagged on ingested documents.

    Only explicitly-tagged collections count — an untagged document is only
    reachable by an unscoped (``collection=None``) query (see the collection
    isolation fix in commit 91bf15d), so an empty return here means "nothing
    has been tagged yet," not "no documents exist."
    """
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore

    doc_store = SQLiteDocStore(db_path)
    return {doc["collection"] for doc in doc_store.list_documents() if doc.get("collection")}


@dataclass
class _RetrievalResolution:
    """Bundles a retrieval run's result with what config actually executed.

    ``QueryPipeline.query()`` on its own only reports ``QueryResult`` —
    exactly what mode/provider/reranker actually ran (vs. what was merely
    requested) has to be tracked alongside it so ``retrieve``/
    ``retrieve_with_metadata`` can surface it in their responses and traces
    (TB1a/TB2's "trace 需记录本次实际执行的 retrieval mode、provider、
    model、reranker、降级原因" requirement).
    """

    query_result: "object"
    retrieval_mode: str
    embedding_provider: Optional[str]
    embedding_model: Optional[str]
    reranker_provider: Optional[str]
    degraded_reason: Optional[str]


def _run_retrieval(input: RetrieveInput) -> _RetrievalResolution:
    """Shared retrieval logic used by retrieve and retrieve_with_metadata.

    Raises:
        InvalidCollectionError: ``input.collection`` was given but no
            document was ever tagged with it.
        IndexNotFoundError: Vector search was resolved but no compatible
            index exists (strict mode only).
        IndexConfigMismatchError: Vector search was resolved but the index
            was built with a different provider/model/dimension (strict
            mode only).
        EmbeddingProviderUnavailableError: The resolved embedding provider's
            dependency (API key, package) is unavailable.
        RerankerUnavailableError: The resolved reranker's dependency (API
            key, package) is unavailable.
        RetrievalInternalError: The pipeline itself failed at query time —
            previously this was silently absorbed into an ``error`` field on
            an otherwise-successful-looking empty result (定案 #13).
    """
    from rag.app.mcp_server.index_validation import validate_index_compatibility
    from rag.app.mcp_server.retrieval_config import (
        build_embedding_provider,
        build_reranker,
        resolve_embedding_config,
        resolve_reranker_config,
    )
    from rag.infra.indexes.index_manager import IndexManager
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.query_pipeline import QueryPipeline

    if input.collection:
        known = _known_collections(input.db_path)
        if known and input.collection not in known:
            raise InvalidCollectionError(
                f"Collection '{input.collection}' does not exist. "
                f"Known collections: {sorted(known)}."
            )

    mgr = IndexManager(index_dir=input.index_dir)
    trace_store = SQLiteTraceStore(input.db_path)

    resolved_embed = resolve_embedding_config(
        input.embedding_provider, input.embedding_model, input.vector_dimension
    )
    validation = validate_index_compatibility(mgr, resolved_embed)
    embed_provider = None
    vec_index = None
    if validation.use_vector_search:
        embed_provider = build_embedding_provider(resolved_embed)
        vec_index = mgr.faiss

    resolved_rerank = resolve_reranker_config(input.reranker_provider, input.reranker_model)
    reranker = build_reranker(resolved_rerank)

    pipeline = QueryPipeline(
        keyword_index=mgr.bm25,
        trace_store=trace_store,
        vector_index=vec_index,
        embedding_provider=embed_provider,
        answer_composer=None,   # retrieval only, no LLM
        reranker=reranker,
        top_k=input.top_k,
    )
    result = pipeline.query(input.query, collection=input.collection)

    if result.error:
        # QueryPipeline.query() catches its own internal exceptions into
        # .error and still returns a "successful" QueryResult — left alone,
        # that becomes an internal failure indistinguishable from a
        # legitimate empty result once wrapped in RetrieveOutput(chunks=[]).
        # Raise instead, so it surfaces as a real MCP tool error.
        raise RetrievalInternalError(result.error)

    if vec_index is not None and reranker is not None:
        mode = "hybrid_rerank"
    elif vec_index is not None:
        mode = "hybrid"
    else:
        mode = "bm25_only"

    resolved_provider = resolved_embed.provider if vec_index is not None else None
    resolved_model = resolved_embed.model if vec_index is not None else None

    trace_store.save_run(
        run_type="retrieval_config",
        metadata={
            "run_id": result.run_id,
            "retrieval_mode": mode,
            "embedding_provider": resolved_provider,
            "embedding_model": resolved_model,
            "reranker_provider": resolved_rerank.provider,
            "degraded_reason": validation.degraded_reason,
        },
    )

    return _RetrievalResolution(
        query_result=result,
        retrieval_mode=mode,
        embedding_provider=resolved_provider,
        embedding_model=resolved_model,
        reranker_provider=resolved_rerank.provider,
        degraded_reason=validation.degraded_reason,
    )


@mcp.tool(
    name="retrieve",
    description=(
        "Retrieve the most relevant chunks for a query without generating an answer. "
        "Returns raw chunk content, RRF fusion score, and full metadata dict. "
        "Use this when you want to inspect or post-process the retrieved evidence yourself."
    ),
)
def retrieve(input: RetrieveInput) -> RetrieveOutput:
    """Retrieve ranked chunks for a query.

    Internal failures (missing/incompatible index, unavailable embedding or
    reranker dependency, invalid collection, pipeline error) raise a
    ``RagMCPError`` subclass with a stable error-code prefix, which FastMCP
    surfaces as a real MCP tool error (``isError=True``) — see
    ``rag.core.errors``. Legal zero hits are a normal success with
    ``chunks=[]``.

    Args:
        input: Validated RetrieveInput payload.

    Returns:
        RetrieveOutput with a ranked list of chunks (content, score, metadata)
        plus the retrieval config that actually executed.
    """
    logger.info("retrieve called: query=%r top_k=%d", input.query, input.top_k)
    try:
        resolution = _run_retrieval(input)
    except RagMCPError:
        raise
    except Exception as exc:
        logger.exception("retrieve failed: %s", exc)
        raise RetrievalInternalError(str(exc)) from exc

    result = resolution.query_result
    chunks = []
    for cand in result.candidates[: input.top_k]:
        chunks.append(
            ChunkResult(
                content=cand.display_text,
                score=float(cand.final_score),
                metadata={
                    "chunk_id": cand.chunk_id,
                    "doc_id": cand.doc_id,
                    "source_path": cand.metadata.get("source_path", ""),
                    "page": cand.metadata.get("start_page") or cand.metadata.get("page"),
                    "retrieval_source": cand.source_label,
                    "rrf_score": cand.rrf_score,
                },
            )
        )
    return RetrieveOutput(
        chunks=chunks,
        retrieval_mode=resolution.retrieval_mode,
        embedding_provider=resolution.embedding_provider,
        embedding_model=resolution.embedding_model,
        reranker_provider=resolution.reranker_provider,
        degraded_reason=resolution.degraded_reason,
    )


# ---------------------------------------------------------------------------
# retrieve_with_metadata
# ---------------------------------------------------------------------------


@mcp.tool(
    name="retrieve_with_metadata",
    description=(
        "Retrieve the most relevant chunks for a query with explicit metadata fields. "
        "Returns content, RRF score, source_file, page, and chunk_id as top-level fields "
        "for easy consumption by downstream agents."
    ),
)
def retrieve_with_metadata(input: RetrieveInput) -> RetrieveWithMetadataOutput:
    """Retrieve ranked chunks with explicit metadata fields.

    See ``retrieve`` for the error-handling contract — internal failures
    raise a ``RagMCPError`` subclass instead of being encoded in the response.

    Args:
        input: Validated RetrieveInput payload.

    Returns:
        RetrieveWithMetadataOutput with chunks containing content, score,
        source_file, page, and chunk_id, plus the retrieval config that
        actually executed.
    """
    logger.info("retrieve_with_metadata called: query=%r top_k=%d", input.query, input.top_k)
    try:
        resolution = _run_retrieval(input)
    except RagMCPError:
        raise
    except Exception as exc:
        logger.exception("retrieve_with_metadata failed: %s", exc)
        raise RetrievalInternalError(str(exc)) from exc

    result = resolution.query_result
    chunks = []
    for cand in result.candidates[: input.top_k]:
        source_path = cand.metadata.get("source_path", "") or ""
        source_file = os.path.basename(source_path) or cand.doc_id
        page = cand.metadata.get("start_page") or cand.metadata.get("page")
        chunks.append(
            ChunkResultWithMetadata(
                content=cand.display_text,
                score=float(cand.final_score),
                source_file=source_file,
                page=page,
                chunk_id=cand.chunk_id,
            )
        )
    return RetrieveWithMetadataOutput(
        chunks=chunks,
        retrieval_mode=resolution.retrieval_mode,
        embedding_provider=resolution.embedding_provider,
        embedding_model=resolution.embedding_model,
        reranker_provider=resolution.reranker_provider,
        degraded_reason=resolution.degraded_reason,
    )


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


@mcp.tool(
    name="list_collections",
    description=(
        "List all document collections available in the RAG corpus. "
        "Returns each collection's name, document count, and description."
    ),
)
def list_collections(db_path: str = "data/rag.db") -> ListCollectionsOutput:
    """List available collections in the docstore.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        ListCollectionsOutput with a list of CollectionInfo entries.
    """
    logger.info("list_collections called: db_path=%s", db_path)
    try:
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore

        doc_store = SQLiteDocStore(db_path)
        # Aggregate document counts grouped by collection field
        collections: dict[str, int] = {}
        for doc in doc_store.list_documents():
            col = doc.get("collection") or "default"
            collections[col] = collections.get(col, 0) + 1

        if not collections:
            # Always return at least the default collection
            collections["default"] = 0

        result = [
            CollectionInfo(name=name, doc_count=count)
            for name, count in sorted(collections.items())
        ]
        return ListCollectionsOutput(collections=result)
    except Exception as exc:
        logger.exception("list_collections failed: %s", exc)
        return ListCollectionsOutput(error=str(exc))


# ---------------------------------------------------------------------------
# stdio hardening
# ---------------------------------------------------------------------------


def _harden_stdio() -> None:
    """Make stdout-safety an explicit, structural guarantee, not an accident.

    Without this, stdout only stays clean for the JSON-RPC protocol because
    nothing in this codebase happens to call ``logging.basicConfig()`` with
    a stdout stream, and Python's logging "handler of last resort" defaults
    to stderr. That's correct today but not *guaranteed* — a future
    contributor's debug ``print()``, or a new dependency that logs to
    stdout by default, would silently corrupt the protocol stream with no
    clear signal pointing at the cause. This makes the invariant explicit:

    1. Root logger gets an explicit ``StreamHandler(sys.stderr)`` — turns
       "nobody configured stdout" into "stderr is configured on purpose."
    2. ``print()`` is rebound to always write to stderr, so any print
       anywhere in the imported ``rag`` package (or a dependency) physically
       cannot reach the real stdout for the remainder of the process.

    Call once, before ``mcp.run()`` — not at import time, so importing this
    module for testing (which relies on capturing stdout/stderr normally)
    is unaffected.
    """
    root = logging.getLogger()
    root.handlers = [logging.StreamHandler(sys.stderr)]
    root.setLevel(logging.INFO)

    original_print = builtins.print

    def _stderr_print(*args, **kwargs):
        # Force stderr regardless of any explicit file= the caller passed —
        # overriding rather than pre-binding avoids a "multiple values for
        # file" TypeError if some code does print(..., file=sys.stdout).
        kwargs["file"] = sys.stderr
        return original_print(*args, **kwargs)

    builtins.print = _stderr_print
    logger.info("stdio hardened: logging and print() routed to stderr.")


# ---------------------------------------------------------------------------
# Entry point for running as a standalone MCP server
# ---------------------------------------------------------------------------

def _run() -> None:
    """Start the server with the transport named by MCP_TRANSPORT.

    Defaults to stdio (the right choice when an MCP client spawns this as a
    subprocess — see README §8). docker-compose.yml sets
    MCP_TRANSPORT=streamable-http, since a detached container has no client
    attached to its stdin/stdout to talk to.
    """
    _harden_stdio()
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    _run()
