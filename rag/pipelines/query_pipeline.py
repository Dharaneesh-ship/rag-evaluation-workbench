"""Query pipeline — retrieval, fusion, optional reranking, and answer generation.

This module is built incrementally across Phase 5 tasks:
  5.1  — attribute_candidates() source attribution merger
  5.2  — RRF fusion (rrf_fusion.py)
  5.3  — Full QueryPipeline (BM25 + vector → fusion → citations)
  5.4+ — Reranking, context packing, LLM generation
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rag.core.contracts.answer import Answer
from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.citation import Citation
from rag.core.contracts.trace import AnswerTrace
from rag.core.interfaces.doc_store import BaseDocStore
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.core.interfaces.keyword_index import BaseKeywordIndex
from rag.core.interfaces.reranker import BaseReranker
from rag.core.interfaces.trace_store import BaseTraceStore
from rag.core.interfaces.vector_index import BaseVectorIndex
from rag.eval.split_guard import assert_not_test_probe_query
from rag.infra.indexes.rrf_fusion import RRFFusion

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _has_cjk(text: str) -> bool:
    """Return True if text contains CJK characters."""
    return bool(_CJK_RE.search(text))


def _language_filter(
    query: str,
    candidates: list[Candidate],
    min_results: int = 1,
) -> list[Candidate]:
    """Remove candidates whose language doesn't match the query language.

    A candidate is dropped when ALL of the following hold:
      1. The query contains CJK characters (Chinese/Japanese/Korean query).
      2. The candidate's text contains NO CJK characters (Latin-script chunk).
      3. The candidate has bm25_score == 0 (no keyword overlap — purely a
         vector-similarity match, i.e. a semantic fluke, not a real hit).

    Rule 3 is intentional: if a Latin chunk somehow scored in BM25 it means
    there was genuine token overlap (e.g. a named entity like "GPT-4" appearing
    in both), so we keep it.

    ``min_results`` ensures we never return an empty list — the best-scoring
    candidates are kept as fallback even if they would otherwise be filtered.

    Args:
        query: Raw query string.
        candidates: RRF-fused candidates in ranked order.
        min_results: Minimum number of candidates to always retain.

    Returns:
        Filtered candidate list, length >= min(min_results, len(candidates)).
    """
    if not _has_cjk(query):
        return candidates  # Non-CJK query — no language filtering needed

    kept = [
        c for c in candidates
        if _has_cjk(c.stable_text or c.display_text or "")
        or (c.bm25_score or 0.0) > 0.0
    ]

    # Safety: never return fewer than min_results
    if len(kept) < min_results:
        kept = candidates[:min_results]

    removed = len(candidates) - len(kept)
    if removed:
        logger.debug(
            "Language filter removed %d non-CJK candidate(s) with zero BM25 score.",
            removed,
        )
    return kept


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Output of a single query pipeline run.

    Attributes:
        query: The original query string.
        candidates: RRF-fused and ranked candidates (top_k).
        citations: Citation objects derived from the top candidates.
        answer: Generated Answer (populated when an answer_composer is wired in).
        answer_trace: AnswerTrace with token counts and per-step latency.
        run_id: TraceStore run identifier for observability.
        elapsed_ms: Total wall-clock time in milliseconds.
        error: Error message if the query failed, or None on success.
    """

    query: str
    candidates: list[Candidate] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    answer: Optional[Answer] = None
    answer_trace: Optional[AnswerTrace] = None
    run_id: str = ""
    elapsed_ms: float = 0.0
    error: Optional[str] = None


def attribute_candidates(
    bm25_results: list[Candidate],
    vector_results: list[Candidate],
) -> list[Candidate]:
    """Merge BM25 and vector candidate lists with source attribution.

    Candidates appearing in only one list are marked ``bm25_only`` or
    ``vector_only``. Candidates appearing in both lists are merged into a
    single ``Candidate`` carrying both scores and marked ``both``.

    The merged list preserves all unique chunk_ids from both inputs.
    Order within each source list is not guaranteed to be preserved — callers
    should apply RRF fusion or another ranking step afterwards.

    Args:
        bm25_results: Candidates returned by the BM25 keyword index.
        vector_results: Candidates returned by the FAISS vector index.

    Returns:
        Merged list of Candidates with ``retrieval_source`` and score fields
        set to reflect which system(s) surfaced each candidate.
    """
    # Index BM25 results by chunk_id
    bm25_by_id: dict[str, Candidate] = {c.chunk_id: c for c in bm25_results}
    # Index vector results by chunk_id
    vec_by_id: dict[str, Candidate] = {c.chunk_id: c for c in vector_results}

    merged: dict[str, Candidate] = {}

    # BM25-only or both
    for chunk_id, bm25_cand in bm25_by_id.items():
        if chunk_id in vec_by_id:
            vec_cand = vec_by_id[chunk_id]
            merged[chunk_id] = bm25_cand.model_copy(
                update={
                    "vector_score": vec_cand.vector_score,
                    "retrieval_source": RetrievalSource.HYBRID,
                }
            )
        else:
            merged[chunk_id] = bm25_cand.model_copy(
                update={"retrieval_source": RetrievalSource.BM25}
            )

    # Vector-only (not already in merged)
    for chunk_id, vec_cand in vec_by_id.items():
        if chunk_id not in merged:
            merged[chunk_id] = vec_cand.model_copy(
                update={"retrieval_source": RetrievalSource.VECTOR}
            )

    result = list(merged.values())
    logger.debug(
        "attribute_candidates: %d bm25, %d vector → %d merged "
        "(%d bm25_only, %d vector_only, %d both)",
        len(bm25_results),
        len(vector_results),
        len(result),
        sum(1 for c in result if c.source_label == "bm25_only"),
        sum(1 for c in result if c.source_label == "vector_only"),
        sum(1 for c in result if c.source_label == "both"),
    )
    return result


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------


def _build_citations(candidates: list[Candidate]) -> list[Citation]:
    """Build a Citation list from the top-ranked candidates.

    Args:
        candidates: RRF-fused candidates in ranked order.

    Returns:
        Citations with 1-based ref_number, one per candidate.
    """
    citations: list[Citation] = []
    for i, cand in enumerate(candidates, start=1):
        page = cand.metadata.get("start_page") or cand.metadata.get("page")
        source_path = cand.metadata.get("source_path", cand.doc_id)
        source_filename = Path(source_path).name if source_path else cand.doc_id
        source_label = source_filename
        if page:
            source_label = f"{source_filename} — page {page}"
        citations.append(
            Citation(
                ref_number=i,
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                source_label=source_label,
                page=page,
                display_text=cand.display_text[:200],
            )
        )
    return citations


# ---------------------------------------------------------------------------
# QueryPipeline
# ---------------------------------------------------------------------------


class QueryPipeline:
    """Hybrid retrieval query pipeline — BM25 + vector → RRF → citations.

    Runs BM25 and (optionally) vector search, merges results with source
    attribution, fuses using RRF, optionally reranks, and optionally generates
    a grounded answer via an LLM composer.

    Query run metadata — including pre- and post-rerank candidate order and
    generation token usage — is recorded in the TraceStore for observability.

    Usage::

        pipeline = QueryPipeline(
            keyword_index=bm25,
            vector_index=faiss,
            embedding_provider=provider,
            trace_store=trace_store,
            reranker=NoOpReranker(),
            answer_composer=BasicAnswerComposer(llm_client=client),
        )
        result = pipeline.query("What is retrieval augmented generation?")

    Args:
        keyword_index: BM25 or other keyword index.
        trace_store: TraceStore for recording query runs.
        vector_index: FAISS or other vector index. Optional — if omitted,
            only BM25 retrieval is performed.
        embedding_provider: Used to embed the query for vector search.
            Required when ``vector_index`` is provided.
        reranker: Optional reranker applied after RRF fusion. When None,
            no reranking is performed and ``final_score = rrf_score``.
        answer_composer: Optional BasicAnswerComposer (or compatible) instance.
            When provided, the pipeline calls ``compose()`` after reranking and
            attaches the Answer + AnswerTrace to the QueryResult.
        top_k: Number of candidates to retrieve per index. Defaults to 10.
        rrf_k: Smoothing constant for RRF fusion. Defaults to 60.
    """

    def __init__(
        self,
        keyword_index: BaseKeywordIndex,
        trace_store: BaseTraceStore,
        vector_index: Optional[BaseVectorIndex] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        reranker: Optional[BaseReranker] = None,
        answer_composer=None,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._keyword_index = keyword_index
        self._vector_index = vector_index
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._answer_composer = answer_composer
        self._trace_store = trace_store
        self._top_k = top_k
        self._fusion = RRFFusion(k=rrf_k)

    def query(self, query: str, collection: Optional[str] = None) -> QueryResult:
        """Run the full hybrid retrieval pipeline for a query string.

        Args:
            query: Natural-language query from the user.
            collection: Optional collection name to scope retrieval.
                When provided, only candidates whose ``metadata["collection"]``
                matches this value are returned. Useful for separating corpora
                (e.g. ``"resumes"`` vs ``"knowledge_base"``).

        Returns:
            QueryResult with ranked candidates, citations, and run metadata.

        Raises:
            TestSplitRevealError: If ``query`` is one of the frozen TB5
                test-split probe queries and this process has not declared a
                reveal. 附录 CP.4: the split guard lives here, at the one
                funnel every retrieval in this repository passes through
                (CLI eval, the MCP retrieve tools, rag.cli.query, and any
                ad-hoc probe script that imports any of them), precisely
                because a guard mounted on a single entry point is the thing
                that failed in 附录 CO.0. Yes, this couples the pipeline to an
                eval fixture; the check is a hash-set lookup, degrades to a
                no-op wherever that fixture is not shipped, and buys an
                invariant no entry-point-local check can.
        """
        assert_not_test_probe_query(query, context="QueryPipeline.query")

        start = time.monotonic()
        run_id = self._trace_store.save_run(
            run_type="query",
            metadata={"query": query, "top_k": self._top_k, "collection": collection},
        )

        try:
            result = self._run(query, run_id, collection=collection)
        except Exception as exc:
            logger.exception("Query failed for '%s': %s", query, exc)
            result = QueryResult(query=query, run_id=run_id, error=str(exc))

        result.elapsed_ms = (time.monotonic() - start) * 1000
        self._trace_store.save_run(
            run_type="query_complete",
            metadata={
                "query": query,
                "run_id": run_id,
                "candidate_count": len(result.candidates),
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
            },
        )
        return result

    def _run(self, query: str, run_id: str, collection: Optional[str] = None) -> QueryResult:
        """Internal query execution.

        Args:
            query: Raw query string.
            run_id: TraceStore run identifier.
            collection: Optional collection filter applied after fusion.

        Returns:
            Populated QueryResult on success.
        """
        # 1. BM25 retrieval (optionally pre-filtered by collection)
        bm25_results = self._keyword_index.search(
            query, top_k=self._top_k, collection=collection
        )
        logger.debug("BM25 returned %d candidates.", len(bm25_results))

        # 2. Vector retrieval (optional, with same collection filter as BM25)
        vector_results: list[Candidate] = []
        if self._vector_index is not None and self._embedding_provider is not None:
            query_vec = self._embedding_provider.embed([query])[0]
            vector_results = self._vector_index.search(
                query_vec, top_k=self._top_k, collection=collection
            )
            logger.debug("Vector search returned %d candidates.", len(vector_results))

        # 3. Source attribution merge
        attributed = attribute_candidates(bm25_results, vector_results)

        # 4. RRF fusion — feed original ranked lists for correct rank positions
        ranked_lists = [l for l in [bm25_results, vector_results] if l]
        if ranked_lists:
            fused = self._fusion.fuse(ranked_lists)[: self._top_k]
        else:
            fused = []

        # 4b. Language filter — remove cross-language candidates that have no
        # keyword overlap (zero BM25) and don't share the query's script.
        fused = _language_filter(query, fused, min_results=1)

        # 5. Record pre-rerank order in trace
        pre_rerank_ids = [c.chunk_id for c in fused]
        self._trace_store.save_run(
            run_type="query_pre_rerank",
            metadata={
                "run_id": run_id,
                "query": query,
                "pre_rerank_chunk_ids": pre_rerank_ids,
                "candidate_count": len(fused),
            },
        )

        # 6. Optional rerank stage
        final_candidates = fused
        if self._reranker is not None and fused:
            final_candidates = self._reranker.rerank(query, fused, top_k=self._top_k)
            post_rerank_ids = [c.chunk_id for c in final_candidates]
            self._trace_store.save_run(
                run_type="query_post_rerank",
                metadata={
                    "run_id": run_id,
                    "query": query,
                    "post_rerank_chunk_ids": post_rerank_ids,
                    "reranker": type(self._reranker).__name__,
                },
            )
            logger.debug(
                "Reranked %d → %d candidates via %s.",
                len(fused),
                len(final_candidates),
                type(self._reranker).__name__,
            )

        # 7. Build citations
        citations = _build_citations(final_candidates)

        # 8. Optional generation stage
        answer: Optional[Answer] = None
        answer_trace: Optional[AnswerTrace] = None
        if self._answer_composer is not None and final_candidates:
            reranker_name = (
                type(self._reranker).__name__ if self._reranker is not None else None
            )
            answer, answer_trace = self._answer_composer.compose(
                query,
                final_candidates,
                run_id=run_id,
                rerank_provider=reranker_name,
                candidates_before_rerank=len(fused),
            )
            # Store generation stats in trace
            self._trace_store.save_run(
                run_type="query_generation",
                metadata={
                    "run_id": run_id,
                    "query": query,
                    "model": answer_trace.model,
                    "prompt_tokens": answer_trace.prompt_tokens,
                    "completion_tokens": answer_trace.completion_tokens,
                    "total_tokens": answer_trace.total_tokens,
                    "context_chunks_used": answer_trace.context_chunks_used,
                    "abstained": answer.abstained,
                    "generation_latency_ms": answer_trace.total_latency_ms,
                },
            )
            # Override citations from composer (filtered to inline-referenced only)
            if answer.citations:
                citations = answer.citations

        logger.info(
            "Query '%s': %d candidates after fusion+rerank (run=%s)",
            query,
            len(final_candidates),
            run_id,
        )

        return QueryResult(
            query=query,
            candidates=final_candidates,
            citations=citations,
            answer=answer,
            answer_trace=answer_trace,
            run_id=run_id,
        )
