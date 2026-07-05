"""LLM-based batch metadata enrichment for chunks — Task 12.3.

Enriches chunks with AI-generated ``title``, ``summary``, and ``tags``
using the ``BaseLLMClient`` abstraction.  Chunks are processed in batches
to respect token-rate limits and to amortise LLM call overhead.

Fallback behaviour
------------------
If the LLM call fails for any batch, the enricher logs the error and falls
back to ``RulesMetadataEnricher`` for that batch so callers always receive
metadata even when the LLM is unavailable.

Token tracing
-------------
Total ``prompt_tokens`` and ``completion_tokens`` consumed are accumulated
in ``LLMEnrichmentSummary`` and can be forwarded to the TraceStore by the
caller.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from rag.core.contracts.chunk import Chunk
from rag.core.interfaces.llm_client import BaseLLMClient
from rag.infra.chunking.metadata_enricher_rules import RulesMetadataEnricher

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a metadata extraction assistant. For each numbered chunk below, \
return a JSON array where each element has exactly three fields:
  "title"   : a concise title (≤ 12 words)
  "summary" : one sentence summary (≤ 40 words)
  "tags"    : list of 1–5 keyword tags (lowercase, no spaces — use underscores)

Return ONLY the JSON array, no markdown fences, no extra text.
"""

_USER_TEMPLATE = "Chunks:\n{chunks_text}\n\nReturn JSON array:"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMEnrichmentResult:
    """Per-chunk LLM enrichment result.

    Attributes:
        chunk_id: Identifier of the enriched chunk.
        title: LLM-generated title (or rules fallback).
        summary: LLM-generated summary (or rules fallback).
        tags: LLM-generated tags (or rules fallback).
        used_fallback: True if the rules-based fallback was used.
    """

    chunk_id: str
    title: str
    summary: str
    tags: list[str] = field(default_factory=list)
    used_fallback: bool = False


@dataclass
class LLMEnrichmentSummary:
    """Aggregate stats for a full enrichment run.

    Attributes:
        results: Per-chunk results.
        prompt_tokens: Total prompt tokens consumed.
        completion_tokens: Total completion tokens consumed.
        total_tokens: Sum of prompt + completion tokens.
        fallback_batches: Number of batches that fell back to rules.
    """

    results: list[LLMEnrichmentResult] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    fallback_batches: int = 0

    def add_usage(self, prompt: int, completion: int) -> None:
        """Accumulate token usage from one LLM response."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------


class LLMBatchMetadataEnricher:
    """Batch-enrich chunk metadata using an LLM.

    Args:
        llm_client: Any ``BaseLLMClient`` implementation.
        batch_size: Number of chunks per LLM call.  Defaults to 8.
        annotate: If True (default), write results into ``chunk.metadata``.
        overwrite: If False (default), skip fields already present.
        fallback_enricher: Rules-based enricher used when the LLM fails.
            Created automatically if not provided.

    Example::

        client = OpenAILLMClient(model="gpt-4o-mini")
        enricher = LLMBatchMetadataEnricher(client, batch_size=5)
        summary = enricher.enrich_chunks(chunks)
        print(f"Used {summary.total_tokens} tokens")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        batch_size: int = 8,
        annotate: bool = True,
        overwrite: bool = False,
        fallback_enricher: RulesMetadataEnricher | None = None,
    ) -> None:
        self._llm = llm_client
        self._batch_size = max(1, batch_size)
        self._annotate = annotate
        self._overwrite = overwrite
        self._fallback = fallback_enricher or RulesMetadataEnricher(
            annotate=annotate, overwrite=overwrite
        )

    def _build_prompt(self, chunks: list[Chunk]) -> str:
        """Build the user-turn prompt for a batch of chunks."""
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.stable_text[:400].replace("\n", " ")
            lines.append(f"[{i}] {text}")
        return _USER_TEMPLATE.format(chunks_text="\n".join(lines))

    def _parse_response(
        self, raw: str, chunks: list[Chunk]
    ) -> list[LLMEnrichmentResult]:
        """Parse the LLM JSON response into per-chunk results."""
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        results: list[LLMEnrichmentResult] = []
        for i, chunk in enumerate(chunks):
            item = items[i] if i < len(items) else {}
            results.append(
                LLMEnrichmentResult(
                    chunk_id=chunk.chunk_id or chunk.chunk_signature,
                    title=str(item.get("title", "")).strip(),
                    summary=str(item.get("summary", "")).strip(),
                    tags=[str(t).strip() for t in item.get("tags", []) if t],
                )
            )
        return results

    def _apply_to_metadata(
        self, chunk: Chunk, result: LLMEnrichmentResult
    ) -> None:
        """Write result fields into chunk.metadata respecting overwrite flag."""
        if self._overwrite or "title" not in chunk.metadata:
            chunk.metadata["title"] = result.title
        if self._overwrite or "summary" not in chunk.metadata:
            chunk.metadata["summary"] = result.summary
        if self._overwrite or "tags" not in chunk.metadata:
            chunk.metadata["tags"] = result.tags

    def _enrich_batch(
        self, chunks: list[Chunk], summary: LLMEnrichmentSummary
    ) -> list[LLMEnrichmentResult]:
        """Enrich a single batch; falls back to rules on LLM failure."""
        user_prompt = self._build_prompt(chunks)
        # BaseLLMClient.generate() takes a single prompt string, not a
        # chat-style messages list — concatenate system instructions and
        # the per-batch content into one prompt to match the real interface.
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
        try:
            response = self._llm.generate(full_prompt)
            summary.add_usage(response.prompt_tokens, response.completion_tokens)
            results = self._parse_response(response.text, chunks)

            if self._annotate:
                for chunk, result in zip(chunks, results):
                    self._apply_to_metadata(chunk, result)

            return results

        except Exception as exc:
            logger.warning(
                "LLM enrichment failed for batch of %d chunks (%s) — "
                "falling back to rules",
                len(chunks),
                exc,
            )
            summary.fallback_batches += 1
            fallback_results = self._fallback.enrich_chunks(chunks)
            return [
                LLMEnrichmentResult(
                    chunk_id=r.chunk_id,
                    title=r.title,
                    summary=r.summary,
                    tags=r.tags,
                    used_fallback=True,
                )
                for r in fallback_results
            ]

    def enrich_chunks(self, chunks: list[Chunk]) -> LLMEnrichmentSummary:
        """Batch-enrich a list of chunks and return aggregate stats.

        Args:
            chunks: Chunks to enrich.

        Returns:
            ``LLMEnrichmentSummary`` with per-chunk results and token counts.
        """
        summary = LLMEnrichmentSummary()
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            results = self._enrich_batch(batch, summary)
            summary.results.extend(results)
        return summary
