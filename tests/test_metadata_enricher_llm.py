"""Tests for Task 12.3 — LLM batch metadata enrichment.

Acceptance criteria:
- Chunks can be batch-enriched using LLM
- Usage (prompt_tokens, completion_tokens) appears in summary
- Falls back to rules on LLM failure
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rag.core.contracts.chunk import Chunk
from rag.core.interfaces.llm_client import LLMResponse
from rag.infra.chunking.metadata_enricher_llm_batch import (
    LLMBatchMetadataEnricher,
    LLMEnrichmentResult,
    LLMEnrichmentSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        stable_text=text,
        display_text=text,
        chunk_signature=chunk_id,
        block_hashes=[chunk_id],
    )


def _make_llm_response(items: list[dict], prompt_tokens=10, completion_tokens=20) -> LLMResponse:
    from rag.core.interfaces.llm_client import LLMResponse
    return LLMResponse(
        text=json.dumps(items),
        model="gpt-test",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _mock_llm(items: list[dict], prompt_tokens=10, completion_tokens=20) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = _make_llm_response(items, prompt_tokens, completion_tokens)
    return llm


class _RealSignatureLLMClient:
    """Implements BaseLLMClient.generate()'s real signature exactly —
    generate(self, prompt: str, **kwargs) — unlike MagicMock, this raises
    TypeError on a call shape the real interface doesn't accept. Used to
    catch interface-mismatch regressions that a loose mock would hide.
    """

    def __init__(self, response_items: list[dict]):
        self._response_items = response_items
        self.received_prompts: list[str] = []

    def generate(self, prompt: str, **kwargs):
        self.received_prompts.append(prompt)
        return _make_llm_response(self._response_items)


# ---------------------------------------------------------------------------
# Basic enrichment
# ---------------------------------------------------------------------------


class TestLLMBatchEnrichment:
    def test_returns_summary_with_results(self):
        llm = _mock_llm([{"title": "T1", "summary": "S1", "tags": ["tag1"]}])
        enricher = LLMBatchMetadataEnricher(llm)
        chunks = [_make_chunk("Some content about the pipeline here.", "c1")]
        summary = enricher.enrich_chunks(chunks)
        assert isinstance(summary, LLMEnrichmentSummary)
        assert len(summary.results) == 1

    def test_result_fields_populated(self):
        llm = _mock_llm([{"title": "My Title", "summary": "My summary.", "tags": ["rag", "pipeline"]}])
        enricher = LLMBatchMetadataEnricher(llm)
        chunks = [_make_chunk("Content for enrichment here.", "c1")]
        summary = enricher.enrich_chunks(chunks)
        r = summary.results[0]
        assert r.title == "My Title"
        assert r.summary == "My summary."
        assert "rag" in r.tags

    def test_chunk_id_set_on_result(self):
        llm = _mock_llm([{"title": "T", "summary": "S", "tags": []}])
        enricher = LLMBatchMetadataEnricher(llm)
        chunks = [_make_chunk("Content.", "my-chunk")]
        summary = enricher.enrich_chunks(chunks)
        assert summary.results[0].chunk_id == "my-chunk"

    def test_metadata_annotated(self):
        llm = _mock_llm([{"title": "LLM Title", "summary": "LLM Summary.", "tags": ["ai"]}])
        enricher = LLMBatchMetadataEnricher(llm, annotate=True)
        chunk = _make_chunk("Content for enrichment.", "c1")
        enricher.enrich_chunks([chunk])
        assert chunk.metadata["title"] == "LLM Title"
        assert chunk.metadata["summary"] == "LLM Summary."
        assert "ai" in chunk.metadata["tags"]

    def test_calls_generate_with_real_base_llm_client_signature(self):
        # Regression guard: _enrich_batch() previously called
        # self._llm.generate(messages=[...]) — BaseLLMClient.generate()'s
        # real signature is generate(self, prompt: str, **kwargs), which
        # has no `messages` parameter and requires positional `prompt`.
        # MagicMock accepts any call shape and silently hid this — using a
        # class with the real signature makes the mismatch a hard TypeError
        # instead of a masked no-op.
        llm = _RealSignatureLLMClient([{"title": "T", "summary": "S", "tags": ["x"]}])
        enricher = LLMBatchMetadataEnricher(llm)
        chunks = [_make_chunk("Real interface content.", "c1")]

        summary = enricher.enrich_chunks(chunks)

        assert summary.fallback_batches == 0  # no TypeError forced a fallback
        assert summary.results[0].title == "T"
        assert len(llm.received_prompts) == 1
        assert "Real interface content." in llm.received_prompts[0]


# ---------------------------------------------------------------------------
# Token usage tracing
# ---------------------------------------------------------------------------


class TestTokenUsageTracing:
    def test_token_counts_in_summary(self):
        llm = _mock_llm([{"title": "T", "summary": "S", "tags": []}], prompt_tokens=15, completion_tokens=25)
        enricher = LLMBatchMetadataEnricher(llm)
        summary = enricher.enrich_chunks([_make_chunk("Content.", "c1")])
        assert summary.prompt_tokens == 15
        assert summary.completion_tokens == 25
        assert summary.total_tokens == 40

    def test_token_counts_accumulate_across_batches(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                [{"title": "T", "summary": "S", "tags": []}],
                prompt_tokens=10, completion_tokens=5
            ),
            _make_llm_response(
                [{"title": "T", "summary": "S", "tags": []}],
                prompt_tokens=10, completion_tokens=5
            ),
        ]
        enricher = LLMBatchMetadataEnricher(llm, batch_size=1)
        chunks = [_make_chunk("Content.", f"c{i}") for i in range(2)]
        summary = enricher.enrich_chunks(chunks)
        assert summary.prompt_tokens == 20
        assert summary.completion_tokens == 10
        assert summary.total_tokens == 30

    def test_empty_input_zero_tokens(self):
        llm = _mock_llm([])
        enricher = LLMBatchMetadataEnricher(llm)
        summary = enricher.enrich_chunks([])
        assert summary.total_tokens == 0
        assert summary.results == []


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    def test_multiple_chunks_in_one_batch(self):
        items = [
            {"title": f"Title {i}", "summary": f"Summary {i}.", "tags": [f"tag{i}"]}
            for i in range(3)
        ]
        llm = _mock_llm(items)
        enricher = LLMBatchMetadataEnricher(llm, batch_size=10)
        chunks = [_make_chunk(f"Content {i}.", f"c{i}") for i in range(3)]
        summary = enricher.enrich_chunks(chunks)
        assert len(summary.results) == 3
        assert llm.generate.call_count == 1  # one batch

    def test_batch_split_into_multiple_calls(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response([{"title": f"T{i}", "summary": "S.", "tags": []} for i in range(2)]),
            _make_llm_response([{"title": "T2", "summary": "S.", "tags": []}]),
        ]
        enricher = LLMBatchMetadataEnricher(llm, batch_size=2)
        chunks = [_make_chunk(f"Content {i}.", f"c{i}") for i in range(3)]
        summary = enricher.enrich_chunks(chunks)
        assert len(summary.results) == 3
        assert llm.generate.call_count == 2


# ---------------------------------------------------------------------------
# Fallback to rules on LLM failure
# ---------------------------------------------------------------------------


class TestFallbackOnFailure:
    def test_falls_back_on_llm_exception(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("API unavailable")
        enricher = LLMBatchMetadataEnricher(llm, annotate=True)
        chunk = _make_chunk("The pipeline processes documents efficiently here.", "c1")
        summary = enricher.enrich_chunks([chunk])
        # Should still have a result via rules fallback
        assert len(summary.results) == 1
        assert summary.results[0].used_fallback is True
        assert summary.fallback_batches == 1

    def test_fallback_writes_metadata(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("API down")
        enricher = LLMBatchMetadataEnricher(llm, annotate=True)
        chunk = _make_chunk("The pipeline processes documents efficiently here.", "c1")
        enricher.enrich_chunks([chunk])
        # Rules fallback should have written metadata
        assert "title" in chunk.metadata
        assert "summary" in chunk.metadata

    def test_fallback_not_triggered_on_success(self):
        llm = _mock_llm([{"title": "T", "summary": "S", "tags": []}])
        enricher = LLMBatchMetadataEnricher(llm)
        summary = enricher.enrich_chunks([_make_chunk("Content.", "c1")])
        assert summary.fallback_batches == 0
        assert summary.results[0].used_fallback is False

    def test_falls_back_on_invalid_json(self):
        llm = MagicMock()
        from rag.core.interfaces.llm_client import LLMResponse
        llm.generate.return_value = LLMResponse(
            text="not valid json at all",
            model="gpt-test",
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
        )
        enricher = LLMBatchMetadataEnricher(llm, annotate=True)
        chunk = _make_chunk("The pipeline processes documents efficiently here.", "c1")
        summary = enricher.enrich_chunks([chunk])
        assert summary.results[0].used_fallback is True


# ---------------------------------------------------------------------------
# LLMEnrichmentSummary.add_usage
# ---------------------------------------------------------------------------


class TestLLMEnrichmentSummary:
    def test_add_usage_accumulates(self):
        s = LLMEnrichmentSummary()
        s.add_usage(10, 20)
        s.add_usage(5, 15)
        assert s.prompt_tokens == 15
        assert s.completion_tokens == 35
        assert s.total_tokens == 50

    def test_initial_zeros(self):
        s = LLMEnrichmentSummary()
        assert s.prompt_tokens == 0
        assert s.completion_tokens == 0
        assert s.total_tokens == 0
        assert s.fallback_batches == 0
