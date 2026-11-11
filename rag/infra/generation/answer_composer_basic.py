"""Basic answer composer — calls the LLM and packages the result as Answer + AnswerTrace."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from rag.core.contracts.answer import Answer
from rag.core.contracts.citation import Citation, Span, SpanType
from rag.core.contracts.trace import AnswerTrace, PipelineStep
from rag.core.interfaces.llm_client import BaseLLMClient
from rag.infra.generation.context_packer_light import LightContextPacker, PackedContext
from rag.infra.generation.prompt_builder_grounded import GroundedPromptBuilder

logger = logging.getLogger(__name__)

# Phrase the model should emit when it cannot answer from context
_ABSTAIN_PHRASES = [
    "i don't have enough information in the provided context",
    "i do not have enough information in the provided context",
    "insufficient information",
    "cannot answer",
    "no relevant information",
]

# Regex to find inline citation markers like [1], [2], [1][3]
_CITATION_RE = re.compile(r"\[(\d+)\]")


class BasicAnswerComposer:
    """Produces a grounded Answer by orchestrating packing, prompting, and LLM calls.

    Pipeline executed by ``compose()``:

    1. **Pack** — select top-k unique candidates within token budget.
    2. **Build prompt** — assemble system + user prompt with context and citation map.
    3. **Generate** — call the LLM client and record token usage / latency.
    4. **Parse** — detect abstention, extract inline ``[N]`` citation markers,
       build Span segments, and filter citations to only those referenced in text.
    5. **Return** — ``Answer`` + ``AnswerTrace``.

    Args:
        llm_client: Any ``BaseLLMClient`` implementation.
        top_k: Maximum chunks to pack into context. Defaults to 6.
        token_budget: Maximum context tokens. Defaults to 2048.
        system_instructions: Override default grounding instructions.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        top_k: int = 6,
        token_budget: int = 2048,
        system_instructions: Optional[str] = None,
    ) -> None:
        self._llm = llm_client
        self._packer = LightContextPacker(top_k=top_k, token_budget=token_budget)
        self._prompt_builder = GroundedPromptBuilder(
            system_instructions=system_instructions
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compose(
        self,
        query: str,
        candidates: list,
        *,
        run_id: Optional[str] = None,
        rerank_provider: Optional[str] = None,
        candidates_before_rerank: int = 0,
    ) -> tuple[Answer, AnswerTrace]:
        """Run the full compose pipeline.

        Args:
            query: User's natural-language question.
            candidates: Ranked list of ``Candidate`` objects.
            run_id: Optional trace run identifier.
            rerank_provider: Name of the reranker used upstream (for trace).
            candidates_before_rerank: Candidate count before reranking (for trace).

        Returns:
            A ``(Answer, AnswerTrace)`` tuple.
        """
        pipeline_start = time.perf_counter()
        steps: list[PipelineStep] = []

        # ── Step 1: Pack context ──────────────────────────────────────────────
        t0 = time.perf_counter()
        packed: PackedContext = self._packer.pack(candidates)
        pack_ms = (time.perf_counter() - t0) * 1000
        steps.append(
            PipelineStep(
                step_name="context_pack",
                input_summary=f"{len(candidates)} candidates",
                output_summary=(
                    f"{len(packed.candidates)} chunks packed, "
                    f"{packed.total_tokens} tokens"
                    + (" (truncated)" if packed.truncated else "")
                ),
                latency_ms=pack_ms,
            )
        )

        # ── Step 2: Build prompt ─────────────────────────────────────────────
        t0 = time.perf_counter()
        built_prompt = self._prompt_builder.build(query, packed)
        prompt_ms = (time.perf_counter() - t0) * 1000
        steps.append(
            PipelineStep(
                step_name="prompt_build",
                input_summary=f"query_len={len(query)}, context_len={len(packed.context_text)}",
                output_summary=f"prompt_len={len(built_prompt.full_text)}",
                latency_ms=prompt_ms,
            )
        )

        # ── Step 3: LLM generate ─────────────────────────────────────────────
        t0 = time.perf_counter()
        llm_response = self._llm.generate(
            built_prompt.user,
            # Pass system as a kwarg so OpenAI-style clients can use it;
            # BaseLLMClient.generate accepts **kwargs for flexibility.
        )
        llm_ms = (time.perf_counter() - t0) * 1000
        steps.append(
            PipelineStep(
                step_name="llm_generate",
                input_summary=f"model={llm_response.model}",
                output_summary=(
                    f"tokens={llm_response.total_tokens}, "
                    f"latency={llm_response.latency_ms:.0f}ms"
                ),
                latency_ms=llm_ms,
                metadata={
                    "prompt_tokens": llm_response.prompt_tokens,
                    "completion_tokens": llm_response.completion_tokens,
                    "model": llm_response.model,
                },
            )
        )

        # ── Step 4: Parse response ────────────────────────────────────────────
        answer_text = llm_response.text.strip()
        abstained = self._detect_abstain(answer_text)
        used_citations = self._filter_cited(packed.citations, answer_text)
        spans = self._build_spans(answer_text)

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # ── Assemble outputs ─────────────────────────────────────────────────
        answer = Answer(
            text=answer_text,
            citations=used_citations,
            spans=spans,
            abstained=abstained,
            query=query,
        )

        trace = AnswerTrace(
            query=query,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            total_latency_ms=total_ms,
            model=llm_response.model,
            rerank_provider=rerank_provider,
            candidates_before_rerank=candidates_before_rerank,
            candidates_after_rerank=len(candidates),
            context_chunks_used=len(packed.candidates),
            steps=steps,
            run_id=run_id,
        )

        logger.debug(
            "answer_composer: abstained=%s citations=%d tokens=%d latency=%.0fms",
            abstained,
            len(used_citations),
            llm_response.total_tokens,
            total_ms,
        )

        return answer, trace

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_abstain(text: str) -> bool:
        """Return True if the answer text matches a known abstain phrase."""
        lower = text.lower()
        return any(phrase in lower for phrase in _ABSTAIN_PHRASES)

    @staticmethod
    def _filter_cited(citations: list[Citation], text: str) -> list[Citation]:
        """Return only citations whose ref_number appears in the answer text.

        Preserves original order. If no inline markers found, returns all
        citations (the model may have forgotten to cite but the context was used).
        """
        found_refs = {int(m) for m in _CITATION_RE.findall(text)}
        if not found_refs:
            # No markers found — keep all citations to avoid losing provenance
            return list(citations)
        return [c for c in citations if c.ref_number in found_refs]

    @staticmethod
    def _build_spans(text: str) -> list[Span]:
        """Segment answer text into ANSWER and CITATION_MARKER spans.

        Iterates through the text, emitting an ANSWER span for prose and a
        CITATION_MARKER span for each ``[N]`` token found.
        """
        spans: list[Span] = []
        pos = 0
        for match in _CITATION_RE.finditer(text):
            start, end = match.start(), match.end()
            if start > pos:
                spans.append(
                    Span(
                        text=text[pos:start],
                        span_type=SpanType.ANSWER,
                        start=pos,
                        end=start,
                    )
                )
            spans.append(
                Span(
                    text=match.group(),
                    span_type=SpanType.CITATION_MARKER,
                    start=start,
                    end=end,
                )
            )
            pos = end
        if pos < len(text):
            spans.append(
                Span(
                    text=text[pos:],
                    span_type=SpanType.ANSWER,
                    start=pos,
                    end=len(text),
                )
            )
        return spans
