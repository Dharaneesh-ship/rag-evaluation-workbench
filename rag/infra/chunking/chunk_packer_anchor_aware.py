"""Anchor-aware chunk packer — packs TextBlocks into Chunks respecting token budgets.

Heading-absorption rule
-----------------------
A chunk that contains *only* anchor (heading) blocks with no paragraph content
is semantically empty — it carries a section label but no retrievable facts.
Such a chunk wastes a context-window slot and scores poorly in retrieval because
it lacks body text for BM25/vector matching.

To prevent this, the packer defers flushing when the current buffer is all-
heading.  When the next content block (paragraph) arrives, it is added to the
same buffer, so the final chunk starts with the heading and immediately contains
its content.  This means a chunk like:

    [HEADING] EDUCATION
    Wake Forest University…  GPA: 3.291 …

is produced instead of two separate chunks:

    [HEADING] EDUCATION          ← useless on its own
    Wake Forest University…      ← missing its section label

The rule also applies across consecutive headings: if the parser emits two
back-to-back headings (e.g. a parent section and a sub-section title), they are
kept together until actual body text arrives.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from rag.core.contracts.chunk import Chunk
from rag.core.contracts.text_block import TextBlock
from rag.core.interfaces.chunk_packer import BaseChunkPacker
from rag.infra.chunking.anchor_annotator_rules import AnchorAnnotation, AnchorAnnotator

logger = logging.getLogger(__name__)

# Default token budget per chunk
_DEFAULT_TOKEN_BUDGET = 512

# Tiktoken encoder — used for accurate token counting across all scripts
# (English, Chinese, Japanese, etc.). Falls back to char-based estimate
# if tiktoken is unavailable.
try:
    import tiktoken as _tiktoken

    _ENCODER = _tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover
    _ENCODER = None
    _TIKTOKEN_AVAILABLE = False

# Fallback: chars per token for English-only heuristic
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) or fall back to char heuristic.

    tiktoken gives accurate counts for all scripts including CJK where each
    character is typically 1 token, vs ~0.25 tokens in the naïve 4-chars
    heuristic. Without it a 1000-char Chinese document is underestimated by
    4× and never gets split into multiple chunks.

    Args:
        text: Input text.

    Returns:
        Token count (exact if tiktoken available, approximate otherwise).
    """
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        return max(1, len(_ENCODER.encode(text)))
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _compute_chunk_signature(block_hashes: list[str], part: int | None = None) -> str:
    """Compute SHA-256 over the ordered block hashes.

    Args:
        block_hashes: Ordered list of block_hash strings.
        part: Optional part index for chunks produced by splitting a single
            oversized block — keeps signatures unique when several chunks
            share the same block hash.

    Returns:
        64-character hex digest.
    """
    combined = "|".join(block_hashes)
    if part is not None:
        combined += f"#part{part}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# Sentence boundary used when force-splitting a single oversized block.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _split_oversized_text(text: str, token_budget: int) -> list[str]:
    """Split one oversized text into sentence groups within the budget.

    Args:
        text: Block text whose token count exceeds the budget.
        token_budget: Maximum tokens per resulting part.

    Returns:
        List of non-empty text parts, each within the budget where possible
        (a single sentence longer than the budget stays intact).
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        s_tokens = _approx_tokens(sentence)
        if current and current_tokens + s_tokens > token_budget:
            parts.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += s_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def _build_chunk(doc_id: str, blocks: list[TextBlock], annotations: list[AnchorAnnotation]) -> Chunk:
    """Assemble a Chunk from a group of consecutive TextBlocks.

    Args:
        doc_id: Parent document ID.
        blocks: Consecutive TextBlocks forming this chunk.
        annotations: Anchor annotations parallel to blocks.

    Returns:
        Assembled Chunk with stable_text, display_text, chunk_signature.
    """
    parts: list[str] = []
    display_parts: list[str] = []

    # Track the most recent heading seen so it can be prepended to the
    # stable_text of pure-content blocks that follow it within the chunk.
    # This gives BM25 keyword access to the section label even when a block
    # contains only numeric/tabular content (e.g. "OVERALL 37.50 36.00 3.291").
    last_heading: str = ""

    for block, ann in zip(blocks, annotations):
        text = block.text.strip()
        if ann.anchor_type != "none":
            # Prefix display text with anchor type tag for UI readability
            display_parts.append(f"[{ann.anchor_type.upper()}] {text}")
            parts.append(text)
            last_heading = text
        else:
            display_parts.append(text)
            # Inject the section heading as a context prefix in stable_text so
            # BM25 / vector indices can match via the heading keywords even when
            # the paragraph itself is purely numeric or tabular.
            if last_heading:
                parts.append(f"{last_heading}: {text}")
            else:
                parts.append(text)

    stable_text = "\n\n".join(parts)
    display_text = "\n\n".join(display_parts)
    block_hashes = [b.block_hash for b in blocks]
    chunk_signature = _compute_chunk_signature(block_hashes)
    token_count = _approx_tokens(stable_text)

    # Collect metadata from the first block
    metadata: dict[str, Any] = {}
    first = blocks[0]
    if first.page is not None:
        metadata["start_page"] = first.page
    last = blocks[-1]
    if last.page is not None:
        metadata["end_page"] = last.page
    if first.section_path:
        metadata["section_path"] = first.section_path
    metadata["loc_span"] = [first.sequence, last.sequence]

    return Chunk(
        doc_id=doc_id,
        stable_text=stable_text,
        display_text=display_text,
        chunk_signature=chunk_signature,
        block_hashes=block_hashes,
        token_count=token_count,
        metadata=metadata,
    )


class AnchorAwareChunkPacker(BaseChunkPacker):
    """Chunk packer that respects anchor boundaries and token budgets.

    Packing strategy:
    1. Annotate all blocks with anchor information.
    2. Iterate blocks in sequence. Start a new chunk when:
       - The current block is an anchor (heading/section) AND the buffer
         is non-empty (anchor = natural break point).
       - Adding the current block would exceed the token budget.
    3. Build each chunk from the accumulated buffer.

    This produces chunks that start at natural section boundaries and
    stay within the token budget.

    Usage::

        packer = AnchorAwareChunkPacker(token_budget=512)
        chunks = packer.pack(text_blocks)

    Args:
        token_budget: Maximum approximate token count per chunk.
            Defaults to 512.
        annotator_config_path: Path to anchors.yaml. If None, auto-detected.
    """

    def __init__(
        self,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        annotator_config_path: str | Path | None = None,
    ) -> None:
        self._token_budget = token_budget
        self._annotator = AnchorAnnotator(config_path=annotator_config_path)

    def pack(self, blocks: list[TextBlock]) -> list[Chunk]:
        """Pack TextBlocks into anchor-aware Chunks.

        Args:
            blocks: Ordered TextBlocks from the block splitter.
                Must all share the same doc_id.

        Returns:
            List of Chunks. Empty input → empty output.
        """
        if not blocks:
            return []

        doc_id = blocks[0].doc_id
        annotations = self._annotator.annotate(blocks)

        chunks: list[Chunk] = []
        buffer_blocks: list[TextBlock] = []
        buffer_anns: list[AnchorAnnotation] = []
        buffer_tokens = 0

        def _buffer_has_content() -> bool:
            """Return True if the buffer contains at least one non-anchor block."""
            return any(a.anchor_type == "none" for a in buffer_anns)

        # Rolling section heading within the current buffer — mirrors the
        # prefix injection in _build_chunk so budget accounting matches the
        # tokens actually emitted into stable_text.
        buffer_heading = ""

        def _effective_tokens(block: TextBlock, is_anchor: bool, heading: str) -> int:
            """Tokens this block will contribute to stable_text (+1 separator)."""
            text = block.text.strip()
            if not is_anchor and heading:
                text = f"{heading}: {text}"
            return _approx_tokens(text) + 1

        for block, ann in zip(blocks, annotations):
            is_anchor = ann.anchor_type != "none"

            # A single block larger than the whole budget can never fit —
            # flush the buffer and force-split it by sentences.
            if _approx_tokens(block.text) > self._token_budget:
                if buffer_blocks:
                    chunks.append(_build_chunk(doc_id, buffer_blocks, buffer_anns))
                    buffer_blocks, buffer_anns = [], []
                    buffer_tokens, buffer_heading = 0, ""
                chunks.extend(
                    self._build_oversized_chunks(doc_id, block, ann)
                )
                continue

            block_tokens = _effective_tokens(block, is_anchor, buffer_heading)

            # Decide whether to flush the current buffer:
            # - At an anchor boundary (new section heading arrives) AND
            #   the buffer already has paragraph content (not heading-only).
            #   Heading-only buffers are held open so the heading absorbs the
            #   content that follows it rather than becoming an empty chunk.
            # - When adding this block would exceed the token budget regardless
            #   of type (safety valve — prevents unbounded accumulation).
            should_flush = buffer_blocks and (
                (is_anchor and _buffer_has_content())
                or (buffer_tokens + block_tokens > self._token_budget)
            )

            if should_flush:
                chunks.append(_build_chunk(doc_id, buffer_blocks, buffer_anns))
                buffer_blocks = []
                buffer_anns = []
                buffer_tokens = 0
                buffer_heading = ""
                # Recompute cost against the now-empty buffer (no prefix)
                block_tokens = _effective_tokens(block, is_anchor, buffer_heading)

            buffer_blocks.append(block)
            buffer_anns.append(ann)
            buffer_tokens += block_tokens
            if is_anchor:
                buffer_heading = block.text.strip()

        # Flush remaining buffer
        if buffer_blocks:
            chunks.append(_build_chunk(doc_id, buffer_blocks, buffer_anns))

        return chunks

    def _build_oversized_chunks(
        self,
        doc_id: str,
        block: TextBlock,
        ann: AnchorAnnotation,
    ) -> list[Chunk]:
        """Split one over-budget block into sentence-group chunks.

        Every resulting chunk references the same source block (same
        ``block_hash``); a part index is mixed into the chunk signature to
        keep chunk ids unique.

        Args:
            doc_id: Parent document ID.
            block: The oversized TextBlock.
            ann: Anchor annotation for the block.

        Returns:
            One Chunk per sentence group.
        """
        text = block.text.strip()
        parts = _split_oversized_text(text, self._token_budget)
        if len(parts) > 1:
            logger.info(
                "Block %s exceeds token budget (%d) — split into %d chunks.",
                block.block_hash[:12],
                self._token_budget,
                len(parts),
            )

        metadata: dict[str, Any] = {}
        if block.page is not None:
            metadata["start_page"] = block.page
            metadata["end_page"] = block.page
        if block.section_path:
            metadata["section_path"] = block.section_path
        metadata["loc_span"] = [block.sequence, block.sequence]

        chunks: list[Chunk] = []
        for i, part in enumerate(parts):
            display = part
            if i == 0 and ann.anchor_type != "none":
                display = f"[{ann.anchor_type.upper()}] {part}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    stable_text=part,
                    display_text=display,
                    chunk_signature=_compute_chunk_signature(
                        [block.block_hash],
                        part=i if len(parts) > 1 else None,
                    ),
                    block_hashes=[block.block_hash],
                    token_count=_approx_tokens(part),
                    metadata={**metadata, **({"part": i} if len(parts) > 1 else {})},
                )
            )
        return chunks
