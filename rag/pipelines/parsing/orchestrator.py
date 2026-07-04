"""Parser orchestrator — routes artifacts to parsers and handles fallbacks."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser
from rag.infra.loading.local_file_loader import RawArtifact
from rag.infra.sniffing.composite_sniffer import SniffResult
from rag.pipelines.parsing.plans import ParsePlan

logger = logging.getLogger(__name__)


def _unsupported_document(artifact: RawArtifact, detected_type: str) -> Document:
    """Produce a Document that signals an unsupported format.

    Args:
        artifact: The raw artifact that could not be parsed.
        detected_type: The detected type string.

    Returns:
        A Document with a single IRBlock containing an error message
        and a ParseReport flagging the unsupported format.
    """
    doc_id = hashlib.sha256(artifact.source_path.encode()).hexdigest()[:16]
    block = IRBlock(
        block_type=BlockType.UNKNOWN,
        text=f"[unsupported_format: {detected_type}]",
    )
    report = ParseReport(
        char_count=0,
        block_count=0,
        non_printable_ratio=0.0,
        repetition_score=0.0,
        parser_used="none",
        fallback_triggered=False,
    )
    return Document(
        doc_id=doc_id,
        source_path=artifact.source_path,
        mime_type=artifact.metadata.get("extension", ""),
        metadata={"unsupported_format": True, "detected_type": detected_type},
        blocks=[block],
        parse_report=report,
    )


class ParserOrchestrator:
    """Routes documents to parsers and manages fallback chains.

    Loads the parser routing table from ``configs/routers/parser_candidates.yaml``
    and uses the registered parser instances to attempt parsing in candidate order.

    Usage::

        registry = {"pymupdf": PyMuPDFParser(), "md_parser": MdParser()}
        orchestrator = ParserOrchestrator(registry)
        plan = orchestrator.route(sniff_result)
        document = orchestrator.parse(artifact, plan)

    Args:
        parser_registry: Dict mapping parser name → BaseParser instance.
        router_config_path: Path to the parser_candidates.yaml config file.
            If None, the default project config is used.
    """

    def __init__(
        self,
        parser_registry: dict[str, BaseParser],
        router_config_path: str | Path | None = None,
    ) -> None:
        self._registry = parser_registry
        # Defaults — overwritten by _load_routes() if config is found
        self._ocr_enabled: bool = True
        self._ocr_min_chars: int = 100
        self._vision_enabled: bool = True
        self._vision_min_ocr_confidence: float = 0.6
        self._routes = self._load_routes(router_config_path)

    def _load_routes(self, config_path: str | Path | None) -> dict[str, list[str]]:
        """Load the routing table from YAML.

        Args:
            config_path: Path to parser_candidates.yaml, or None for default.

        Returns:
            Dict mapping detected_type → list of parser names.
        """
        if config_path is None:
            # Walk up from this file to find the project configs/ directory
            candidate = Path(__file__).resolve()
            for parent in candidate.parents:
                yaml_path = parent / "configs" / "routers" / "parser_candidates.yaml"
                if yaml_path.exists():
                    config_path = yaml_path
                    break

        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "parser_candidates.yaml not found. Set router_config_path explicitly."
            )

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        routes: dict[str, list[str]] = {}
        for detected_type, route_cfg in data.get("routes", {}).items():
            routes[detected_type] = route_cfg.get("candidates", [])

        # Load OCR fallback config
        ocr_cfg = data.get("ocr", {})
        self._ocr_enabled = bool(ocr_cfg.get("enabled", True))
        self._ocr_min_chars = int(ocr_cfg.get("min_chars_threshold", 100))

        # Load vision escalation config
        vision_cfg = data.get("vision", {})
        self._vision_enabled = bool(vision_cfg.get("enabled", True))
        self._vision_min_ocr_confidence = float(
            vision_cfg.get("min_ocr_confidence", 0.6)
        )

        return routes

    def route(self, sniff_result: SniffResult) -> ParsePlan:
        """Produce a ParsePlan from a SniffResult.

        Args:
            sniff_result: Output of CompositeSniffer.sniff().

        Returns:
            ParsePlan with the ordered candidate list for this detected type.
        """
        detected_type = sniff_result.detected_type
        candidates = self._routes.get(detected_type, [])
        unsupported = detected_type in ("unsupported", "unknown") or not candidates

        return ParsePlan(
            detected_type=detected_type,
            mime_type=sniff_result.mime_type,
            candidates=candidates,
            unsupported=unsupported,
        )

    def parse(self, artifact: RawArtifact, plan: ParsePlan) -> Document:
        """Parse a RawArtifact using the candidates in the ParsePlan.

        Tries each candidate parser in order. Returns the first successful
        result. If all candidates fail or the plan is unsupported, returns
        an unsupported-format Document.

        Args:
            artifact: RawArtifact from the loader stage.
            plan: ParsePlan produced by ``route()``.

        Returns:
            A Document. If format is unsupported or all parsers fail,
            the Document's metadata will contain ``unsupported_format=True``.
        """
        if plan.unsupported or not plan.candidates:
            logger.warning(
                "Unsupported format '%s' for %s",
                plan.detected_type,
                artifact.source_path,
            )
            return _unsupported_document(artifact, plan.detected_type)

        last_error: Exception | None = None
        fallback_triggered = False
        # Best low-quality result seen so far — returned if every later
        # tier is unavailable or fails, so escalation never loses data.
        best_document: Document | None = None

        for i, parser_name in enumerate(plan.candidates):
            if not self._candidate_usable(parser_name):
                continue

            parser = self._registry.get(parser_name)
            if parser is None:
                logger.debug("Parser '%s' not registered — skipping.", parser_name)
                continue

            try:
                document = parser.parse(artifact.source_path)

                remaining = plan.candidates[i + 1:]
                escalate_reason = self._escalation_reason(
                    parser_name, document, remaining
                )
                if escalate_reason:
                    logger.info(
                        "Parser '%s' result for '%s' %s — escalating to next tier.",
                        parser_name,
                        artifact.source_path,
                        escalate_reason,
                    )
                    fallback_triggered = True
                    last_error = None
                    best_document = self._better_of(best_document, document)
                    continue  # Try next candidate tier

                if i > 0 or fallback_triggered:
                    # A fallback parser was used
                    if document.parse_report:
                        document.parse_report.fallback_triggered = True
                return document
            except Exception as exc:
                logger.warning(
                    "Parser '%s' failed for %s: %s",
                    parser_name,
                    artifact.source_path,
                    exc,
                )
                last_error = exc
                fallback_triggered = True

        # Escalation ran out of usable tiers — return the best partial
        # result rather than discarding extracted content.
        if best_document is not None:
            logger.warning(
                "No further parser tier available for %s — keeping best "
                "low-quality result (parser=%s, chars=%d).",
                artifact.source_path,
                best_document.parse_report.parser_used
                if best_document.parse_report else "?",
                best_document.parse_report.char_count
                if best_document.parse_report else 0,
            )
            if best_document.parse_report:
                best_document.parse_report.fallback_triggered = True
            return best_document

        logger.error(
            "All parsers failed for %s (last error: %s)",
            artifact.source_path,
            last_error,
        )
        return _unsupported_document(artifact, plan.detected_type)

    def _candidate_usable(self, parser_name: str) -> bool:
        """Return True if a candidate is registered and enabled by config."""
        if parser_name == "pdf_ocr" and not self._ocr_enabled:
            logger.debug("OCR disabled — skipping pdf_ocr candidate.")
            return False
        if parser_name in ("pdf_vision", "pdf_hybrid") and not self._vision_enabled:
            logger.debug("Vision disabled — skipping %s candidate.", parser_name)
            return False
        return self._registry.get(parser_name) is not None

    def _escalation_reason(
        self,
        parser_name: str,
        document: Document,
        remaining: list[str],
    ) -> str | None:
        """Decide whether a parse result should escalate to the next tier.

        Args:
            parser_name: Name of the parser that produced ``document``.
            document: The parse result to evaluate.
            remaining: Candidate names after this parser in the plan.

        Returns:
            A human-readable reason string when the result is too weak and
            a later tier is usable, or None to accept the result.
        """
        if document.parse_report is None:
            return None

        # Quality-based escalation only targets the OCR/vision tiers.
        # Generic fallback chains (e.g. alternative HTML parsers) fall
        # through on parse *failure* only, never on thin content.
        escalation_tiers = [
            name
            for name in remaining
            if name in ("pdf_ocr", "pdf_vision") and self._candidate_usable(name)
        ]
        if not escalation_tiers:
            return None  # Nothing to escalate to — accept what we have

        char_count = document.parse_report.char_count
        if char_count < self._ocr_min_chars:
            return (
                f"has only {char_count} chars "
                f"(threshold={self._ocr_min_chars})"
            )

        if parser_name == "pdf_ocr" and "pdf_vision" in escalation_tiers:
            confidence = document.metadata.get("mean_ocr_confidence")
            if (
                confidence is not None
                and confidence < self._vision_min_ocr_confidence
            ):
                return (
                    f"has low OCR confidence {confidence:.2f} "
                    f"(threshold={self._vision_min_ocr_confidence})"
                )

        return None

    @staticmethod
    def _better_of(
        current: Document | None, candidate: Document
    ) -> Document:
        """Return whichever document extracted more characters."""
        if current is None or current.parse_report is None:
            return candidate
        if candidate.parse_report is None:
            return current
        if candidate.parse_report.char_count >= current.parse_report.char_count:
            return candidate
        return current
