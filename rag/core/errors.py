"""Stable MCP tool error codes for retrieval failures.

Per ECOSYSTEM_INTEGRATION_PLAN.md 定案 #13: internal retrieval failures must
surface as a real MCP tool error (``isError=True``), not a protocol-success
response with an ``error`` field buried in the payload — the latter is
indistinguishable from a legitimate empty result to a caller that forgets to
check the field.

FastMCP has no side channel for structured error data (see
``mcp/server/lowlevel/server.py::call_tool`` — any exception raised from a
tool function is converted to ``CallToolResult(isError=True, content=[Text
Content(text=str(exc))])``), so the stable code is conveyed as a
``"<CODE>: <detail>"`` prefix in the exception text. The code prefix is the
stable contract; everything after the colon is free-form debugging detail
and may change between releases.
"""

from __future__ import annotations


class RagMCPError(Exception):
    """Base class for retrieval failures that must surface as MCP tool errors.

    Args:
        detail: Human-readable detail, used for debugging only — not part
            of the stable contract. May be empty.
    """

    error_code: str = "RETRIEVAL_INTERNAL_ERROR"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        message = f"{self.error_code}: {detail}" if detail else self.error_code
        super().__init__(message)


class IndexNotFoundError(RagMCPError):
    """No usable index exists for the requested retrieval configuration."""

    error_code = "INDEX_NOT_FOUND"


class IndexConfigMismatchError(RagMCPError):
    """The saved index was built with a different provider/model/dimension."""

    error_code = "INDEX_CONFIG_MISMATCH"


class EmbeddingProviderUnavailableError(RagMCPError):
    """The requested embedding provider's dependency (API key, package) is unavailable."""

    error_code = "EMBEDDING_PROVIDER_UNAVAILABLE"


class RerankerUnavailableError(RagMCPError):
    """The requested reranker's dependency (API key, package) is unavailable."""

    error_code = "RERANKER_UNAVAILABLE"


class InvalidCollectionError(RagMCPError):
    """The requested collection does not exist in the corpus."""

    error_code = "INVALID_COLLECTION"


class RetrievalInternalError(RagMCPError):
    """Catch-all for unexpected retrieval failures with no more specific code."""

    error_code = "RETRIEVAL_INTERNAL_ERROR"
