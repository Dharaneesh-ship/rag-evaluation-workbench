"""Index manifest — records what an on-disk BM25/FAISS index was actually built with.

Written by ``IndexManager.save()`` and checked before every vector query so a
query-time provider/model/dimension mismatch fails loudly
(``INDEX_CONFIG_MISMATCH``, see ``rag.core.errors``) instead of silently
querying against vectors that mean something different from what the caller
asked for — the "assumed hybrid, actually BM25-only" failure mode described
in HANDOFF_MAEDA_INTEGRATION.md §2.2/4.4.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "index_manifest.json"

#: Bumped only if the manifest's field *semantics* change under the same
#: field names — mirrors the cross-repo convention in
#: ECOSYSTEM_INTEGRATION_PLAN.md 定案 #4b (never redefine a field silently
#: under an unchanged version number).
SCHEMA_VERSION = "1"


@dataclass
class IndexManifest:
    """Describes the embedding configuration an index directory was built with.

    Attributes:
        schema_version: Manifest schema version (see ``SCHEMA_VERSION``).
        embedding_provider: Provider key (e.g. ``"openai"``), or None if no
            vectors have ever been added to this index_dir (BM25-only).
        embedding_model: Model identifier used to produce the stored vectors.
        vector_dimension: Dimensionality of the stored vectors.
    """

    schema_version: str = SCHEMA_VERSION
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    vector_dimension: Optional[int] = None

    @classmethod
    def empty(cls) -> "IndexManifest":
        """Return the manifest for a fresh/BM25-only index_dir."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexManifest":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            embedding_provider=data.get("embedding_provider"),
            embedding_model=data.get("embedding_model"),
            vector_dimension=data.get("vector_dimension"),
        )

    def save(self, index_dir: str | Path) -> None:
        """Write this manifest to ``<index_dir>/index_manifest.json``."""
        path = Path(index_dir) / _MANIFEST_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "IndexManifest":
        """Load the manifest from ``index_dir``, or return :meth:`empty` if absent/corrupt."""
        path = Path(index_dir) / _MANIFEST_FILENAME
        if not path.exists():
            return cls.empty()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning(
                "Failed to parse index manifest at '%s'; treating as empty (BM25-only).",
                path,
            )
            return cls.empty()
