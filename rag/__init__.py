"""RAG Framework — top-level package.

Sets ``KMP_DUPLICATE_LIB_OK`` before any submodule can import ``faiss``,
``torch``, or ``voyageai`` — their native dependencies each statically link
their own copy of OpenMP, and depending on load order (observed during a
real TB1c run: ``rag.ingest`` with OpenAI embeddings, then
``retrieve_with_metadata`` with ``reranker_provider="voyage"``) that
collision raises ``OMP: Error #15`` and aborts the whole process, not just
the offending call — every in-flight and future request on the process
dies with it. This must be set here, at package import time, rather than
lazily inside ``faiss_local``/``voyage_rerank``/``crossencoder_reranker``
themselves, because Python always executes a package's ``__init__.py``
before any of its submodules — by the time any of those modules run their
own ``import faiss``/``import voyageai``/``import torch``, this has
already taken effect regardless of which one loads first. ``docker-compose.
yml`` already sets this for containerized runs; this covers the MCP
server, CLI, and Studio when run directly (e.g. the smoke-test commands in
ECOSYSTEM_INTEGRATION_PLAN.md 附录 F/M, which don't set it).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
