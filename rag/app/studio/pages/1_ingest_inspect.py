"""Ingest & Inspect — combined ingestion, chunk browser, and trace viewer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load API keys from .env so embedding and the pdf_vision parser tier work
# regardless of how the studio process was launched.
load_dotenv(override=False)

st.set_page_config(page_title="Ingest & Inspect · RAG Studio", layout="wide")

st.title("📥 Ingest & Inspect")
st.caption("Upload documents or fetch URLs, then inspect chunks and pipeline traces — all in one place.")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

for _key, _default in [
    ("ingest_result", None),
    ("ingest_config", None),
    ("ingest_error", None),
    ("last_doc_ids", []),       # doc_ids from the most recent ingest run
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    db_path = st.text_input(
        "SQLite DB path", value=os.environ.get("RAG_STUDIO_DB", "data/rag.db")
    )
    index_dir = st.text_input(
        "Index directory", value=os.environ.get("RAG_STUDIO_INDEX_DIR", "data/indexes")
    )
    collection = st.text_input("Collection name", value="default")

    st.divider()
    st.subheader("Embedding")
    embedding_provider = st.selectbox(
        "Provider",
        ["openai", "multilingual", "none"],
        index=0,
        help=(
            "**openai** — OpenAI API (requires OPENAI_API_KEY). Fast, high quality.\n\n"
            "**multilingual** — Local sentence-transformers model. No API key needed. "
            "Supports 50+ languages; enables cross-lingual retrieval.\n\n"
            "**none** — Skip embedding (BM25 only)."
        ),
    )

    _MULTILINGUAL_DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
    _MULTILINGUAL_DEFAULT_DIM = 768
    _OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
    _OPENAI_DEFAULT_DIM = 1536

    if embedding_provider == "multilingual":
        default_model = _MULTILINGUAL_DEFAULT_MODEL
        default_dim = _MULTILINGUAL_DEFAULT_DIM
    else:
        default_model = _OPENAI_DEFAULT_MODEL
        default_dim = _OPENAI_DEFAULT_DIM

    embedding_model = st.text_input(
        "Model", value=default_model, disabled=(embedding_provider == "none")
    )
    vector_dimension = st.number_input(
        "Vector dimension",
        min_value=1, max_value=4096, value=default_dim,
        disabled=(embedding_provider == "none"),
    )

    st.divider()
    st.subheader("Index")
    index_type = st.selectbox("Vector index type", ["FAISS (IndexFlatL2)", "none"], index=0)
    token_budget = st.number_input("Chunk token budget", min_value=64, max_value=2048, value=512)

    st.divider()
    st.subheader("Vision (scanned / complex PDFs)")
    vision_model = st.selectbox(
        "Vision model",
        ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"],
        index=0,
        help=(
            "Used for the pdf_vision (full-document) and pdf_hybrid "
            "(complex-page) parser tiers. Requires OPENAI_API_KEY.\n\n"
            "**gpt-4o-mini** — current default. $0.15 / $0.60 per 1M "
            "input/output tokens.\n\n"
            "**gpt-4.1-mini** — higher quality, higher cost ($0.40 / $1.60).\n\n"
            "**gpt-4.1-nano** — cheapest ($0.10 / $0.40), but a smaller "
            "model — may be less accurate transcribing dense tables/charts, "
            "which is exactly what this tier targets. Test before relying on it."
        ),
    )

    st.divider()
    st.subheader("Chunk Browser")
    sort_by = st.selectbox("Sort by", ["Ingestion order", "Token count ↑", "Token count ↓"])
    search_term = st.text_input("Search text", placeholder="keyword in chunk text…")

    st.divider()
    st.subheader("Traces")
    max_runs = st.number_input("Recent runs to show", min_value=5, max_value=200, value=50)


# ---------------------------------------------------------------------------
# Helper: build pipeline components
# ---------------------------------------------------------------------------

def _build_pipeline(db_path, index_dir, token_budget, embedding_provider,
                    embedding_model, vector_dimension, index_type,
                    vision_model="gpt-4o-mini"):
    """Instantiate stores, indexes, and IngestPipeline. Returns (pipeline, vec_index, kw_index)."""
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)

    embed_provider = None
    vec_index = None
    kw_index = None

    if embedding_provider in ("openai", "multilingual"):
        from rag.infra.indexes.faiss_local import FaissLocalIndex
        from rag.infra.indexes.index_manager import IndexManager

        if embedding_provider == "openai":
            from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
            embed_provider = OpenAIEmbeddingProvider(
                model=embedding_model, dimensions=int(vector_dimension)
            )
        else:
            from rag.infra.embedding.multilingual_embedding import MultilingualEmbeddingProvider
            embed_provider = MultilingualEmbeddingProvider(
                model=embedding_model, dim=int(vector_dimension)
            )

        faiss_path = Path(index_dir) / "faiss.index"
        if faiss_path.exists():
            try:
                existing = FaissLocalIndex()
                existing.load(index_dir)
                if existing.dimension != int(vector_dimension):
                    raise ValueError(
                        f"Dimension mismatch: existing index={existing.dimension}, "
                        f"configured={int(vector_dimension)}"
                    )
            except ValueError:
                raise
            except Exception:
                pass

        mgr = IndexManager(index_dir=index_dir)
        vec_index = mgr.faiss
        kw_index = mgr.bm25

    pipeline = IngestPipeline(
        doc_store=doc_store,
        trace_store=trace_store,
        token_budget=int(token_budget),
        embedding_provider=embed_provider,
        vector_index=vec_index,
        keyword_index=kw_index,
        vision_model=vision_model,
    )
    return pipeline, vec_index, kw_index


# ---------------------------------------------------------------------------
# Section 1 — Ingest (file upload + URL tabs)
# ---------------------------------------------------------------------------

st.subheader("1️⃣ Ingest Documents")

ingest_tab_files, ingest_tab_urls = st.tabs(["📁 Upload Files", "🌐 From URLs"])

with ingest_tab_files:
    uploaded_files = st.file_uploader(
        "Choose files to ingest",
        type=["txt", "md", "pdf", "html", "htm", "docx", "xlsx"],
        accept_multiple_files=True,
        help="Supported: plain text, Markdown, PDF, HTML, Word, Excel. Select multiple for batch ingest.",
    )
    run_btn = st.button(
        "▶ Run Ingest", type="primary",
        disabled=(not uploaded_files), key="run_files",
    )

with ingest_tab_urls:
    st.caption(
        "One URL per line. GitHub blob URLs are auto-converted to raw content. "
        "Supports any public web page, blog post, or documentation site."
    )
    url_input = st.text_area(
        "URLs to ingest",
        placeholder=(
            "https://en.wikipedia.org/wiki/Retrieval-augmented_generation\n"
            "https://github.com/user/repo/blob/main/README.md\n"
            "https://docs.llamaindex.ai/en/stable/"
        ),
        height=150,
    )
    run_url_btn = st.button(
        "▶ Fetch & Ingest", type="primary",
        disabled=(not url_input.strip()), key="run_urls",
    )


# ---------------------------------------------------------------------------
# Ingest handler — files
# ---------------------------------------------------------------------------

if run_btn and uploaded_files:
    st.session_state.ingest_result = None
    st.session_state.ingest_error = None
    st.session_state.last_doc_ids = []
    st.session_state.ingest_config = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model if embedding_provider != "none" else "—",
        "vector_dimension": int(vector_dimension) if embedding_provider != "none" else "—",
        "index_type": index_type,
        "token_budget": int(token_budget),
        "collection": collection,
        "db_path": db_path,
        "index_dir": index_dir,
        "vision_model": vision_model,
    }
    try:
        pipeline, vec_index, kw_index = _build_pipeline(
            db_path, index_dir, token_budget, embedding_provider,
            embedding_model, vector_dimension, index_type,
            vision_model=vision_model,
        )
        results, original_names, doc_ids = [], [], []
        progress = st.progress(0, text="Starting…")
        for i, uf in enumerate(uploaded_files):
            progress.progress(
                i / len(uploaded_files),
                text=f"Ingesting **{uf.name}** ({i + 1}/{len(uploaded_files)})…",
            )
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name
            result = pipeline.ingest(tmp_path, canonical_name=uf.name)
            results.append(result)
            original_names.append(uf.name)
            if result.doc_id:
                doc_ids.append(result.doc_id)
        progress.progress(1.0, text="Done!")
        if vec_index is not None:
            vec_index.save(index_dir)
        if kw_index is not None:
            kw_index.save(index_dir)
        st.session_state.ingest_result = list(zip(original_names, results))
        st.session_state.last_doc_ids = doc_ids
    except Exception as exc:
        st.session_state.ingest_error = f"❌ Ingestion failed: {exc}"
    st.rerun()


# ---------------------------------------------------------------------------
# Ingest handler — URLs
# ---------------------------------------------------------------------------

if run_url_btn and url_input.strip():
    urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
    st.session_state.ingest_result = None
    st.session_state.ingest_error = None
    st.session_state.last_doc_ids = []
    st.session_state.ingest_config = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model if embedding_provider != "none" else "—",
        "vector_dimension": int(vector_dimension) if embedding_provider != "none" else "—",
        "index_type": index_type,
        "token_budget": int(token_budget),
        "collection": collection,
        "db_path": db_path,
        "index_dir": index_dir,
        "vision_model": vision_model,
    }
    try:
        from rag.infra.connectors.web_connector import WebConnector
        pipeline, vec_index, kw_index = _build_pipeline(
            db_path, index_dir, token_budget, embedding_provider,
            embedding_model, vector_dimension, index_type,
            vision_model=vision_model,
        )
        progress = st.progress(0, text=f"Fetching {len(urls)} URL(s)…")
        artifacts = WebConnector().fetch(urls)
        if not artifacts:
            st.session_state.ingest_error = "❌ No content could be fetched from the provided URLs."
            st.rerun()
        results, original_names, doc_ids = [], [], []
        for i, artifact in enumerate(artifacts):
            progress.progress(
                i / len(artifacts),
                text=f"Ingesting **{artifact.canonical_name}** ({i + 1}/{len(artifacts)})…",
            )
            result = pipeline.ingest(artifact.tmp_path, canonical_name=artifact.canonical_name)
            results.append(result)
            original_names.append(artifact.canonical_name)
            if result.doc_id:
                doc_ids.append(result.doc_id)
            artifact.cleanup()
        progress.progress(1.0, text="Done!")
        if vec_index is not None:
            vec_index.save(index_dir)
        if kw_index is not None:
            kw_index.save(index_dir)
        st.session_state.ingest_result = list(zip(original_names, results))
        st.session_state.last_doc_ids = doc_ids
    except Exception as exc:
        st.session_state.ingest_error = f"❌ URL ingestion failed: {exc}"
    st.rerun()


# ---------------------------------------------------------------------------
# Ingest result display
# ---------------------------------------------------------------------------

if st.session_state.ingest_error:
    st.error(st.session_state.ingest_error)

results_raw = st.session_state.ingest_result
if results_raw:
    named_results = results_raw if isinstance(results_raw[0], tuple) else [
        (Path(r.source_path).name, r) for _, r in results_raw
    ]
    results = [r for _, r in named_results]

    def _status_of(r) -> str:
        # Older stored results may predate the status field
        return getattr(r, "status", "failed" if r.error else ("skipped" if r.skipped else "ok"))

    n_ok = sum(1 for r in results if _status_of(r) == "ok")
    n_skipped = sum(1 for r in results if _status_of(r) == "skipped")
    n_degraded = sum(
        1 for r in results if _status_of(r) in ("quality_failed", "empty_content")
    )
    n_err = sum(1 for r in results if _status_of(r) in ("failed", "unsupported"))
    total = len(results)

    if n_err == 0 and n_degraded == 0:
        st.success(
            f"✅ Ingested **{n_ok}** document(s)"
            + (f", ⏭️ skipped **{n_skipped}** (already up-to-date)" if n_skipped else "")
            + f" — {total} file(s) processed."
        )
    else:
        st.warning(
            f"⚠️ {n_ok} succeeded, {n_skipped} skipped, {n_degraded} degraded, "
            f"{n_err} failed out of {total}."
        )

    total_cost = sum(getattr(r, "cost_usd", 0.0) for r in results)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total blocks", sum(r.block_count for r in results))
    c2.metric("Total chunks", sum(r.chunk_count for r in results))
    c3.metric("Total elapsed", f"{sum(r.elapsed_ms for r in results):.0f} ms")
    c4.metric("Total embed tokens", sum(r.embed_tokens for r in results))
    c5.metric("Est. cost", f"${total_cost:.4f}")

    if total > 1:
        for fname, result in named_results:
            status = _status_of(result)
            if status in ("failed", "unsupported"):
                st.error(f"❌ **{fname}** — {result.error}")
            elif status == "skipped":
                st.warning(f"⏭️ **{fname}** — already up-to-date.")
            elif status == "empty_content":
                st.error(
                    f"🕳️ **{fname}** — parsed but produced 0 chunks; "
                    "it will be invisible to retrieval."
                )
            elif status == "quality_failed":
                st.warning(
                    f"⚠️ **{fname}** — ingested, but failed quality gates "
                    f"({result.chunk_count} chunks, content may be degraded)."
                )
            else:
                with st.expander(f"✅ {fname}  ·  {result.chunk_count} chunks  ·  {result.elapsed_ms:.0f} ms"):
                    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                    cc1.metric("Blocks", result.block_count)
                    cc2.metric("Chunks", result.chunk_count)
                    cc3.metric("Elapsed", f"{result.elapsed_ms:.0f} ms")
                    cc4.metric("Embed tokens", result.embed_tokens)
                    cc5.metric("Est. cost", f"${getattr(result, 'cost_usd', 0.0):.4f}")

                    stage_timings = getattr(result, "stage_timings_ms", None)
                    if stage_timings:
                        st.caption(
                            "Stage timings: "
                            + " · ".join(
                                f"{stage}={ms:.0f}ms" for stage, ms in stage_timings.items()
                            )
                        )

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Chunk Browser + Traces (tabs)
# ---------------------------------------------------------------------------

st.subheader("2️⃣ Inspect")

inspect_tab_chunks, inspect_tab_traces = st.tabs(["🗂️ Chunk Browser", "🔬 Ingestion Traces"])


# ── Chunk Browser ────────────────────────────────────────────────────────────

with inspect_tab_chunks:

    @st.cache_data(ttl=10)
    def _load_chunks(db: str):
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        store = SQLiteDocStore(db)
        return store.list_documents(), store.get_all_chunks()

    if not Path(db_path).exists():
        st.info("No database yet — ingest a document first.")
    else:
        try:
            docs, chunks = _load_chunks(db_path)
        except Exception as e:
            st.error(f"Failed to load chunks: {e}")
            chunks, docs = [], []

        if not chunks:
            st.info("No chunks found. Ingest a document above to populate the index.")
        else:
            # Document selector — default to last ingested doc if available
            doc_options = {"All documents": None}
            for doc in docs:
                label = Path(doc["source_path"]).name or doc["doc_id"][:16]
                doc_options[f"{label}  —  {doc['doc_id'][:12]}…"] = doc["doc_id"]

            # Pre-select the most recently ingested document
            default_idx = 0
            last_ids = st.session_state.get("last_doc_ids", [])
            if last_ids:
                for idx, (label, did) in enumerate(doc_options.items()):
                    if did in last_ids:
                        default_idx = idx
                        break

            selected_label = st.selectbox(
                "Document", list(doc_options.keys()), index=default_idx
            )
            selected_doc_id = doc_options.get(selected_label)

            # Filter + sort
            filtered = [c for c in chunks if selected_doc_id is None or c.doc_id == selected_doc_id]
            if search_term.strip():
                q = search_term.strip().lower()
                filtered = [c for c in filtered if
                            q in (c.display_text or "").lower() or
                            q in (c.stable_text or "").lower()]
            if sort_by == "Token count ↑":
                filtered.sort(key=lambda c: c.token_count)
            elif sort_by == "Token count ↓":
                filtered.sort(key=lambda c: c.token_count, reverse=True)

            # Stats bar
            if filtered:
                tok = [c.token_count for c in filtered]
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Chunks", len(filtered))
                s2.metric("Documents", len({c.doc_id for c in filtered}))
                s3.metric("Avg tokens", f"{sum(tok)/len(tok):.0f}")
                s4.metric("Min tokens", min(tok))
                s5.metric("Max tokens", max(tok))

                buckets = {"0–128": 0, "129–256": 0, "257–384": 0, "385–512": 0, "512+": 0}
                for t in tok:
                    if t <= 128: buckets["0–128"] += 1
                    elif t <= 256: buckets["129–256"] += 1
                    elif t <= 384: buckets["257–384"] += 1
                    elif t <= 512: buckets["385–512"] += 1
                    else: buckets["512+"] += 1
                with st.expander("📊 Token distribution", expanded=False):
                    import pandas as pd
                    df = pd.DataFrame({"Range": list(buckets.keys()), "Count": list(buckets.values())})
                    st.bar_chart(df.set_index("Range"))

            st.markdown(f"**🗂️ Chunks ({len(filtered)})**")
            if not filtered:
                st.info("No chunks match the current filters.")
            else:
                for i, chunk in enumerate(filtered, start=1):
                    doc_name = Path(
                        next((d["source_path"] for d in docs if d["doc_id"] == chunk.doc_id), chunk.doc_id)
                    ).name or chunk.doc_id[:12]
                    t = chunk.token_count
                    badge = f"🟢 {t} tok" if t <= 256 else (f"🟡 {t} tok" if t <= 450 else f"🔴 {t} tok")
                    label = f"**#{i}** · `{(chunk.chunk_id or '')[:12]}…` · 📄 {doc_name} · {badge}"
                    with st.expander(label, expanded=False):
                        col_meta, col_text = st.columns([1, 2])
                        with col_meta:
                            st.markdown("**Chunk ID**")
                            st.code(chunk.chunk_id or "—", language=None)
                            st.markdown("**Doc ID**")
                            st.code(chunk.doc_id[:32] + "…", language=None)
                            st.markdown(f"**Tokens:** `{chunk.token_count}`")
                            if chunk.metadata:
                                pages = chunk.metadata.get("start_page") or chunk.metadata.get("page")
                                if pages:
                                    st.markdown(f"**Page:** `{pages}`")
                                st.json(chunk.metadata, expanded=False)
                        with col_text:
                            st.markdown("**Chunk text:**")
                            st.markdown(
                                f"""<div style="background:#f8f9fa;border-left:3px solid #dee2e6;
                                padding:10px 14px;border-radius:4px;font-size:0.9em;
                                white-space:pre-wrap;word-break:break-word;">
                                {chunk.display_text or chunk.stable_text}</div>""",
                                unsafe_allow_html=True,
                            )


# ── Ingestion Traces ─────────────────────────────────────────────────────────

with inspect_tab_traces:

    @st.cache_data(ttl=10)
    def _load_runs(db: str, limit: int) -> list[dict]:
        try:
            from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
            runs = SQLiteTraceStore(db).list_runs(limit=int(limit))
            return [r for r in runs if r["run_type"] in (
                "ingest", "ingest_start", "ingest_complete", "ingest_error",
            )]
        except Exception:
            return []

    @st.cache_data(ttl=10)
    def _load_events(db: str, run_id: str) -> list[dict]:
        try:
            from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
            all_runs = SQLiteTraceStore(db).list_runs(limit=500)
            return [r for r in all_runs if
                    r["run_id"] == run_id or
                    r.get("metadata", {}).get("run_id") == run_id]
        except Exception:
            return []

    if not Path(db_path).exists():
        st.info("No database yet — ingest a document first.")
    else:
        runs = _load_runs(db_path, int(max_runs))
        if not runs:
            st.info("No ingestion traces found. Ingest a document above.")
        else:
            trace_browse, trace_detail = st.tabs(["📋 Browse Runs", "🔍 Run Detail"])

            with trace_browse:
                st.markdown(f"**{len(runs)}** recent ingestion run(s).")
                from rag.app.studio.components.trace_viewer import render_run_summary_table
                render_run_summary_table(runs)

            with trace_detail:
                from rag.app.studio.components.trace_viewer import render_run_selector, render_run_events
                selected_run_id = render_run_selector(runs, label="Select ingestion run")
                if selected_run_id:
                    st.markdown(f"**Run ID:** `{selected_run_id}`")
                    st.divider()
                    render_run_events(_load_events(db_path, selected_run_id))


# ---------------------------------------------------------------------------
# Pipeline overview (always visible at bottom)
# ---------------------------------------------------------------------------

with st.expander("📖 Pipeline overview", expanded=False):
    st.markdown(
        """
        1. **Loader** — read raw bytes from file or URL
        2. **Sniffer** — detect MIME type via magic bytes + extension
        3. **Router → Parser** — dispatch to MdParser / HtmlTrafilaturaParser / PdfPyMuPDFParser
        4. **Quality Gate** — filter documents below minimum quality thresholds
        5. **Cleaner** — unicode fix, empty filter, OCR merge, deduplication
        6. **Block Splitter** — map IRBlocks to hashed TextBlocks
        7. **Chunk Packer** — anchor-aware packing with token budget
        8. **Embedding** — embed chunks via configured provider
        9. **DocStore + TraceStore** — persist document, chunks, and run trace
        """
    )
