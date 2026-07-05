"""Connector Sync page — configure, healthcheck, and sync external sources.

Supported connectors:
- Email   (IMAP/Gmail — reads env vars RAG_EMAIL_*)
- Slack   (Web API   — reads env vars RAG_SLACK_*)
- Notion  (API       — reads env vars RAG_NOTION_*)
- Google Docs (Drive/Docs API — reads env vars RAG_GOOGLE_*)
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Connectors · RAG Studio", layout="wide")

st.title("🔌 Connector Sync")
st.caption(
    "Connect external sources, verify credentials, and pull new content into the corpus."
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB    = "data/rag.db"
_DEFAULT_INDEX = "data/indexes"

# Env var definitions per connector
_CONNECTOR_ENV = {
    "email": [
        {"key": "RAG_EMAIL_SERVER",   "required": True,  "desc": "IMAP hostname (e.g. imap.gmail.com)"},
        {"key": "RAG_EMAIL_USER",     "required": True,  "desc": "Mailbox username / email address"},
        {"key": "RAG_EMAIL_PASSWORD", "required": True,  "desc": "Password or app-specific password", "secret": True},
        {"key": "RAG_EMAIL_PORT",     "required": False, "desc": "IMAP port (default 993)"},
        {"key": "RAG_EMAIL_MAILBOX",  "required": False, "desc": "Folder to sync (default INBOX)"},
    ],
    "slack": [
        {"key": "RAG_SLACK_TOKEN",    "required": True,  "desc": "Slack Bot token (xoxb-…)", "secret": True},
        {"key": "RAG_SLACK_CHANNELS", "required": False, "desc": "Comma-separated channel IDs to sync"},
    ],
    "notion": [
        {"key": "RAG_NOTION_TOKEN",        "required": True,  "desc": "Notion integration token", "secret": True},
        {"key": "RAG_NOTION_DATABASE_IDS", "required": False, "desc": "Comma-separated database IDs (optional)"},
    ],
    "google_docs": [
        {"key": "RAG_GOOGLE_SERVICE_ACCOUNT_JSON", "required": True,  "desc": "Path to service account JSON file"},
        {"key": "RAG_GOOGLE_DRIVE_FOLDER_ID",      "required": False, "desc": "Drive folder ID to scope search (optional)"},
    ],
}

_CONNECTOR_LABELS = {
    "email":       "📧 Email (IMAP)",
    "slack":       "💬 Slack",
    "notion":      "📓 Notion",
    "google_docs": "📄 Google Docs",
}

_CONNECTOR_REGISTRY = {
    "email":       "rag.infra.connectors.email_connector.EmailConnector",
    "slack":       "rag.infra.connectors.slack_connector.SlackConnector",
    "notion":      "rag.infra.connectors.notion_connector.NotionConnector",
    "google_docs": "rag.infra.connectors.google_docs_connector.GoogleDocsConnector",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_status_table(connector: str) -> None:
    """Render a status table of env vars for the connector."""
    rows = _CONNECTOR_ENV[connector]
    all_required_set = all(
        bool(os.environ.get(r["key"]))
        for r in rows if r["required"]
    )

    data = []
    for r in rows:
        val = os.environ.get(r["key"], "")
        if val and r.get("secret"):
            display = "●●●●●●●● (set)"
        elif val:
            display = val
        else:
            display = "—  (not set)"
        status = "✅" if val else ("❌ required" if r["required"] else "ℹ️ optional")
        data.append({"Env var": r["key"], "Status": status, "Value": display, "Description": r["desc"]})

    st.dataframe(data, use_container_width=True, hide_index=True)
    if not all_required_set:
        st.warning(
            "One or more **required** env vars are missing.  "
            "Set them in your `.env` file and restart Streamlit."
        )
    return all_required_set


def _load_connector(name: str):
    import importlib
    dotted = _CONNECTOR_REGISTRY[name]
    mod_path, cls_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def _render_connector_tab(connector: str) -> None:
    """Render the full UI for a single connector."""
    st.markdown(f"#### Credentials")
    ready = _env_status_table(connector)

    st.divider()

    # ── Healthcheck ──────────────────────────────────────────────────────────
    st.markdown("#### Connection Test")
    if st.button("🔍 Test Connection", key=f"hc_{connector}", disabled=not ready):
        with st.spinner("Testing…"):
            try:
                conn   = _load_connector(connector)
                result = conn.healthcheck()
                status = result.get("status", "unknown")
                detail = result.get("detail", "")
                if status == "ok":
                    st.success(f"✅ Connected successfully.  {detail}")
                elif status == "degraded":
                    st.warning(f"⚠️ Connected but degraded: {detail}")
                else:
                    st.error(f"❌ Connection failed: {detail}")
            except Exception as exc:
                st.error(f"❌ Exception during healthcheck: {exc}")

    st.divider()

    # ── Sync ─────────────────────────────────────────────────────────────────
    st.markdown("#### Sync Now")

    with st.form(f"sync_form_{connector}"):
        sc_l, sc_r = st.columns(2)
        with sc_l:
            db_path   = st.text_input("Database path",  value=_DEFAULT_DB,    key=f"db_{connector}")
            index_dir = st.text_input("Index directory", value=_DEFAULT_INDEX, key=f"idx_{connector}")
        with sc_r:
            cursor_override = st.text_input(
                "Since cursor (optional)",
                value="",
                key=f"cursor_{connector}",
                help=(
                    "Override the saved cursor.  Leave blank to continue from last sync.  "
                    "Use an empty string to re-sync from the beginning."
                ),
            )
            emb_provider = st.selectbox(
                "Embedding provider",
                ["openai", "multilingual", "none"],
                index=0,
                key=f"emb_{connector}",
            )

        sync_submitted = st.form_submit_button(
            "▶ Sync Now",
            disabled=not ready,
            use_container_width=True,
        )

    if sync_submitted:
        with st.spinner(f"Syncing {_CONNECTOR_LABELS[connector]}…"):
            try:
                from rag.infra.stores.docstore_sqlite import SQLiteDocStore
                from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
                from rag.pipelines.connector_sync_pipeline import ConnectorSyncPipeline
                from rag.app.mcp_server.wiring import build_ingest_pipeline

                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                Path(index_dir).mkdir(parents=True, exist_ok=True)

                ingest_pipeline, index_mgr = build_ingest_pipeline(
                    db_path=db_path,
                    index_dir=index_dir,
                    embedding_provider=emb_provider if emb_provider != "none" else None,
                )
                doc_store   = SQLiteDocStore(db_path)
                trace_store = SQLiteTraceStore(db_path)
                conn        = _load_connector(connector)

                pipeline = ConnectorSyncPipeline(
                    connector=conn,
                    ingest_pipeline=ingest_pipeline,
                    doc_store=doc_store,
                    trace_store=trace_store,
                )

                since = cursor_override.strip() if cursor_override.strip() else None
                result = pipeline.run(since_cursor=since)
                # Persist the same IndexManager instance the sync actually
                # mutated — constructing a fresh IndexManager here would
                # reload a stale, pre-sync snapshot instead.
                index_mgr.save()

                # Display results
                st.success("✅ Sync complete")
                r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                r_col1.metric("Fetched",   result.fetched)
                r_col2.metric("Ingested",  result.ingested)
                r_col3.metric("Skipped",   result.skipped)
                r_col4.metric("Failed",    result.failed)
                st.caption(
                    f"Cursor before: `{result.cursor_before or '(start)'}` → "
                    f"after: `{result.cursor_after or '(unchanged)'}`  |  "
                    f"Elapsed: {result.elapsed_ms:.0f} ms  |  "
                    f"Run ID: `{result.run_id}`"
                )
                if result.error:
                    st.error(f"Sync error: {result.error}")

            except Exception as exc:
                st.error(f"❌ Sync failed: {exc}")


# ---------------------------------------------------------------------------
# Page layout — one tab per connector
# ---------------------------------------------------------------------------

st.divider()

tab_email, tab_slack, tab_notion, tab_gdocs = st.tabs([
    _CONNECTOR_LABELS["email"],
    _CONNECTOR_LABELS["slack"],
    _CONNECTOR_LABELS["notion"],
    _CONNECTOR_LABELS["google_docs"],
])

with tab_email:
    _render_connector_tab("email")

with tab_slack:
    _render_connector_tab("slack")

with tab_notion:
    _render_connector_tab("notion")

with tab_gdocs:
    _render_connector_tab("google_docs")
