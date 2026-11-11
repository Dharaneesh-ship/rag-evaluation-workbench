"""Experiment Comparison — historical RAG experiment dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Experiment Comparison · RAG Studio",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "outputs" / "experiments"


@st.cache_data
def load_experiments():
    records = []

    if not RESULTS_DIR.exists():
        return records

    for path in sorted(RESULTS_DIR.glob("experiment_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for result in payload.get("results", []):
                if result.get("status") != "completed":
                    continue

                records.append({
                    "run": path.stem,
                    "created_at": payload.get("created_at_utc", ""),
                    "experiment": result.get("experiment", "Unnamed"),
                    "num_queries": result.get("num_queries", 0),
                    "recall_at_k": result.get("recall_at_k"),
                    "mrr": result.get("mrr"),
                    "ndcg_at_k": result.get("ndcg_at_k"),
                    "latency_ms": result.get("mean_query_latency_ms"),
                    "settings": result.get("settings", {}),
                    "k": result.get("k"),
                    "source": path.name,
                })
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            st.warning(f"Could not load {path.name}: {exc}")

    return records


st.title("📊 Experiment Comparison")
st.caption("Compare retrieval configurations, evaluation metrics and execution efficiency.")

records = load_experiments()

if not records:
    st.info(
        "No completed experiment results found. "
        "Run the experiment runner to populate outputs/experiments."
    )
    st.stop()

df = pd.DataFrame(records)

# Sidebar filters
st.sidebar.header("Experiment filters")

available_runs = sorted(df["run"].unique(), reverse=True)
selected_runs = st.sidebar.multiselect(
    "Experiment runs",
    available_runs,
    default=available_runs,
)

available_experiments = sorted(
    df[df["run"].isin(selected_runs)]["experiment"].unique()
)

selected_experiments = st.sidebar.multiselect(
    "Configurations",
    available_experiments,
    default=available_experiments,
)

filtered = df[
    df["run"].isin(selected_runs)
    & df["experiment"].isin(selected_experiments)
].copy()

if filtered.empty:
    st.warning("Select at least one experiment run and configuration.")
    st.stop()

# Summary metrics
st.subheader("Performance overview")

latest_run = filtered["run"].iloc[0]
latest = filtered[filtered["run"] == latest_run]

best_recall = latest.loc[latest["recall_at_k"].idxmax()]
best_ndcg = latest.loc[latest["ndcg_at_k"].idxmax()]
fastest = latest.loc[latest["latency_ms"].idxmin()]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Configurations", len(latest))
c2.metric("Best Recall@K", f"{best_recall['recall_at_k']:.3f}")
c3.metric("Best nDCG@K", f"{best_ndcg['ndcg_at_k']:.3f}")
c4.metric("Lowest latency", f"{fastest['latency_ms']:.2f} ms")

st.caption(f"Summary based on latest selected run: {latest_run}")

# Metric comparisons
st.divider()
st.subheader("Retrieval quality")

metric_tabs = st.tabs(["Recall@K", "MRR", "nDCG@K", "Latency"])

with metric_tabs[0]:
    chart = filtered.pivot_table(
        index="experiment", columns="run",
        values="recall_at_k", aggfunc="mean"
    )
    st.bar_chart(chart, y_label="Recall")

with metric_tabs[1]:
    chart = filtered.pivot_table(
        index="experiment", columns="run",
        values="mrr", aggfunc="mean"
    )
    st.bar_chart(chart, y_label="Mean Reciprocal Rank")

with metric_tabs[2]:
    chart = filtered.pivot_table(
        index="experiment", columns="run",
        values="ndcg_at_k", aggfunc="mean"
    )
    st.bar_chart(chart, y_label="nDCG")

with metric_tabs[3]:
    chart = filtered.pivot_table(
        index="experiment", columns="run",
        values="latency_ms", aggfunc="mean"
    )
    st.bar_chart(chart, y_label="Latency (ms)")

# Detailed comparison
st.divider()
st.subheader("Detailed results")

display_df = filtered[
    [
        "run", "experiment", "num_queries",
        "recall_at_k", "mrr", "ndcg_at_k",
        "latency_ms",
    ]
].rename(columns={
    "run": "Run",
    "experiment": "Configuration",
    "num_queries": "Queries",
    "recall_at_k": "Recall@K",
    "mrr": "MRR",
    "ndcg_at_k": "nDCG@K",
    "latency_ms": "Latency (ms)",
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download comparison CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="rag_experiment_comparison.csv",
    mime="text/csv",
)

# Configuration details
st.divider()
st.subheader("Experiment configurations")

for _, row in filtered.iterrows():
    with st.expander(f"{row['experiment']} — {row['run']}"):
        st.json(row["settings"])
        st.write(f"Evaluation queries: {row['num_queries']}")
        st.write(f"Metric cutoff: {row['k']}")
        st.write(f"Source file: {row['source']}")

st.caption("RAG Evaluation & Optimization Workbench · Experiment Analytics")
