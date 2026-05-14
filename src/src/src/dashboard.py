import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
sys.path.insert(0, "src")

from drift import compute_wkcs
from attribution import attribute_drift
from windower import create_windows

st.set_page_config(page_title="SemDriftBD", layout="wide", page_icon="📡")

st.title("📡 SemDriftBD")
st.markdown("**Distributed Semantic Drift Detection in Large-Scale Text Streams**")
st.markdown("---")

# Load drift scores
@st.cache_data
def load_drift_scores():
    return pd.read_csv("data/drift_scores.csv", parse_dates=["window_start"])

@st.cache_data
def load_baseline():
    return pd.read_csv("data/baseline_comparison.csv")

@st.cache_data
def load_windows():
    return create_windows("data/raw/corpus.parquet")

@st.cache_resource
def load_embeddings():
    files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    windows = []
    for f in files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            windows.append(pickle.load(fp))
    return windows

df = load_drift_scores()
baseline_df = load_baseline()

# --- Row 1: Key metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Windows", len(df) + 1)
col2.metric("Mean WKCS", f"{df['wkcs'].mean():.3f}")
col3.metric("Peak Drift", f"{df['wkcs'].max():.3f}")
col4.metric("Peak Drift Window", str(df.loc[df['wkcs'].idxmax(), 'window_start'].date()))

st.markdown("---")

# --- Row 2: WKCS over time ---
st.subheader("📈 Semantic Drift Over Time (WKCS Score)")

chart_df = df[["window_start", "wkcs", "wasserstein", "kl_divergence"]].copy()
chart_df = chart_df.rename(columns={
    "window_start": "Date",
    "wkcs": "WKCS",
    "wasserstein": "Wasserstein",
    "kl_divergence": "KL Divergence"
})
chart_df = chart_df.set_index("Date")

metric_choice = st.radio("Show metric:", ["WKCS", "Wasserstein", "KL Divergence"], horizontal=True)
st.line_chart(chart_df[[metric_choice]], height=280)

# Highlight pea
peak_row = df.loc[df['wkcs'].idxmax()]
st.info(f"🔺 Peak drift detected: **{peak_row['window_start'].date()} → {pd.Timestamp(peak_row['next_window_start']).date()}** — WKCS: **{peak_row['wkcs']:.4f}**  \nCorresponds to: Las Vegas shooting (Oct 1, 2017) — dominant topic shift detected.")
st.markdown("---")

# --- Row 3: Baseline comparison ---
st.subheader("📊 WKCS vs Baseline Methods")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Correlation with WKCS**")
    corr_data = {
        "Method": ["Cosine Distance", "Wasserstein-only", "KL-only", "WKCS (ours)"],
        "Correlation": [
            round(baseline_df["cosine"].corr(baseline_df["wkcs"]), 4),
            round(baseline_df["wasserstein_only"].corr(baseline_df["wkcs"]), 4),
            round(baseline_df["kl_only"].corr(baseline_df["wkcs"]), 4),
            1.0
        ]
    }
    corr_df = pd.DataFrame(corr_data)
    st.dataframe(corr_df, hide_index=True, use_container_width=True)

with col2:
    st.markdown("**Per-pair comparison (all 25 pairs)**")
    plot_df = baseline_df[["pair", "cosine", "wasserstein_only", "kl_only", "wkcs"]].set_index("pair")
    st.line_chart(plot_df[["cosine", "wkcs"]], height=200)

st.markdown("---")

# --- Row 4: Interactive attribution ---
st.subheader("🔍 Causal Drift Attribution")
st.markdown("Select a window pair to see which topics drove the drift.")

pair_options = [f"Pair {row['pair']:02d}: {pd.Timestamp(row['window_start']).date()} → {pd.Timestamp(row['next_window_start']).date()} (WKCS: {row['wkcs']:.3f})" 
                for _, row in df.iterrows()]

selected = st.selectbox("Choose window pair:", pair_options, index=int(df['wkcs'].idxmax()))
pair_idx = int(selected.split(":")[0].replace("Pair ", "").strip()) - 1

if st.button("Run Attribution Analysis"):
    with st.spinner("Running BERTopic attribution..."):
        raw_windows = load_windows()
        emb_windows = load_embeddings()

        w1 = {"texts": raw_windows[pair_idx]["texts"], "embeddings": emb_windows[pair_idx]["embeddings"]}
        w2 = {"texts": raw_windows[pair_idx+1]["texts"], "embeddings": emb_windows[pair_idx+1]["embeddings"]}

        result = attribute_drift(w1, w2)

        if result["topics"]:
            st.success(f"Primary drift driver: **{result['top_driver']}**")
            topic_df = pd.DataFrame(result["topics"])[
                ["topic_words", "drift_contribution_pct", "wkcs", "w1_articles", "w2_articles"]
            ].rename(columns={
                "topic_words": "Topic Keywords",
                "drift_contribution_pct": "Drift Contribution %",
                "wkcs": "Topic WKCS",
                "w1_articles": "Articles (Window 1)",
                "w2_articles": "Articles (Window 2)"
            })
            st.dataframe(topic_df, hide_index=True, use_container_width=True)
            st.bar_chart(topic_df.set_index("Topic Keywords")["Drift Contribution %"])
        else:
            st.warning("Not enough per-topic articles for attribution on this pair.")

st.markdown("---")
st.caption("SemDriftBD | WKCS: Wasserstein-KL Composite Score | Built with Spark + Kafka + sentence-transformers")
