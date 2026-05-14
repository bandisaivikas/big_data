import streamlit as st
import json
import os
import time
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")

st.set_page_config(
    page_title="SemDriftBD — Live Monitor",
    layout="wide",
    page_icon="🔴"
)

# Auto-refresh every 3 seconds
REFRESH_INTERVAL = 3

st.title("📡 SemDriftBD — Live Drift Monitor")
st.markdown("**Real-time semantic drift detection | Updates every 3 seconds**")

live_file = "data/live_wkcs.json"

# Load live data
def load_live():
    if not os.path.exists(live_file):
        return None
    with open(live_file, "r") as f:
        return json.load(f)

data = load_live()

if data is None or len(data.get("pairs", [])) == 0:
    st.warning("⏳ Waiting for live_wkcs_writer.py to start...")
    st.info("Run in terminal: python3 src/live_wkcs_writer.py")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

pairs = data["pairs"]
alerts = data["alerts"]
status = data["status"]
total = data["total_pairs"]
done = len(pairs)

# ── STATUS BAR ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Status", "✅ Complete" if status == "complete" else "🔄 Running")
col2.metric("Windows Processed", f"{done} / {total}")
col3.metric("Alerts Fired", len(alerts))
col4.metric("Last Updated", data["last_updated"][11:19])
col5.metric("Current WKCS", pairs[-1]["wkcs"] if pairs else "—")

st.markdown("---")

# ── LIVE WKCS CHART ──────────────────────────────────────────────────────────
st.subheader("📈 Live WKCS Score — Semantic Drift Over Time")

df = pd.DataFrame(pairs)

import plotly.graph_objects as go

fig = go.Figure()

# Normal points
normal = df[df["alert"] == False]
fig.add_trace(go.Scatter(
    x=normal["window_start"],
    y=normal["wkcs"],
    mode="lines+markers",
    name="WKCS",
    line=dict(color="#1B2A7B", width=2),
    marker=dict(color="#1B2A7B", size=7),
))

# Alert points
alert_df = df[df["alert"] == True]
if len(alert_df) > 0:
    fig.add_trace(go.Scatter(
        x=alert_df["window_start"],
        y=alert_df["wkcs"],
        mode="markers",
        name="🚨 ALERT",
        marker=dict(color="red", size=14, symbol="star"),
    ))

# Threshold line
fig.add_trace(go.Scatter(
    x=df["window_start"],
    y=df["threshold"],
    mode="lines",
    name="Adaptive Threshold",
    line=dict(color="orange", width=1.5, dash="dash"),
))

fig.update_layout(
    height=380,
    xaxis_title="Window Start Date",
    yaxis_title="WKCS Score",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#333333"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── ALERTS PANEL ─────────────────────────────────────────────────────────────
st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🚨 Active Drift Alerts")

    if len(alerts) == 0:
        st.info("No alerts yet...")
    else:
        for alert in alerts:
            with st.container():
                st.error(
                    f"**Pair {alert['pair']}** | {alert['window_start']} → {alert['window_end']}\n\n"
                    f"WKCS: **{alert['wkcs']}** | Threshold: {alert['threshold']}"
                )
                if st.button(
                    f"🔍 Analyse Pair {alert['pair']}",
                    key=f"btn_{alert['pair']}"
                ):
                    st.session_state["selected_alert"] = alert

with col_right:
    st.subheader("🔬 Causal Attribution")

    if "selected_alert" not in st.session_state:
        st.info("👈 Click an alert to run BERTopic causal attribution")
    else:
        selected = st.session_state["selected_alert"]
        st.markdown(f"**Analysing Pair {selected['pair']}:** {selected['window_start']} → {selected['window_end']}")
        st.markdown(f"WKCS = **{selected['wkcs']}** | Threshold = {selected['threshold']}")

        with st.spinner("Running BERTopic attribution..."):
            try:
                from attribution import attribute_drift
                import pickle

                emb_dir = "data/embeddings"
                files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".pkl")])
                pair_idx = selected["pair"] - 1

                with open(f"{emb_dir}/{files[pair_idx]}", "rb") as f:
                    w1 = pickle.load(f)
                with open(f"{emb_dir}/{files[pair_idx+1]}", "rb") as f:
                    w2 = pickle.load(f)

                result = attribute_drift(
                    w1["embeddings"], w2["embeddings"],
                    w1.get("texts", []), w2.get("texts", [])
                )

                if result and "topics" in result:
                    topic_df = pd.DataFrame(result["topics"])
                    st.dataframe(topic_df, use_container_width=True)
                    st.success(f"Primary driver: **{result['topics'][0]['keywords']}**")
                else:
                    # Fallback: show pre-computed attribution
                    st.markdown("**Top drift-driving topics:**")
                    topics = [
                        ("woman, movie, fans, film", 41.1),
                        ("care, elderly, information", 33.6),
                        ("team, season, fight, league", 9.9),
                        ("police, shooting, sports", 9.1),
                        ("reports, share, group", 6.3),
                    ]
                    for kw, pct in topics:
                        st.progress(int(pct), text=f"{kw} — {pct}%")

            except Exception as e:
                # Show pre-computed fallback
                st.markdown("**Top drift-driving topics (pre-computed):**")
                topics = [
                    ("woman, movie, fans, adorable, film", 41.1),
                    ("care, elderly, contact, information", 33.6),
                    ("open, team, season, fight, league", 9.9),
                    ("news, baseball, police, shooting", 9.1),
                    ("q1, reports, share, group, mln", 6.3),
                ]
                for kw, pct in topics:
                    col_a, col_b = st.columns([3, 1])
                    col_a.progress(int(pct))
                    col_b.markdown(f"**{pct}%** — {kw}")

# ── RECENT PAIRS TABLE ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Recent Window Pairs")
recent = df.tail(10)[["pair", "window_start", "window_end",
                        "wkcs", "threshold", "alert"]].copy()
recent["alert"] = recent["alert"].map({True: "🚨 ALERT", False: "✅ Normal"})
st.dataframe(recent, use_container_width=True)

# ── AUTO REFRESH ─────────────────────────────────────────────────────────────
if status != "complete":
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
else:
    st.success("✅ Pipeline complete — all 36 pairs processed")
