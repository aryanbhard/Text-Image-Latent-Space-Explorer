"""
PAGE 4: Cluster Analysis with LLM (Unit 2 Requirements)
Applies K-Means clustering to actual high-dimensional embeddings, evaluates silhouette scores,
and generates automated cluster names and explanations via Gemini LLM (with transparent fallback).
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.config import DEFAULT_TEXT_SAMPLES, GEMINI_API_KEY
from models.embedding_models import encode_sentences
from models.clustering import perform_kmeans_clustering
from services.embedding_service import compute_pca_2d, compute_umap_2d
from services.llm_service import generate_cluster_summary

st.set_page_config(page_title="Cluster Analysis with LLM", page_icon="🧩", layout="wide")

st.title("🧩 Cluster Analysis & LLM Summarization")
st.markdown(
    "Discover natural semantic groupings in high-dimensional embedding spaces via **K-Means clustering**, "
    "and use an **LLM** to automatically synthesize descriptive cluster titles and thematic rationales."
)

# Sidebar controls
st.sidebar.header("⚙️ Clustering Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=6, value=4)
reduction_method = st.sidebar.radio("2D Visualization Layout:", ["PCA", "UMAP"], index=0)

# API Key input option in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Gemini LLM Integration")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (optional):",
    value="",
    type="password",
    help="Leave blank to use environment variable or transparent rule-based semantic analyzer.",
)

# Data source
sentences = DEFAULT_TEXT_SAMPLES

# 1. Compute Embeddings
with st.spinner("Generating Sentence-BERT embeddings (384-D)..."):
    embeddings = encode_sentences(sentences, normalize=True)

# 2. Run K-Means Clustering
with st.spinner(f"Running K-Means clustering (K={n_clusters})..."):
    cluster_results = perform_kmeans_clustering(embeddings, n_clusters=n_clusters)

labels = cluster_results["labels"]
centroids = cluster_results["centroids"]
silhouette = cluster_results["silhouette"]
inertia = cluster_results["inertia"]
representatives = cluster_results["representatives"]
cluster_counts = cluster_results["cluster_counts"]

# 3. Dimensionality Reduction for Visualization
all_points = np.vstack([embeddings, centroids])
if reduction_method == "PCA":
    coords_2d, _ = compute_pca_2d(all_points)
else:
    coords_2d = compute_umap_2d(all_points, n_neighbors=5)

sample_coords = coords_2d[: len(embeddings)]
centroid_coords = coords_2d[len(embeddings) :]

# Metrics Bar
st.subheader("1. Clustering Evaluation Metrics")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Samples", len(sentences))
with m2:
    st.metric("Clusters (K)", n_clusters)
with m3:
    st.metric("Silhouette Score", f"{silhouette:.3f}", help="Range: [-1, +1]. Higher values indicate well-separated clusters.")
with m4:
    st.metric("Inertia (WCSS)", f"{inertia:.1f}", help="Within-Cluster Sum of Squares.")

st.markdown("---")

# 2. Interactive Scatter Plot
st.subheader("2. Cluster Distribution Map")

fig = go.Figure()

colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

for k in range(n_clusters):
    indices = [i for i, lbl in enumerate(labels) if lbl == k]
    c_color = colors[k % len(colors)]
    
    # Points in cluster
    fig.add_trace(
        go.Scatter(
            x=sample_coords[indices, 0],
            y=sample_coords[indices, 1],
            mode="markers",
            name=f"Cluster #{k+1} ({len(indices)} items)",
            marker=dict(size=11, color=c_color, opacity=0.85),
            text=[f"<b>#{idx}:</b> {sentences[idx][:80]}..." for idx in indices],
            hoverinfo="text",
        )
    )
    
    # Centroid marker
    fig.add_trace(
        go.Scatter(
            x=[centroid_coords[k, 0]],
            y=[centroid_coords[k, 1]],
            mode="markers+text",
            name=f"Centroid #{k+1}",
            text=[f"C{k+1}"],
            textposition="top center",
            marker=dict(size=16, color=c_color, symbol="star", line=dict(width=2, color="black")),
            showlegend=False,
            hoverinfo="text",
            hovertext=f"Centroid of Cluster #{k+1}",
        )
    )

fig.update_layout(
    title=dict(text=f"<b>K-Means Clusters in 2D Latent Space</b> ({reduction_method})", font=dict(size=16)),
    xaxis=dict(title=f"{reduction_method} Dim 1", gridcolor="#f0f2f6"),
    yaxis=dict(title=f"{reduction_method} Dim 2", gridcolor="#f0f2f6"),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 3. LLM Cluster Summaries
st.subheader("3. Automated Cluster Summarization & Interpretation")
st.write(
    "Below, each discovered cluster is interpreted using an LLM. "
    "If no Gemini API key is provided, our transparent rule-based semantic analyzer generates technical summaries."
)

active_key = user_api_key.strip() if user_api_key.strip() else None

for k in range(n_clusters):
    indices = [i for i, lbl in enumerate(labels) if lbl == k]
    cluster_sentences = [sentences[i] for i in indices]
    rep_idx = representatives[k] if k < len(representatives) else -1
    rep_text = sentences[rep_idx] if rep_idx >= 0 else "N/A"

    with st.spinner(f"Analyzing Cluster #{k+1}..."):
        summary = generate_cluster_summary(k, cluster_sentences, api_key=active_key)

    with st.expander(f"📌 Cluster #{k+1}: {summary['title']} ({len(cluster_sentences)} sentences)", expanded=True):
        st.markdown(f"**Generated Title:** `{summary['title']}`")
        st.markdown(f"**Explanation:** {summary['explanation']}")
        st.caption(f"Source: **{summary['source']}**")
        
        st.markdown(f"**Exemplar Sample (Closest to Centroid):**")
        st.info(f"\"*{rep_text}*\"")
        
        with st.expander(f"Show all {len(cluster_sentences)} sentences in Cluster #{k+1}"):
            for s in cluster_sentences:
                st.markdown(f"- {s}")
