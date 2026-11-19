"""
PAGE 1: Text Embedding Explorer (Unit 1 & Unit 3 Requirements)
Generates real Sentence-BERT embeddings, projects into 2D with PCA & UMAP,
and provides interactive nearest-neighbor exploration.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import streamlit as st
import numpy as np
from utils.config import DEFAULT_TEXT_SAMPLES, SBERT_MODEL_NAME
from models.embedding_models import encode_sentences
from services.embedding_service import compute_pca_2d, compute_umap_2d, find_nearest_neighbors
from models.clustering import perform_kmeans_clustering
from utils.visualization import plot_embedding_scatter_2d

st.set_page_config(page_title="Text Embedding Explorer", page_icon="📝", layout="wide")

st.title("📝 Text Embedding Explorer")
st.markdown(
    "Explore how high-dimensional sentence representations (Sentence-BERT 384-D) "
    "are mapped to interactive 2D latent spaces via **PCA** and **UMAP**."
)

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
reduction_method = st.sidebar.radio("Dimensionality Reduction:", ["PCA", "UMAP"], index=0)
k_neighbors = st.sidebar.slider("Nearest Neighbors (k):", min_value=1, max_value=8, value=3)
auto_clusters = st.sidebar.slider("Number of Clusters (K-Means):", min_value=2, max_value=6, value=4)

# Text Corpus Management
st.subheader("1. Sentence Corpus Input")
col_input, col_info = st.columns([2, 1])

with col_input:
    corpus_mode = st.radio(
        "Choose Corpus Source:",
        ["Curated Diverse Corpus (25 sentences across 5 domains)", "Custom Sentences (Enter your own)"],
        horizontal=True,
    )
    if corpus_mode.startswith("Curated"):
        sentences = DEFAULT_TEXT_SAMPLES
        st.caption("Using 25 pre-configured sentences spanning Technology, Nature, Art, Sports, and Food.")
    else:
        custom_text = st.text_area(
            "Enter sentences (one per line, minimum 3 sentences):",
            value=(
                "Deep learning models process large text corpora.\n"
                "Neural networks extract dense numerical features.\n"
                "A walk in the green forest refreshes the mind.\n"
                "The ocean waves crash gently against the sandy beach.\n"
                "The chef prepared a delicious gourmet dinner."
            ),
            height=130,
        )
        sentences = [line.strip() for line in custom_text.split("\n") if line.strip()]

with col_info:
    st.markdown(
        f"""
        <div style="background:#f8fafc; padding:12px; border-radius:6px; border:1px solid #e2e8f0;">
            <b>Active Model:</b> <code>{SBERT_MODEL_NAME}</code><br>
            <b>Embedding Dimension:</b> <code>384</code> floats<br>
            <b>Active Sentences:</b> <code>{len(sentences)}</code><br>
            <b>Metric:</b> Cosine Similarity / Euclidean Distance
        </div>
        """,
        unsafe_allow_html=True,
    )

if len(sentences) < 2:
    st.warning("Please provide at least 2 sentences to perform dimensionality reduction and neighborhood queries.")
    st.stop()

# Generate Embeddings
with st.spinner("Computing real Sentence-BERT embeddings (384-D)..."):
    embeddings = encode_sentences(sentences, normalize=True)

# Dimensionality Reduction
if reduction_method == "PCA":
    coords_2d, explained_var = compute_pca_2d(embeddings)
    dim_info = f"Explained Variance (2 components): {explained_var*100:.1f}%"
else:
    coords_2d = compute_umap_2d(embeddings, n_neighbors=min(5, len(sentences) - 1))
    dim_info = "Non-linear manifold optimization via fuzzy simplicial sets"

# Clustering for color coding
cluster_res = perform_kmeans_clustering(embeddings, n_clusters=min(auto_clusters, len(sentences)))
labels = cluster_res["labels"]
color_names = [f"Cluster {lbl+1}" for lbl in labels]

st.markdown("---")
st.subheader("2. Interactive 2D Latent Space Projection")
st.caption(f"Method: **{reduction_method}** | {dim_info}")

# Selection UI for nearest neighbor probe
col_plot, col_nn = st.columns([2, 1])

with col_nn:
    st.markdown("#### 🎯 Query Nearest Neighbors")
    selected_idx = st.selectbox(
        "Select Anchor Sentence to Inspect:",
        options=list(range(len(sentences))),
        format_func=lambda i: f"#{i}: {sentences[i][:45]}...",
    )
    
    # Compute Nearest Neighbors for selected point
    neighbors = find_nearest_neighbors(embeddings, query_idx=selected_idx, k=k_neighbors)
    neighbor_indices = [n["index"] for n in neighbors]

with col_plot:
    fig = plot_embedding_scatter_2d(
        coords_2d=coords_2d,
        labels=[f"#{i}" for i in range(len(sentences))],
        hover_texts=sentences,
        color_names=color_names,
        title=f"Sentence-BERT Latent Space",
        reduction_method=reduction_method,
        highlight_idx=selected_idx,
        neighbor_indices=neighbor_indices,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_nn:
    st.markdown(f"**Selected Query:**")
    st.info(f"\"*{sentences[selected_idx]}*\"")
    
    st.markdown(f"**Top {len(neighbors)} Nearest Neighbors:**")
    for rank, n in enumerate(neighbors, start=1):
        idx = n["index"]
        sim = n["cosine_similarity"]
        dist = n["euclidean_distance"]
        st.markdown(
            f"**{rank}. Point #{idx}** (Cosine Sim: `+{sim:.4f}`, Dist: `{dist:.3f}`)<br>"
            f"*{sentences[idx]}*",
            unsafe_allow_html=True,
        )

# Vector Inspection Section
st.markdown("---")
st.subheader("3. Raw Embedding Vector Inspection")
with st.expander("🔍 View Raw Numerical Embeddings (384 Dimensions)"):
    st.write(f"Sample #{selected_idx} normalized vector (first 24 dimensions shown):")
    sample_vec = embeddings[selected_idx][:24]
    st.code(", ".join([f"{v:.4f}" for v in sample_vec]) + ", ...")
    st.caption("All 384 floating-point coordinates are strictly normalized so that Euclidean norm ||v|| = 1.0.")

# Theoretical Foundation (Unit 1 Requirements)
st.markdown("---")
st.subheader("4. Foundational Principles & Theory")
col_th1, col_th2 = st.columns(2)
with col_th1:
    st.markdown(
        """
        #### Generative vs Discriminative AI
        - **Generative AI:** Estimates the data distribution $P(X)$ or joint distribution $P(X, Y)$ to generate novel, plausible samples. Models understand how the data itself was composed.
        - **Discriminative AI:** Estimates the decision boundary $P(Y \mid X)$ to map inputs to existing class labels. It does not synthesize new data instances.
        """
    )
with col_th2:
    st.markdown(
        """
        #### High-Dimensional Latent Spaces & Metrics
        - In $\mathbb{R}^{384}$, raw Euclidean distances can suffer from the *curse of dimensionality* (concentrated distance distributions).
        - **Cosine Similarity:** $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ evaluates angular alignment rather than magnitude, measuring conceptual semantic agreement.
        - **PCA** finds the linear axes of maximum variance, whereas **UMAP** models the Riemannian manifold preserving local neighborhoods.
        """
    )
