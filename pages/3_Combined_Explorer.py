"""
PAGE 3: Combined Text and Image Explorer (Unit 1 & Unit 3 Requirements)
Contrasts Sentence-BERT vs CLIP independent spaces and demonstrates genuine
joint multimodal contrastive alignment using OpenAI CLIP.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import streamlit as st
import numpy as np
from PIL import Image
from utils.config import SBERT_MODEL_NAME, CLIP_MODEL_NAME, DEFAULT_TEXT_SAMPLES
from models.embedding_models import (
    encode_sentences,
    encode_clip_texts,
    encode_clip_images,
    compute_cross_modal_similarity,
)
from services.embedding_service import compute_pca_2d, compute_umap_2d
from services.image_service import create_curated_sample_images, validate_image_file
from utils.visualization import plot_combined_multimodal_scatter, render_table_as_markdown

st.set_page_config(page_title="Combined Text & Image Explorer", page_icon="🔗", layout="wide")

st.title("🔗 Combined Text & Image Explorer")
st.markdown(
    "Analyze multimodal representations across disparate and joint embedding manifolds. "
    "Investigate how vision and language interact in machine latent spaces."
)

# Load curated demo images
curated_items = create_curated_sample_images()
demo_texts = DEFAULT_TEXT_SAMPLES[:12]

# Sidebar configuration
st.sidebar.header("⚙️ Visualization Mode")
mode = st.sidebar.radio(
    "Select Projection Mode:",
    [
        "1. CLIP Shared Multimodal Space (Genuine Cross-Modal Alignment)",
        "2. Independent Projections (Sentence-BERT Text + CLIP Image)",
    ],
    index=0,
)
reduction_method = st.sidebar.radio("Dimensionality Reduction:", ["PCA", "UMAP"], index=0)

# Mode 1: Shared Joint CLIP Space
if mode.startswith("1. CLIP Shared"):
    st.subheader("1. Unified Multimodal Space (OpenAI CLIP)")
    st.info(
        "💡 **Theoretical Foundation:** In this mode, both text and images are projected into the **same 512-dimensional CLIP space**. "
        "Trained via symmetric contrastive cross-entropy loss, text and image vectors reside in a shared geometric manifold, "
        "enabling genuine cosine similarity and zero-shot cross-modal retrieval."
    )

    # Interactive Cross-Modal Query Bar
    st.markdown("#### 🔍 Zero-Shot Text-to-Image Semantic Search")
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query_text = st.text_input(
            "Enter any text query to search across the visual gallery:",
            value="a comfortable running sneaker or sports shoe",
        )
    with col_q2:
        top_k_matches = st.slider("Number of Matches:", 1, 6, 3)

    # Compute CLIP embeddings
    gallery_images = [item["image"] for item in curated_items]
    gallery_names = [item["name"] for item in curated_items]

    with st.spinner("Computing shared 512-D CLIP representations..."):
        text_queries = [query_text] + demo_texts[:8]
        clip_text_embs = encode_clip_texts(text_queries, normalize=True)
        clip_image_embs = encode_clip_images(gallery_images, normalize=True)

    # Cross-modal ranking for the user's query
    query_emb = clip_text_embs[0:1]
    similarities = compute_cross_modal_similarity(query_emb, clip_image_embs)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k_matches]

    st.markdown(f"**Top Matches for:** *\"{query_text}\"*")
    res_cols = st.columns(top_k_matches)
    for rank, idx in enumerate(ranked_indices):
        with res_cols[rank]:
            st.image(gallery_images[idx], caption=f"Rank #{rank+1}: {curated_items[idx]['short_name']}\n(Sim: {similarities[idx]:.4f})", use_container_width=True)

    st.markdown("---")
    st.subheader("2. Joint 2D Multimodal Map")
    
    # Combined joint 2D reduction
    all_clip_embs = np.vstack([clip_text_embs, clip_image_embs])
    if reduction_method == "PCA":
        joint_coords_2d, _ = compute_pca_2d(all_clip_embs)
    else:
        joint_coords_2d = compute_umap_2d(all_clip_embs, n_neighbors=4)

    n_texts = len(text_queries)
    t_coords = joint_coords_2d[:n_texts]
    i_coords = joint_coords_2d[n_texts:]

    fig = plot_combined_multimodal_scatter(
        text_coords=t_coords,
        text_labels=text_queries,
        image_coords=i_coords,
        image_labels=gallery_names,
        reduction_method=reduction_method,
        is_shared_clip_space=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# Mode 2: Independent Projections (Categorical Modality Mapping)
else:
    st.subheader("2. Independent Model Projections (Warning & Demonstration)")
    st.warning(
        "⚠️ **CRITICAL ARCHITECTURAL WARNING (Official P6 Syllabus Requirement):**<br>"
        "Sentence-BERT produces **384-dimensional** text embeddings optimized for semantic textual similarity (STS).<br>"
        "CLIP produces **512-dimensional** image embeddings optimized for vision-language contrastive alignment.<br>"
        "**These raw vectors exist in fundamentally disparate geometric spaces.** You CANNOT compute cosine similarity "
        "between raw Sentence-BERT vectors and CLIP image vectors. Below, they are projected independently and displayed "
        "to illustrate the danger of confounding heterogeneous latent spaces.",
        icon="⚠️",
    )

    with st.spinner("Computing independent embeddings..."):
        sbert_embs = encode_sentences(demo_texts, normalize=True)
        clip_img_embs = encode_clip_images([item["image"] for item in curated_items], normalize=True)

    # Reduce each independently to 2D
    if reduction_method == "PCA":
        sbert_2d, _ = compute_pca_2d(sbert_embs)
        clip_2d, _ = compute_pca_2d(clip_img_embs)
    else:
        sbert_2d = compute_umap_2d(sbert_embs, n_neighbors=4)
        clip_2d = compute_umap_2d(clip_img_embs, n_neighbors=4)

    fig = plot_combined_multimodal_scatter(
        text_coords=sbert_2d,
        text_labels=demo_texts,
        image_coords=clip_2d,
        image_labels=[item["name"] for item in curated_items],
        reduction_method=reduction_method,
        is_shared_clip_space=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# Comparison Table
st.markdown("---")
st.subheader("3. Architectural Comparison: Sentence-BERT vs CLIP")
comparison_data = [
    {"Feature": "Architecture Backbone", "Sentence-BERT (MiniLM-L6)": "Transformer Encoder (BERT-style)", "OpenAI CLIP (ViT-B/32)": "Dual Encoder (ViT Image + Transformer Text)"},
    {"Feature": "Embedding Dimension", "Sentence-BERT (MiniLM-L6)": "384 Dimensions", "OpenAI CLIP (ViT-B/32)": "512 Dimensions (Text & Image)"},
    {"Feature": "Training Objective", "Sentence-BERT (MiniLM-L6)": "Multiple Negatives Ranking / Cosine Similarity Loss", "OpenAI CLIP (ViT-B/32)": "InfoNCE Symmetric Contrastive Cross-Entropy"},
    {"Feature": "Cross-Modal Capability", "Sentence-BERT (MiniLM-L6)": "Text-Only (Unimodal)", "OpenAI CLIP (ViT-B/32)": "Text & Image Aligned (Multimodal)"},
    {"Feature": "Direct Dot Product Compatibility", "Sentence-BERT (MiniLM-L6)": "Compatible only with other SBERT vectors", "OpenAI CLIP (ViT-B/32)": "Directly compatible between CLIP text & CLIP image"},
]
st.markdown(render_table_as_markdown(comparison_data), unsafe_allow_html=True)
