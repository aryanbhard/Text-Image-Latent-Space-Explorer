"""
PAGE 2: Image Embedding Explorer (Unit 1 & Unit 3 Requirements)
Generates real CLIP image embeddings (512-D), projects into 2D with PCA & UMAP,
and discovers visually similar images using genuine cosine similarities.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import streamlit as st
import numpy as np
from PIL import Image
from utils.config import CLIP_MODEL_NAME
from models.embedding_models import encode_clip_images
from services.embedding_service import compute_pca_2d, compute_umap_2d, find_nearest_neighbors
from services.image_service import create_curated_sample_images, validate_image_file
from utils.visualization import plot_embedding_scatter_2d

st.set_page_config(page_title="Image Embedding Explorer", page_icon="🖼️", layout="wide")

st.title("🖼️ Image Embedding Explorer")
st.markdown(
    "Explore how visual content is encoded into a high-dimensional continuous representation space "
    f"using the **CLIP Vision Transformer** (<code>{CLIP_MODEL_NAME}</code>, 512-D).",
    unsafe_allow_html=True,
)

# Load curated demo images
curated_items = create_curated_sample_images()

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
reduction_method = st.sidebar.radio("Dimensionality Reduction:", ["PCA", "UMAP"], index=0)
k_neighbors = st.sidebar.slider("Nearest Neighbors (k):", min_value=1, max_value=5, value=3)

# Image Upload Section
st.subheader("1. Image Collection & User Uploads")
col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    uploaded_files = st.file_uploader(
        "Upload Custom Images (JPG, JPEG, PNG, max 5MB each):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

custom_images = []
if uploaded_files:
    for f in uploaded_files:
        valid, msg, img = validate_image_file(f, max_size_mb=5.0)
        if valid and img is not None:
            custom_images.append({
                "name": f"User: {f.name}",
                "short_name": f.name[:12],
                "category": "user_upload",
                "image": img,
            })
        else:
            st.error(f"File '{f.name}': {msg}")

# Combine curated and uploaded images
active_items = curated_items + custom_images
images_list = [item["image"] for item in active_items]
names_list = [item["name"] for item in active_items]
categories_list = [item.get("category", "general") for item in active_items]

with col_up2:
    st.markdown(
        f"""
        <div style="background:#f8fafc; padding:12px; border-radius:6px; border:1px solid #e2e8f0;">
            <b>Active Model:</b> <code>{CLIP_MODEL_NAME}</code><br>
            <b>Vision Backbone:</b> ViT-B/32<br>
            <b>Image Vector Dim:</b> <code>512</code> floats<br>
            <b>Total Images:</b> <code>{len(active_items)}</code> ({len(curated_items)} curated + {len(custom_images)} uploaded)
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gallery preview
with st.expander("👁️ View Thumbnail Gallery of Active Images"):
    cols = st.columns(6)
    for i, item in enumerate(active_items):
        with cols[i % 6]:
            st.image(item["image"], caption=f"#{i}: {item['short_name']}", use_container_width=True)

# Generate Real CLIP Image Embeddings
with st.spinner("Computing real 512-D CLIP image embeddings via Vision Transformer..."):
    image_embeddings = encode_clip_images(images_list, normalize=True)

# Dimensionality Reduction
if reduction_method == "PCA":
    coords_2d, explained_var = compute_pca_2d(image_embeddings)
    dim_info = f"Explained Variance (2 components): {explained_var*100:.1f}%"
else:
    coords_2d = compute_umap_2d(image_embeddings, n_neighbors=min(4, len(active_items) - 1))
    dim_info = "Non-linear manifold optimization via fuzzy simplicial sets"

st.markdown("---")
st.subheader("2. Interactive 2D Image Latent Space")
st.caption(f"Method: **{reduction_method}** | {dim_info}")

col_plot, col_inspect = st.columns([2, 1])

with col_inspect:
    st.markdown("#### 🎯 Query Visually Similar Images")
    selected_idx = st.selectbox(
        "Select Anchor Image:",
        options=list(range(len(active_items))),
        format_func=lambda i: f"#{i}: {active_items[i]['name']}",
    )
    
    # Compute Nearest Neighbors in 512-D CLIP space
    neighbors = find_nearest_neighbors(image_embeddings, query_idx=selected_idx, k=k_neighbors)
    neighbor_indices = [n["index"] for n in neighbors]

with col_plot:
    fig = plot_embedding_scatter_2d(
        coords_2d=coords_2d,
        labels=[f"#{i}" for i in range(len(active_items))],
        hover_texts=names_list,
        color_names=[c.capitalize() for c in categories_list],
        title="CLIP Image Embedding Space",
        reduction_method=reduction_method,
        highlight_idx=selected_idx,
        neighbor_indices=neighbor_indices,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_inspect:
    st.markdown(f"**Query Image #{selected_idx}:**")
    st.image(active_items[selected_idx]["image"], width=130, caption=active_items[selected_idx]["name"])

    st.markdown(f"**Top {len(neighbors)} Visually Similar Matches:**")
    nn_cols = st.columns(len(neighbors) if neighbors else 1)
    for rank, n in enumerate(neighbors):
        idx = n["index"]
        sim = n["cosine_similarity"]
        with nn_cols[rank]:
            st.image(active_items[idx]["image"], width=90, caption=f"#{idx} ({sim:.3f})")

# Raw Embedding Vectors
st.markdown("---")
st.subheader("3. Raw Embedding Vector Inspection")
with st.expander("🔍 View Raw Numerical Embeddings (512 Dimensions)"):
    st.write(f"Image #{selected_idx} normalized CLIP feature vector (first 24 dimensions shown):")
    sample_vec = image_embeddings[selected_idx][:24]
    st.code(", ".join([f"{v:.4f}" for v in sample_vec]) + ", ...")
    st.caption("Extracted from the final projection layer of ViT-B/32, normalized to unit length.")
