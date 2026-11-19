"""
Unit tests for Sentence-BERT and CLIP embedding models and dimensionality reduction.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from models.embedding_models import (
    encode_sentences,
    encode_clip_texts,
    encode_clip_images,
    compute_cross_modal_similarity,
)
from services.embedding_service import (
    compute_pca_2d,
    compute_umap_2d,
    find_nearest_neighbors,
)
from models.clustering import perform_kmeans_clustering


def test_sbert_embedding_dimensions():
    sample_texts = [
        "Artificial intelligence revolutionizes latent spaces.",
        "A classical symphony features harmonic counterpoint.",
    ]
    embs = encode_sentences(sample_texts, normalize=True)
    assert embs.shape == (2, 384), f"Expected shape (2, 384), got {embs.shape}"
    # Verify normalization
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)


def test_clip_text_and_image_dimensions():
    texts = ["a red square", "a blue circle"]
    clip_text_embs = encode_clip_texts(texts, normalize=True)
    assert clip_text_embs.shape == (2, 512), f"Expected CLIP text shape (2, 512), got {clip_text_embs.shape}"

    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    clip_img_embs = encode_clip_images([img], normalize=True)
    assert clip_img_embs.shape == (1, 512), f"Expected CLIP image shape (1, 512), got {clip_img_embs.shape}"

    # Cross-modal similarity
    sim = compute_cross_modal_similarity(clip_text_embs, clip_img_embs)
    assert sim.shape == (2, 1), f"Expected similarity shape (2, 1), got {sim.shape}"
    # Red square text should have higher similarity to red image than blue circle text
    assert sim[0, 0] > sim[1, 0], f"Red text similarity ({sim[0,0]}) should exceed blue text ({sim[1,0]})"


def test_pca_and_umap_projections():
    X = np.random.randn(8, 384).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    pca_2d, var_ratio = compute_pca_2d(X)
    assert pca_2d.shape == (8, 2), f"Expected PCA shape (8, 2), got {pca_2d.shape}"
    assert 0.0 <= var_ratio <= 1.0, f"Variance ratio {var_ratio} must be between 0 and 1"

    umap_2d = compute_umap_2d(X, n_neighbors=3)
    assert umap_2d.shape == (8, 2), f"Expected UMAP shape (8, 2), got {umap_2d.shape}"


def test_nearest_neighbors_and_clustering():
    X = np.random.randn(10, 64).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    nns = find_nearest_neighbors(X, query_idx=0, k=3)
    assert len(nns) == 3, f"Expected 3 neighbors, got {len(nns)}"
    assert "cosine_similarity" in nns[0]
    assert nns[0]["index"] != 0  # Should not be self

    # Clustering
    cl_res = perform_kmeans_clustering(X, n_clusters=3)
    assert len(cl_res["labels"]) == 10
    assert cl_res["centroids"].shape == (3, 64)
    assert len(cl_res["representatives"]) == 3
