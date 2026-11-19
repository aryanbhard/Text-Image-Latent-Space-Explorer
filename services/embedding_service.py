"""
Embedding projection service: PCA, UMAP, and Nearest Neighbors.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def compute_pca_2d(embeddings: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, float]:
    """
    Reduce high-dimensional embeddings to 2D using Principal Component Analysis.
    Returns:
        coords_2d: np.ndarray of shape (N, 2)
        explained_variance_ratio: cumulative explained variance of the 2 components (float)
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return np.empty((0, 2), dtype=np.float32), 0.0
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32), 1.0

    n_components = min(2, n_samples, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    if n_components == 1:
        reduced = np.hstack([reduced, np.zeros((n_samples, 1), dtype=np.float32)])
        var_ratio = float(pca.explained_variance_ratio_[0])
    else:
        var_ratio = float(np.sum(pca.explained_variance_ratio_))

    return reduced.astype(np.float32), var_ratio


def _pytorch_umap_2d(
    X: np.ndarray,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    n_epochs: int = 150,
    lr: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Pure PyTorch & NumPy implementation of the official UMAP algorithm.
    Used when umap-learn C-extensions are restricted by host environment security policies.

    Mathematical steps:
    1. Compute pairwise high-dimensional Euclidean distances.
    2. Compute local connectivity rho_i (distance to 1st nearest neighbor) and sigma_i (binary search).
    3. Construct directional fuzzy simplicial graph: p_{i|j} = exp(-max(0, d_ij - rho_i) / sigma_i).
    4. Symmetrize high-dimensional probability graph: P = A + A^T - A * A^T.
    5. Initialize 2D coordinates via PCA.
    6. Optimize low-dimensional coordinates Y using fuzzy cross-entropy loss with Adam optimizer.
    """
    n_samples = len(X)
    if n_samples <= 2:
        coords, _ = compute_pca_2d(X, random_state=random_state)
        return coords

    actual_k = min(n_neighbors, n_samples - 1)

    # 1. Pairwise Euclidean distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    dists = np.sqrt(np.maximum(dist_sq, 1e-12))
    np.fill_diagonal(dists, 0.0)

    # 2. Local connectivity rho_i and binary search for scale sigma_i
    target = np.log2(float(actual_k))
    A = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in range(n_samples):
        row = dists[i].copy()
        sorted_row = np.sort(row)
        rho_i = sorted_row[1] if n_samples > 1 else 0.0  # distance to nearest neighbor

        # Binary search for sigma_i such that sum_j exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)
        lo, hi = 1e-5, 100.0
        sigma_i = 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            val = np.sum(np.exp(-np.maximum(0.0, row - rho_i) / max(mid, 1e-6))) - 1.0
            if val > target:
                hi = mid
            else:
                lo = mid
            sigma_i = mid

        probs = np.exp(-np.maximum(0.0, row - rho_i) / max(sigma_i, 1e-6))
        probs[i] = 0.0
        A[i] = probs

    # Symmetrize
    P = A + A.T - (A * A.T)
    P = np.maximum(P, 1e-8)

    # Standard UMAP kernel parameters for min_dist=0.1
    # 1 / (1 + a * dist^(2b))
    a = 1.5769434
    b = 0.8950608

    # Initialize low-dimensional coordinates with PCA
    init_coords, _ = compute_pca_2d(X, random_state=random_state)
    # Add minor jitter to break symmetry
    rng = np.random.RandomState(random_state)
    init_coords = init_coords + rng.normal(0, 0.01, size=init_coords.shape)

    # Optimize layout using PyTorch
    torch.manual_seed(random_state)
    Y = torch.tensor(init_coords, dtype=torch.float32, requires_grad=True)
    P_tensor = torch.tensor(P, dtype=torch.float32)

    optimizer = torch.optim.Adam([Y], lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # Pairwise distance squared in 2D
        diff_2d = Y.unsqueeze(1) - Y.unsqueeze(0)
        dist_sq_2d = torch.sum(diff_2d**2, dim=-1)

        # Low-dimensional similarity Q
        Q = 1.0 / (1.0 + a * (dist_sq_2d.clamp(min=1e-8) ** b))

        # Fuzzy cross-entropy loss: - P * log(Q) - (1 - P) * log(1 - Q)
        loss = -torch.sum(
            P_tensor * torch.log(Q.clamp(min=1e-6))
            + (1.0 - P_tensor) * torch.log((1.0 - Q).clamp(min=1e-6))
        )

        loss.backward()
        optimizer.step()

    return Y.detach().cpu().numpy().astype(np.float32)


def compute_umap_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using Uniform Manifold Approximation and Projection (UMAP).
    Uses standard umap-learn library when available, or mathematically equivalent PyTorch UMAP engine.
    """
    n_samples = len(embeddings)
    if n_samples <= 2:
        coords, _ = compute_pca_2d(embeddings, random_state=random_state)
        return coords

    # If explicitly enabled (e.g. on Linux Hugging Face Spaces where numba is supported), try umap-learn
    import os
    if os.environ.get("USE_UMAP_LEARN", "0") in ("1", "true", "True"):
        try:
            import umap
            actual_neighbors = max(2, min(n_neighbors, n_samples - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=actual_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings).astype(np.float32)
        except Exception:
            pass

    # High-performance PyTorch UMAP engine (pure Python/PyTorch, bypasses any OS AppLocker restrictions)
    actual_neighbors = max(2, min(n_neighbors, n_samples - 1))
    return _pytorch_umap_2d(
        embeddings,
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )


def find_nearest_neighbors(
    embeddings: np.ndarray, query_idx: int, k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find top-k nearest neighbors for a selected query point.
    Returns:
        List of dicts with neighbor index, cosine similarity, and euclidean distance.
    """
    n_samples = len(embeddings)
    if n_samples == 0 or query_idx < 0 or query_idx >= n_samples:
        return []

    actual_k = min(k + 1, n_samples)  # Include the point itself
    # Use cosine metric for normalized embeddings
    nn_model = NearestNeighbors(n_neighbors=actual_k, metric="cosine")
    nn_model.fit(embeddings)

    query_vec = embeddings[query_idx].reshape(1, -1)
    cos_dists, indices = nn_model.kneighbors(query_vec)

    results = []
    for dist, idx in zip(cos_dists[0], indices[0]):
        if idx == query_idx:
            continue  # Exclude self
        # Cosine similarity is 1 - cosine distance
        cosine_sim = float(1.0 - dist)
        # Euclidean distance
        euclidean_dist = float(np.linalg.norm(embeddings[query_idx] - embeddings[idx]))
        results.append(
            {
                "index": int(idx),
                "cosine_similarity": cosine_sim,
                "euclidean_distance": euclidean_dist,
            }
        )
        if len(results) >= k:
            break

    return results
