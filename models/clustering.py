"""
Clustering algorithms and representative sample extraction.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import numpy as np
from typing import Dict, Any, List
try:
    from sklearn.cluster import KMeans
except ImportError:
    from sklearn.cluster._kmeans import KMeans
from sklearn.metrics import silhouette_score


def perform_kmeans_clustering(
    embeddings: np.ndarray, n_clusters: int = 3, random_state: int = 42
) -> Dict[str, Any]:
    """
    Cluster embeddings using K-Means and extract cluster centroids, labels,
    silhouette scores, and representative indices.

    Args:
        embeddings: np.ndarray of shape (N, D)
        n_clusters: number of clusters (K)
        random_state: random seed for reproducibility

    Returns:
        Dictionary containing labels, centroids, representative indices, and metrics.
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return {
            "labels": np.array([]),
            "centroids": np.empty((0, 0)),
            "representatives": [],
            "silhouette": 0.0,
            "cluster_counts": {},
            "inertia": 0.0,
        }

    # Ensure n_clusters does not exceed n_samples
    actual_k = max(1, min(n_clusters, n_samples))
    kmeans = KMeans(n_clusters=actual_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Calculate silhouette score if valid (requires 2 <= k < n_samples and distinct clusters)
    silhouette = 0.0
    if 1 < actual_k < n_samples and len(np.unique(labels)) > 1:
        try:
            silhouette = float(silhouette_score(embeddings, labels))
        except Exception:
            silhouette = 0.0

    # Identify representative exemplar for each cluster (closest sample to centroid)
    representatives = []
    cluster_counts = {}
    for cluster_id in range(actual_k):
        indices = np.where(labels == cluster_id)[0]
        cluster_counts[cluster_id] = len(indices)
        if len(indices) > 0:
            cluster_points = embeddings[indices]
            center = centroids[cluster_id]
            # Euclidean distance to centroid
            dists = np.linalg.norm(cluster_points - center, axis=1)
            closest_local_idx = np.argmin(dists)
            representatives.append(int(indices[closest_local_idx]))
        else:
            representatives.append(-1)

    return {
        "labels": labels,
        "centroids": centroids,
        "representatives": representatives,
        "silhouette": silhouette,
        "cluster_counts": cluster_counts,
        "inertia": float(kmeans.inertia_),
        "n_clusters": actual_k,
    }
