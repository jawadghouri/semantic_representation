import numpy as np
from scipy.spatial.distance import cdist


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Return N×N Euclidean distance matrix for N response embeddings."""
    return cdist(embeddings, embeddings, metric="euclidean")


def extract_distance_sets(dist_matrix: np.ndarray, labels: list):
    """
    From a pairwise distance matrix and G/H labels, extract:
      d_gg: intra-genuine pairwise distances (upper triangle)
      d_hh: intra-hallucinated pairwise distances (upper triangle)
      d_gh: cross-class pairwise distances (all G×H pairs)
    """
    n = len(labels)
    g_idx = [i for i, l in enumerate(labels) if l == "G"]
    h_idx = [i for i, l in enumerate(labels) if l == "H"]

    d_gg = []
    for i in range(len(g_idx)):
        for j in range(i + 1, len(g_idx)):
            d_gg.append(dist_matrix[g_idx[i], g_idx[j]])

    d_hh = []
    for i in range(len(h_idx)):
        for j in range(i + 1, len(h_idx)):
            d_hh.append(dist_matrix[h_idx[i], h_idx[j]])

    d_gh = []
    for gi in g_idx:
        for hi in h_idx:
            d_gh.append(dist_matrix[gi, hi])

    return np.array(d_gg), np.array(d_hh), np.array(d_gh)
