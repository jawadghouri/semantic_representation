"""
Prompt-response semantic distance analysis.

Computes Euclidean distances between prompt embeddings and response embeddings,
measuring how closely each response adheres to its prompt topic.

This is the first phase of prompt-anchor comparison analysis.
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def compute_prompt_response_distances(prompt_embeddings, response_embeddings):
    """
    Compute Euclidean distances from a prompt to multiple responses.

    Args:
        prompt_embeddings: shape (1, D) or (D,) — single prompt embedding
        response_embeddings: shape (N, D) — N response embeddings

    Returns:
        1D array of shape (N,) — distance from prompt to each response
    """
    if prompt_embeddings.ndim == 1:
        prompt_embeddings = prompt_embeddings.reshape(1, -1)

    distances = euclidean_distances(prompt_embeddings, response_embeddings).squeeze()
    return distances


def compute_statistics(distances):
    """
    Compute summary statistics for prompt-response distances.

    Args:
        distances: 1D array of distances

    Returns:
        dict with keys: mean, std, min, max, median
    """
    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "median": float(np.median(distances)),
        "n": len(distances),
    }


def rank_responses_by_proximity(distances, response_ids):
    """
    Rank responses by closeness to prompt (shortest distance first).

    Args:
        distances: 1D array of distances
        response_ids: list of response identifiers

    Returns:
        list of tuples (response_id, distance, rank) sorted by distance
    """
    ranked = sorted(
        zip(response_ids, distances),
        key=lambda x: x[1]
    )
    return [
        (rid, dist, i + 1)
        for i, (rid, dist) in enumerate(ranked)
    ]
