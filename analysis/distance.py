import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def compute_distance_matrix(embeddings):

    return euclidean_distances(
        embeddings
    )