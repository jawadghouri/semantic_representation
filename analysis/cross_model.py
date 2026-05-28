import numpy as np
from scipy.stats import spearmanr


def compute_cross_model_agreement(matrix_a, matrix_b):
    """Flattens two pairwise Euclidean distance matrices and calculates

    the Spearman rank correlation. This checks if Model A and Model B
    agree on the relative geometric spacing of your LLM responses.
    """
    # Extract the upper triangle of the matrices to avoid duplicating pairs and diagonal 0s
    triangle_indices = np.triu_indices_from(matrix_a, k=1)

    flat_a = matrix_a[triangle_indices]
    flat_b = matrix_b[triangle_indices]

    correlation, p_value = spearmanr(flat_a, flat_b)
    return {"spearman_r": float(correlation), "p_value": float(p_value)}