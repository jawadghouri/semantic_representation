import numpy as np


def compute_statistics(embeddings):
    """Computes geometric magnitude distribution metrics for unnormalized

    embedding arrays across the vector space.
    """
    # Calculate the L2 norm (magnitude) for every single vector row-wise
    norms = np.linalg.norm(embeddings, axis=1)

    stats = {
        "num_vectors": len(embeddings),
        "dimension": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
    }

    return stats
"""
Why this file is vital for your TU Ilmenau report:
Because you chose unnormalized embeddings, these values will capture how "intensely" 
or "densely" a model encodes concepts. Models like BGE might have a mean norm of ~29.4, 
while MiniLM might have ~8.5. This variance is a key piece of your quantitative findings!
"""