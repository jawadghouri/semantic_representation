import numpy as np


def compute_statistics(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    return {

        "num_vectors":
            len(embeddings),

        "dimension":
            embeddings.shape[1],

        "mean_norm":
            float(np.mean(norms)),

        "std_norm":
            float(np.std(norms)),

        "min_norm":
            float(np.min(norms)),

        "max_norm":
            float(np.max(norms))
    }