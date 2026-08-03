from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)

import matplotlib.pyplot as plt


def save_dendrogram(
        embeddings,
        labels,
        output_file
):

    Z = linkage(
        embeddings,
        method="ward"
    )

    plt.figure(
        figsize=(8, 6)
    )

    dendrogram(
        Z,
        labels=labels
    )

    plt.tight_layout()

    plt.savefig(
        output_file
    )

    plt.close()