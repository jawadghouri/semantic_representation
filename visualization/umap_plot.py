import numpy as np
import matplotlib.pyplot as plt
import umap


def create_umap(
        embeddings,
        labels,
        output_file):

    reducer = umap.UMAP(
        random_state=42
    )

    reduced = reducer.fit_transform(
        embeddings
    )

    plt.figure(
        figsize=(8, 6)
    )

    for i in range(len(labels)):

        plt.scatter(
            reduced[i,0],
            reduced[i,1]
        )

        plt.text(
            reduced[i,0],
            reduced[i,1],
            labels[i]
        )

    plt.title(
        "UMAP Semantic Space"
    )

    plt.tight_layout()

    plt.savefig(
        output_file
    )

    plt.close()
