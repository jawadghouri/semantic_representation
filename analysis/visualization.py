import umap

import matplotlib.pyplot as plt


def create_umap_plot(
        embeddings,
        labels,
        output_path
):

    reducer = umap.UMAP(
        random_state=42
    )

    reduced = reducer.fit_transform(
        embeddings
    )

    plt.figure(
        figsize=(10, 8)
    )

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels
    )

    plt.colorbar(scatter)

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()