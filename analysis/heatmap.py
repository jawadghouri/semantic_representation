import matplotlib.pyplot as plt


def save_heatmap(
        matrix,
        labels,
        output_file
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        matrix
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45
    )

    plt.yticks(
        range(len(labels)),
        labels
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        output_file
    )

    plt.close()