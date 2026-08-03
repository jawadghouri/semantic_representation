# visualization/heatmap.py

import json

import numpy as np
import matplotlib.pyplot as plt


def create_heatmap(
        json_file,
        output_file):

    with open(json_file) as f:
        data = json.load(f)

    labels = []
    distances = []

    for item in data["results"]:
        labels.append(item["model"])
        distances.append(item["distance"])

    matrix = np.array(
        [distances]
    )

    plt.figure(
        figsize=(8, 2)
    )

    plt.imshow(
        matrix,
        aspect="auto"
    )

    plt.xticks(
        range(len(labels)),
        labels
    )

    plt.yticks(
        [0],
        ["Prompt"]
    )

    plt.colorbar(
        label="Distance"
    )

    plt.title(
        f"Prompt Similarity Heatmap ({data['embedding_model']})"
    )

    plt.tight_layout()

    plt.savefig(
        output_file
    )

    plt.close()

