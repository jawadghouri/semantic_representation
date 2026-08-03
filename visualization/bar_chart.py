import json
from pathlib import Path

import matplotlib.pyplot as plt


def create_bar_chart(json_file, output_file):

    with open(json_file) as f:
        data = json.load(f)

    models = []
    distances = []

    for item in data["results"]:
        models.append(item["model"])
        distances.append(item["distance"])

    plt.figure(figsize=(8, 5))

    plt.bar(models, distances)

    plt.title(f"Prompt Similarity ({data['embedding_model']})")

    plt.xlabel("LLM")

    plt.ylabel("Euclidean Distance")

    plt.tight_layout()

    plt.savefig(output_file)

    plt.close()


