import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import os

def plot_pca_scatter(data_dir: str, output_dir: str, models: list, ids: list, normalized: bool):
    """
    Loads embedding files grouped by model, runs PCA to 2D,
    and plots each response ID as a labeled scatter point.
    """
    os.makedirs(output_dir, exist_ok=True)
    dir_path = Path(data_dir)

    # Color groups — matches your three constraint groups
    group_colors = {
        "R1": "steelblue", "R2": "steelblue", "R3": "steelblue",   # similar context, different synonyms
        "R4": "tomato",    "R5": "tomato",    "R6": "tomato",        # different context, similar wording
        "R7": "seagreen",  "R8": "seagreen",  "R9": "seagreen",    # everything different
    }
    # group_colors = {
    #     "R1": "steelblue", "R2": "steelblue", "R3": "steelblue",
    #     "R4": "steelblue", "R5": "steelblue", "R6": "steelblue",   # similar context, different synonyms
    #     "R7": "tomato",    "R8": "tomato",    "R9": "tomato",
    #     "R10": "tomato",    "R11": "tomato",    "R12": "tomato",         # different context, similar wording
    #     "R13": "seagreen",  "R14": "seagreen",  "R15": "seagreen",    
    #     "R16": "seagreen",  "R17": "seagreen",  "R18": "seagreen",    # everything different
    # }

    for embed_name in models:
        embeddings = []
        valid_ids = []

        # --- Load all embeddings for this model ---
        for current_id in ids:
            file_name = f"{current_id}_{embed_name}_norm.npy" if normalized else f"{current_id}_{embed_name}.npy"
            file_path = dir_path / file_name

            if file_path.exists():
                try:
                    vec = np.load(file_path)
                    if vec.ndim > 1:
                        vec = vec.flatten()
                    embeddings.append(vec)
                    valid_ids.append(current_id)
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
            else:
                print(f"Warning: Could not find {file_path}")

        if len(embeddings) < 2:
            print(f"Skipping {embed_name} - not enough data for PCA.")
            continue

        # --- Stack into matrix (N, D) and run PCA ---
        matrix = np.stack(embeddings)           # shape: (9, 768) for BGE
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(matrix)     # shape: (9, 2)

        explained = pca.explained_variance_ratio_ * 100

        # --- Plot ---
        plt.figure(figsize=(8, 6))

        for i, current_id in enumerate(valid_ids):
            x, y = reduced[i, 0], reduced[i, 1]
            color = group_colors.get(current_id, "gray")

            plt.scatter(x, y, color=color, s=120, zorder=3)
            plt.annotate(
                current_id,
                xy=(x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=10,
                color=color
            )

        # --- Legend for groups ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=10, label="Similar context, diff synonyms"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",    markersize=10, label="Diff context, similar wording"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",  markersize=10, label="Everything different"),
        ]
        plt.legend(handles=legend_elements, loc="best", fontsize=9)

        plt.title(
            f"PCA 2D Projection — {embed_name.upper()} ({'Normalized' if normalized else 'Unnormalized'})\n"
            f"PC1: {explained[0]:.1f}%  PC2: {explained[1]:.1f}% variance explained",
            fontsize=13
        )
        plt.xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=11)
        plt.ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        suffix = "norm" if normalized else "unnorm"
        save_path = os.path.join(output_dir, f"{embed_name}_pca_{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
    PLOTS_FOLDER = "progress_meeting/plots_response_pca"

    my_models = ["minilm", "bge", "e5"]
    my_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
    # my_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18"]

    plot_pca_scatter(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=False)

    print("\nNow plotting normalized embeddings...\n")

    EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"
    PLOTS_FOLDER = "progress_meeting/plots_response_pca_norm"

    plot_pca_scatter(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=True)