##
# This script applies Principal Component Analysis (PCA) to pre-computed text 
# embeddings (MiniLM, BGE, E5) to analyze and visualize their underlying vector spaces. 
# It generates two main outputs: a series of 2D scatter plots projecting 
# the first three principal components (color-coded by predefined semantic groups), 
# and a cumulative variance line graph illustrating how many dimensions are needed 
# to retain the data's information. Additionally, it prints a terminal summary table 
# detailing the exact number of components each model requires to hit 
# 60%, 80%, and 90% explained variance, running the entire analysis for both 
# unnormalized and normalized embeddings.
##

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from pathlib import Path
import os


# -------------------------------------------------------
# CONFIG — group membership and colors
# -------------------------------------------------------
GROUP_CONFIG = {
    "Similar context, diff synonyms": {
        "ids": ["R1", "R2", "R3", "R4", "R5", "R6"],
        "color": "steelblue"
    },
    "Diff context, similar wording": {
        "ids": ["R7", "R8", "R9", "R10", "R11", "R12"],
        "color": "tomato"
    },
    "Everything different": {
        "ids": ["R13", "R14", "R15", "R16", "R17", "R18"],
        "color": "seagreen"
    },
}

def build_color_map(group_config):
    color_map = {}
    for group_label, cfg in group_config.items():
        for rid in cfg["ids"]:
            color_map[rid] = cfg["color"]
    return color_map


def load_embeddings(data_dir, ids, embed_name, normalized):
    dir_path = Path(data_dir)
    embeddings, valid_ids = [], []
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
    return np.stack(embeddings) if embeddings else None, valid_ids


# -------------------------------------------------------
# OPTION 1 — PCA 3-component scatter (three pair plots)
# -------------------------------------------------------
def plot_pca_3components(data_dir, output_dir, models, ids, normalized):
    os.makedirs(output_dir, exist_ok=True)
    color_map = build_color_map(GROUP_CONFIG)
    suffix = "Normalized" if normalized else "Unnormalized"
    suffix_file = "norm" if normalized else "unnorm"

    pc_pairs = [(0, 1), (0, 2), (1, 2)]  # PC1v2, PC1v3, PC2v3

    for embed_name in models:
        matrix, valid_ids = load_embeddings(data_dir, ids, embed_name, normalized)
        if matrix is None or len(valid_ids) < 3:
            print(f"Skipping {embed_name} - not enough data.")
            continue

        pca = PCA(n_components=3, random_state=42)
        reduced = pca.fit_transform(matrix)         # (N, 3)
        explained = pca.explained_variance_ratio_ * 100

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"PCA 3-Component Projections — {embed_name.upper()} ({suffix})\n"
            f"PC1: {explained[0]:.1f}%  PC2: {explained[1]:.1f}%  PC3: {explained[2]:.1f}%  "
            f"(Total: {sum(explained):.1f}%)",
            fontsize=13
        )

        for ax, (xi, yi) in zip(axes, pc_pairs):
            for i, current_id in enumerate(valid_ids):
                color = color_map.get(current_id, "gray")
                ax.scatter(reduced[i, xi], reduced[i, yi], color=color, s=120, zorder=3)
                ax.annotate(
                    current_id,
                    xy=(reduced[i, xi], reduced[i, yi]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    color=color
                )

            ax.set_xlabel(f"PC{xi+1} ({explained[xi]:.1f}%)", fontsize=10)
            ax.set_ylabel(f"PC{yi+1} ({explained[yi]:.1f}%)", fontsize=10)
            ax.set_title(f"PC{xi+1} vs PC{yi+1}", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

        # Shared legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cfg["color"],
                   markersize=10, label=label)
            for label, cfg in GROUP_CONFIG.items()
        ]
        axes[2].legend(handles=legend_elements, loc="best", fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{embed_name}_pca3_{suffix_file}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


# -------------------------------------------------------
# OPTION 2 — Cumulative variance curve
# -------------------------------------------------------
def plot_cumulative_variance(data_dir, output_dir, models, ids, normalized):
    os.makedirs(output_dir, exist_ok=True)
    suffix = "Normalized" if normalized else "Unnormalized"
    suffix_file = "norm" if normalized else "unnorm"

    model_colors = {"minilm": "steelblue", "bge": "tomato", "e5": "seagreen"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for embed_name in models:
        matrix, valid_ids = load_embeddings(data_dir, ids, embed_name, normalized)
        if matrix is None:
            continue

        # Fit PCA with max possible components (min of N samples, D dims)
        max_components = min(matrix.shape[0], matrix.shape[1])
        pca = PCA(n_components=max_components, random_state=42)
        pca.fit(matrix)

        cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
        components = np.arange(1, len(cumulative) + 1)

        ax.plot(
            components, cumulative,
            color=model_colors.get(embed_name, "gray"),
            linewidth=2,
            marker="o",
            markersize=4,
            label=embed_name.upper()
        )

        # Annotate 90% crossover point
        for i, val in enumerate(cumulative):
            if val >= 90:
                ax.annotate(
                    f"{embed_name.upper()}: {i+1} PCs",
                    xy=(i+1, val),
                    xytext=(i+3, val - 5),
                    fontsize=8,
                    color=model_colors.get(embed_name, "gray"),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
                )
                break

    # Reference lines
    for threshold in [60, 80, 90]:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(0.5, threshold + 0.5, f"{threshold}%", fontsize=8, color="gray")

    ax.set_xlabel("Number of principal components", fontsize=11)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=11)
    ax.set_title(f"Cumulative PCA Variance — All Models ({suffix})", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"cumulative_variance_{suffix_file}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # Also print the table
    print(f"\nComponents needed per model ({suffix}):")
    print(f"{'Model':<10} {'60%':<8} {'80%':<8} {'90%':<8}")
    print("-" * 34)
    for embed_name in models:
        matrix, _ = load_embeddings(data_dir, ids, embed_name, normalized)
        if matrix is None:
            continue
        max_components = min(matrix.shape[0], matrix.shape[1])
        pca = PCA(n_components=max_components, random_state=42)
        pca.fit(matrix)
        cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
        results = {}
        for threshold in [60, 80, 90]:
            for i, val in enumerate(cumulative):
                if val >= threshold:
                    results[threshold] = i + 1
                    break
        print(f"{embed_name.upper():<10} {results.get(60,'N/A'):<8} {results.get(80,'N/A'):<8} {results.get(90,'N/A'):<8}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    my_models = ["minilm", "bge", "e5"]
    my_ids = [f"R{i}" for i in range(1, 19)]

    # --- Unnormalized ---
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
    OUTPUT_FOLDER = "progress_meeting/plots_pca_variance"

    plot_pca_3components(EMBEDDINGS_FOLDER, OUTPUT_FOLDER, my_models, my_ids, normalized=False)
    plot_cumulative_variance(EMBEDDINGS_FOLDER, OUTPUT_FOLDER, my_models, my_ids, normalized=False)

    print("\nNow plotting normalized embeddings...\n")

    # --- Normalized ---
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"

    plot_pca_3components(EMBEDDINGS_FOLDER, OUTPUT_FOLDER, my_models, my_ids, normalized=True)
    plot_cumulative_variance(EMBEDDINGS_FOLDER, OUTPUT_FOLDER, my_models, my_ids, normalized=True)