"""
Model-centric pairwise distance heatmaps.

Creates one heatmap per embedding model, showing all responses' pairwise
Euclidean distances in full-dimension space. Rows/columns grouped by
semantic constraint groups using flexible GROUP_CONFIG.

Interpretation:
- Low values (blue): responses are semantically similar
- High values (red): responses are semantically distant
- Block-diagonal structure: group structure is reflected in embedding space
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Optional

from config.groups import GROUP_CONFIG, EMBEDDING_MODELS, DATA_DIR, OUTPUT_DIR


def build_id_to_group(group_config: List[Dict]) -> Dict[str, tuple]:
    """
    Build mapping from response ID to (group_name, color).

    Args:
        group_config: list of group dicts with 'name', 'ids', 'color'

    Returns:
        dict mapping ID -> (name, color)
    """
    id_to_group = {}
    for group in group_config:
        for rid in group["ids"]:
            id_to_group[rid] = (group["name"], group["color"])
    return id_to_group


def all_ids_from_config(group_config: List[Dict]) -> List[str]:
    """Extract all response IDs in order from GROUP_CONFIG."""
    ids = []
    for group in group_config:
        ids.extend(group["ids"])
    return ids


def load_embeddings_from_dir(
    data_dir: str,
    ids: List[str],
    model_name: str,
    normalized: bool = False
) -> tuple:
    """
    Load embeddings for a model.

    Args:
        data_dir: path to embeddings directory
        ids: list of response IDs
        model_name: name of embedding model
        normalized: whether to load normalized (_norm) or unnormalized

    Returns:
        (embeddings_array, valid_ids) where embeddings is shape (N, D)
    """
    dir_path = Path(data_dir)
    embeddings, valid_ids = [], []

    for rid in ids:
        suffix = "_norm" if normalized else ""
        fname = f"{rid}_{model_name}{suffix}.npy"
        fpath = dir_path / fname

        if fpath.exists():
            try:
                vec = np.load(fpath)
                if vec.ndim > 1:
                    vec = vec.flatten()
                embeddings.append(vec)
                valid_ids.append(rid)
            except Exception as e:
                print(f"Warning: error loading {fname}: {e}")
        else:
            print(f"Warning: {fpath} not found — skipping {rid}")

    if embeddings:
        return np.stack(embeddings), valid_ids
    else:
        return None, []


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.

    Args:
        embeddings: shape (N, D)

    Returns:
        distance_matrix: shape (N, N)
    """
    from sklearn.metrics.pairwise import euclidean_distances
    return euclidean_distances(embeddings, embeddings)


def plot_heatmap(
    dist_matrix: np.ndarray,
    ids: List[str],
    model_name: str,
    group_config: List[Dict],
    output_path: str,
    normalized: bool = False,
    figsize: tuple = None
):
    """
    Plot and save a distance heatmap.

    Args:
        dist_matrix: shape (N, N) pairwise distance matrix
        ids: list of N response IDs (row/column labels)
        model_name: embedding model name (for title)
        group_config: list of group dicts (for group boundaries & colors)
        output_path: where to save PNG
        normalized: whether data is normalized (for title)
        figsize: (width, height) tuple, auto-calculated if None
    """
    n = len(ids)
    id_to_group = build_id_to_group(group_config)

    if figsize is None:
        figsize = (max(8, n * 0.6) + 2, max(8, n * 0.6))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        dist_matrix,
        ax=ax,
        xticklabels=ids,
        yticklabels=ids,
        cmap="rocket_r",
        vmin=0,
        vmax=dist_matrix.max(),
        annot=True,
        fmt=".2f",
        annot_kws={"size": max(5, 10 - n // 5)},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Euclidean Distance (L2)"}
    )

    cursor = 0
    boundaries = []
    for group in group_config:
        cursor += len(group["ids"])
        boundaries.append(cursor)

    for b in boundaries[:-1]:
        if b < n:
            ax.axhline(b, color="white", linewidth=2.5)
            ax.axvline(b, color="white", linewidth=2.5)

    cursor = 0
    for i, group in enumerate(group_config):
        start = cursor
        end = cursor + len(group["ids"])
        if end > n:
            end = n
        mid = (start + end) / 2 / n

        ax.annotate(
            group["name"],
            xy=(1.22, 1 - mid),
            xycoords="axes fraction",
            fontsize=8,
            ha="left",
            va="center",
            color="dimgray",
            annotation_clip=False
        )

        cursor = end

    suffix_text = "Normalized" if normalized else "Unnormalized"
    ax.set_title(
        f"Pairwise Euclidean Distance Matrix — {model_name.upper()} ({suffix_text})\n"
        f"Full {dist_matrix.shape[1]}D embedding space  |  N={n} responses  |  "
        f"{len(group_config)} groups",
        fontsize=12,
        pad=12
    )
    ax.set_xlabel("Response ID", fontsize=10)
    ax.set_ylabel("Response ID", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_all_models(
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    models: List[str] = EMBEDDING_MODELS,
    group_config: List[Dict] = GROUP_CONFIG,
    normalized: bool = False
):
    """
    Generate heatmaps for all models.

    Args:
        data_dir: directory containing .npy embedding files
        output_dir: directory for output PNG files
        models: list of model names to process
        group_config: flexible grouping configuration
        normalized: whether to load normalized embeddings
    """
    os.makedirs(output_dir, exist_ok=True)

    ids = all_ids_from_config(group_config)

    suffix_text = "norm" if normalized else "unnorm"

    for model_name in models:
        embeddings, valid_ids = load_embeddings_from_dir(
            data_dir, ids, model_name, normalized
        )

        if embeddings is None or len(embeddings) < 2:
            print(f"⚠ Skipping {model_name} — not enough data")
            continue

        dist_matrix = compute_distance_matrix(embeddings)

        output_path = os.path.join(
            output_dir,
            f"{model_name}_heatmap_{suffix_text}.png"
        )

        plot_heatmap(
            dist_matrix,
            valid_ids,
            model_name,
            group_config,
            output_path,
            normalized=normalized
        )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HEATMAP PLOTS — Model-Centric Distance Analysis")
    print("=" * 70)

    print("\n[1/2] Plotting UNNORMALIZED heatmaps...")
    plot_all_models(normalized=False)

    print("\n[2/2] Plotting NORMALIZED heatmaps...")
    plot_all_models(normalized=True)

    print("\n✓ All heatmaps generated successfully!")
    print(f"✓ Check '{OUTPUT_DIR}/' for PNG files")
