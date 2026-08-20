"""
2D PCA scatter plots (Option A - paragraph-centric).

For each response, creates a subplot per embedding model showing where that
response lands in that model's 2D PCA space. Each model's PCA is fitted
independently (no cross-model distortion).

Interpretation:
- Target response shown as a large star, colored by its group
- Other responses shown as smaller points, faded, colored by their groups
- Close clustering = group structure reflected in embedding geometry
- Note: PC1+PC2 typically explain ~40-50% of variance (rest in PC3+)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from pathlib import Path
import os
from typing import Dict, List, Tuple

from config.groups import GROUP_CONFIG, EMBEDDING_MODELS, DATA_DIR, OUTPUT_DIR


def build_id_to_group(group_config: List[Dict]) -> Dict[str, tuple]:
    """Build mapping from response ID to (group_name, color)."""
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
) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings for a model."""
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


def plot_paragraph_centric_2d(
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    models: List[str] = EMBEDDING_MODELS,
    group_config: List[Dict] = GROUP_CONFIG,
    normalized: bool = False,
    random_state: int = 42
):
    """
    Generate 2D PCA scatter plots (Option A).

    One figure per response. Each figure has one subplot per model.
    Each subplot shows the target response's position in that model's
    2D PCA space, with other responses shown faded in the background.

    Args:
        data_dir: directory containing .npy embedding files
        output_dir: directory for output PNG files
        models: list of model names to process
        group_config: flexible grouping configuration
        normalized: whether to load normalized embeddings
        random_state: reproducibility seed for PCA
    """
    os.makedirs(output_dir, exist_ok=True)

    ids = all_ids_from_config(group_config)
    id_to_group = build_id_to_group(group_config)

    suffix_text = "norm" if normalized else "unnorm"

    model_pca_data = {}
    for model_name in models:
        embeddings, valid_ids = load_embeddings_from_dir(
            data_dir, ids, model_name, normalized
        )

        if embeddings is None or len(embeddings) < 2:
            print(f"⚠ Skipping {model_name} — not enough data for 2D PCA")
            continue

        pca = PCA(n_components=2, random_state=random_state)
        reduced = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_ * 100

        model_pca_data[model_name] = {
            "reduced": reduced,
            "valid_ids": valid_ids,
            "explained": explained,
        }

    if not model_pca_data:
        print("⚠ No models had enough data for 2D PCA")
        return

    print(f"✓ Fitted PCA for {len(model_pca_data)} models")

    for target_id in ids:
        if target_id not in id_to_group:
            print(f"⚠ {target_id} not in group config, skipping")
            continue

        fig, axes = plt.subplots(1, len(model_pca_data), figsize=(6 * len(model_pca_data), 5))
        if len(model_pca_data) == 1:
            axes = [axes]

        fig_title_group = id_to_group[target_id][0]
        fig.suptitle(
            f"2D PCA (Option A) — {target_id}  ({suffix_text.upper()})\n"
            f"Group: {fig_title_group}",
            fontsize=13
        )

        for ax_idx, model_name in enumerate(models):
            if model_name not in model_pca_data:
                axes[ax_idx].set_visible(False)
                continue

            data = model_pca_data[model_name]
            reduced = data["reduced"]
            valid_ids = data["valid_ids"]
            explained = data["explained"]

            if target_id not in valid_ids:
                axes[ax_idx].set_visible(False)
                continue

            target_idx = valid_ids.index(target_id)
            ax = axes[ax_idx]

            for i, rid in enumerate(valid_ids):
                if rid == target_id:
                    continue

                _, color = id_to_group.get(rid, ("unknown", "gray"))
                ax.scatter(
                    reduced[i, 0], reduced[i, 1],
                    color=color, s=60, alpha=0.35, zorder=2
                )
                ax.annotate(
                    rid,
                    xy=(reduced[i, 0], reduced[i, 1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    alpha=0.5
                )

            tx, ty = reduced[target_idx, 0], reduced[target_idx, 1]
            _, target_color = id_to_group[target_id]
            ax.scatter(
                tx, ty,
                color=target_color, s=250, zorder=5,
                edgecolors="black", linewidths=1.5, marker="*"
            )
            ax.annotate(
                target_id,
                xy=(tx, ty),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11, fontweight="bold",
                color=target_color
            )

            ax.set_title(
                f"{model_name.upper()}\nPC1: {explained[0]:.1f}%  PC2: {explained[1]:.1f}%",
                fontsize=11
            )
            ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=9)
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=g["color"],
                   markersize=9, label=g["name"])
            for g in group_config
        ]
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=12, label=f"Target: {target_id}")
        )
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=min(4, len(legend_elements)),
            fontsize=8,
            bbox_to_anchor=(0.5, -0.05)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])

        save_path = os.path.join(output_dir, f"{target_id}_pca2d_{suffix_text}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D PCA PLOTS (Option A) — Paragraph-Centric")
    print("=" * 70)

    print("\n[1/2] Plotting UNNORMALIZED 2D PCA...")
    plot_paragraph_centric_2d(normalized=False)

    print("\n[2/2] Plotting NORMALIZED 2D PCA...")
    plot_paragraph_centric_2d(normalized=True)

    print("\n✓ All 2D PCA plots generated successfully!")
    print(f"✓ Check '{OUTPUT_DIR}/' for PNG files")
