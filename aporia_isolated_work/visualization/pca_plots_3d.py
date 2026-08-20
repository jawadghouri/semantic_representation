"""
3D PCA scatter plots (Option A - paragraph-centric).

Extends 2D PCA to 3D. For each response, creates a subplot per embedding model
showing where that response lands in that model's 3D PCA space.

Each model's PCA is fitted independently (no cross-model distortion).

Interpretation:
- Same as 2D version but with one additional principal component
- Can reveal cluster structure not visible in 2D projection
- PC1+PC2+PC3 typically explain ~50-70% of variance
- Rotation/angle controls depth perception (can be adjusted for publication)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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


def plot_paragraph_centric_3d(
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    models: List[str] = EMBEDDING_MODELS,
    group_config: List[Dict] = GROUP_CONFIG,
    normalized: bool = False,
    random_state: int = 42,
    elev: float = 20,
    azim: float = -60
):
    """
    Generate 3D PCA scatter plots (Option A).

    One figure per response. Each figure has one 3D subplot per model.

    Args:
        data_dir: directory containing .npy embedding files
        output_dir: directory for output PNG files
        models: list of model names to process
        group_config: flexible grouping configuration
        normalized: whether to load normalized embeddings
        random_state: reproducibility seed for PCA
        elev: 3D view elevation angle (degrees)
        azim: 3D view azimuth angle (degrees)
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

        if embeddings is None or len(embeddings) < 3:
            print(f"⚠ Skipping {model_name} — need ≥3 samples for 3D PCA")
            continue

        n_components = min(3, embeddings.shape[0], embeddings.shape[1])
        if n_components < 3:
            print(f"⚠ Skipping {model_name} — only {n_components} components available")
            continue

        pca = PCA(n_components=3, random_state=random_state)
        reduced = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_ * 100

        model_pca_data[model_name] = {
            "reduced": reduced,
            "valid_ids": valid_ids,
            "explained": explained,
        }

    if not model_pca_data:
        print("⚠ No models had enough data for 3D PCA")
        return

    print(f"✓ Fitted 3D PCA for {len(model_pca_data)} models")

    for target_id in ids:
        if target_id not in id_to_group:
            print(f"⚠ {target_id} not in group config, skipping")
            continue

        n_models = len(model_pca_data)
        fig = plt.figure(figsize=(6.5 * n_models, 6))

        fig_title_group = id_to_group[target_id][0]
        fig.suptitle(
            f"3D PCA (Option A) — {target_id}  ({suffix_text.upper()})\n"
            f"Group: {fig_title_group}",
            fontsize=13
        )

        plot_idx = 0
        for model_name in models:
            if model_name not in model_pca_data:
                continue

            data = model_pca_data[model_name]
            reduced = data["reduced"]
            valid_ids = data["valid_ids"]
            explained = data["explained"]

            if target_id not in valid_ids:
                continue

            plot_idx += 1
            ax = fig.add_subplot(1, n_models, plot_idx, projection="3d")
            ax.view_init(elev=elev, azim=azim)

            target_idx = valid_ids.index(target_id)

            for i, rid in enumerate(valid_ids):
                if rid == target_id:
                    continue

                _, color = id_to_group.get(rid, ("unknown", "gray"))
                ax.scatter(
                    reduced[i, 0], reduced[i, 1], reduced[i, 2],
                    color=color, s=45, alpha=0.35, zorder=2
                )
                ax.text(
                    reduced[i, 0], reduced[i, 1], reduced[i, 2],
                    rid, fontsize=6.5, color=color, alpha=0.55
                )

            tx, ty, tz = reduced[target_idx]
            _, target_color = id_to_group[target_id]
            ax.scatter(
                tx, ty, tz,
                color=target_color, s=260, zorder=5,
                edgecolors="black", linewidths=1.5, marker="*"
            )
            ax.text(
                tx, ty, tz,
                f"  {target_id}",
                fontsize=10, fontweight="bold", color=target_color
            )

            ax.set_title(
                f"{model_name.upper()}\n"
                f"PC1:{explained[0]:.1f}%  PC2:{explained[1]:.1f}%  PC3:{explained[2]:.1f}%  "
                f"(Σ{sum(explained):.1f}%)",
                fontsize=10
            )
            ax.set_xlabel("PC1", fontsize=8)
            ax.set_ylabel("PC2", fontsize=8)
            ax.set_zlabel("PC3", fontsize=8)
            ax.tick_params(labelsize=7)

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

        save_path = os.path.join(output_dir, f"{target_id}_pca3d_{suffix_text}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("3D PCA PLOTS (Option A) — Paragraph-Centric")
    print("=" * 70)

    print("\n[1/2] Plotting UNNORMALIZED 3D PCA...")
    plot_paragraph_centric_3d(normalized=False)

    print("\n[2/2] Plotting NORMALIZED 3D PCA...")
    plot_paragraph_centric_3d(normalized=True)

    print("\n✓ All 3D PCA plots generated successfully!")
    print(f"✓ Check '{OUTPUT_DIR}/' for PNG files")
