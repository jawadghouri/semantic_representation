import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from pathlib import Path
import os


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
GROUP_CONFIG = {
    "R1":  ("Similar context, diff synonyms", "steelblue"),
    "R2":  ("Similar context, diff synonyms", "steelblue"),
    "R3":  ("Similar context, diff synonyms", "steelblue"),
    "R4":  ("Diff context, similar wording",  "tomato"),
    "R5":  ("Diff context, similar wording",  "tomato"),
    "R6":  ("Diff context, similar wording",  "tomato"),
    "R7": ("Everything different",           "seagreen"),
    "R8": ("Everything different",           "seagreen"),
    "R9": ("Everything different",           "seagreen"),

}

MODEL_COLORS = {
    "minilm": "darkorchid",
    "bge":    "darkorange",
    "e5":     "deepskyblue",
}


def load_all_embeddings(data_dir, ids, embed_name, normalized):
    """Load all response embeddings for one model. Returns (matrix, valid_ids)."""
    dir_path = Path(data_dir)
    embeddings, valid_ids = [], []
    for rid in ids:
        fname = f"{rid}_{embed_name}_norm.npy" if normalized else f"{rid}_{embed_name}.npy"
        fpath = dir_path / fname
        if fpath.exists():
            vec = np.load(fpath)
            if vec.ndim > 1:
                vec = vec.flatten()
            embeddings.append(vec)
            valid_ids.append(rid)
        else:
            print(f"Warning: {fpath} not found")
    return (np.stack(embeddings), valid_ids) if embeddings else (None, [])


def plot_paragraph_centric_A(data_dir, output_dir, models, ids, normalized):
    """
    Option A — PCA fitted separately per model.
    One plot per paragraph (R1-R9).
    Each plot has one subplot per model showing where the target
    paragraph lands relative to all others in that model's 2D PCA space.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix     = "Normalized" if normalized else "Unnormalized"
    suffix_file = "norm"      if normalized else "unnorm"

    # --- Pre-fit one PCA per model using ALL paragraphs ---
    # This gives a stable coordinate system so each paragraph
    # is plotted relative to the full population
    model_pca_data = {}   # embed_name -> {"reduced": ..., "valid_ids": ..., "explained": ...}

    for embed_name in models:
        matrix, valid_ids = load_all_embeddings(data_dir, ids, embed_name, normalized)
        if matrix is None:
            continue
        pca = PCA(n_components=2, random_state=42)
        reduced  = pca.fit_transform(matrix)
        explained = pca.explained_variance_ratio_ * 100
        model_pca_data[embed_name] = {
            "reduced":   reduced,
            "valid_ids": valid_ids,
            "explained": explained,
        }

    # --- One figure per paragraph ---
    for target_id in ids:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
        if len(models) == 1:
            axes = [axes]

        fig.suptitle(
            f"Paragraph-centric PCA — {target_id}  ({suffix})\n"
            f"Group: {GROUP_CONFIG[target_id][0]}",
            fontsize=13
        )

        for ax, embed_name in zip(axes, models):
            if embed_name not in model_pca_data:
                ax.set_visible(False)
                continue

            data      = model_pca_data[embed_name]
            reduced   = data["reduced"]
            valid_ids = data["valid_ids"]
            explained = data["explained"]

            if target_id not in valid_ids:
                ax.set_visible(False)
                continue

            target_idx = valid_ids.index(target_id)

            # --- Plot all other points first (background) ---
            for i, rid in enumerate(valid_ids):
                if rid == target_id:
                    continue
                _, color = GROUP_CONFIG.get(rid, ("unknown", "gray"))
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

            # --- Plot target paragraph on top (highlighted) ---
            tx, ty = reduced[target_idx, 0], reduced[target_idx, 1]
            _, target_color = GROUP_CONFIG[target_id]
            ax.scatter(
                tx, ty,
                color=target_color,
                s=250,
                zorder=5,
                edgecolors="black",
                linewidths=1.5,
                marker="*"
            )
            ax.annotate(
                target_id,
                xy=(tx, ty),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color=target_color
            )

            ax.set_title(
                f"{embed_name.upper()}\nPC1: {explained[0]:.1f}%  PC2: {explained[1]:.1f}%",
                fontsize=11
            )
            ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=9)
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)

        # --- Shared legend ---
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=9,  label="Similar context, diff synonyms"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",    markersize=9,  label="Diff context, similar wording"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",  markersize=9,  label="Everything different"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black",     markersize=12, label=f"Target: {target_id}"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4,
                   fontsize=9, bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        save_path = os.path.join(output_dir, f"{target_id}_pca_A_{suffix_file}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    my_models = ["minilm", "bge", "e5"]
    my_ids    = [f"R{i}" for i in range(1, 10)]

    EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
    PLOTS_FOLDER      = "progress_meeting/plots_paragraph_A"
    plot_paragraph_centric_A(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=False)

    print("\nNow plotting normalized...\n")

    EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"
    PLOTS_FOLDER      = "progress_meeting/plots_paragraph_A_norm"
    plot_paragraph_centric_A(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=True)