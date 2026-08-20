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
    "R7":  ("Everything different",           "seagreen"),
    "R8":  ("Everything different",           "seagreen"),  
    "R9":  ("Everything different",           "seagreen"),
}

MODEL_STYLES = {
    "minilm": {"color": "darkorchid", "marker": "o"},
    "bge":    {"color": "darkorange", "marker": "s"},
    "e5":     {"color": "deepskyblue","marker": "^"},
}


def load_all_embeddings(data_dir, ids, embed_name, normalized):
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


def plot_paragraph_centric_B_3D(data_dir, output_dir, models, ids, normalized):
    """
    3D Version of Option B — all model embeddings concatenated into one matrix,
    single PCA fitted once across all models and all paragraphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix      = "Normalized" if normalized else "Unnormalized"
    suffix_file = "norm"       if normalized else "unnorm"

    # --- Build concatenated matrix across all models ---
    all_vectors = []
    all_tags    = []   # (rid, embed_name) per row

    for embed_name in models:
        matrix, valid_ids = load_all_embeddings(data_dir, ids, embed_name, normalized)
        if matrix is None:
            continue
        all_vectors.append(matrix)
        for rid in valid_ids:
            all_tags.append((rid, embed_name))

    if not all_vectors:
        print("No data found.")
        return

    # Pad to same dimensionality if models differ
    max_dim = max(v.shape[1] for v in all_vectors)
    padded  = []
    for v in all_vectors:
        if v.shape[1] < max_dim:
            pad = np.zeros((v.shape[0], max_dim - v.shape[1]))
            v   = np.hstack([v, pad])
        padded.append(v)

    full_matrix = np.vstack(padded)

    # --- Fit single PCA on the full matrix (Now with 3 components) ---
    pca       = PCA(n_components=3, random_state=42)
    reduced   = pca.fit_transform(full_matrix)   # (N_total, 3)
    explained = pca.explained_variance_ratio_ * 100

    # Calculate a tiny offset for 3D text labels based on data spread
    z_offset = (reduced[:, 2].max() - reduced[:, 2].min()) * 0.02

    # --- One figure per paragraph ---
    for target_id in ids:
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection='3d') # Create 3D axis

        # --- Plot all points (background, faded) ---
        for i, (rid, embed_name) in enumerate(all_tags):
            if rid == target_id:
                continue
            _, group_color = GROUP_CONFIG.get(rid, ("unknown", "gray"))
            style = MODEL_STYLES[embed_name]
            ax.scatter(
                reduced[i, 0], reduced[i, 1], reduced[i, 2], # Added Z-axis
                color=group_color,
                marker=style["marker"],
                s=55,
                alpha=0.25,
                zorder=2
            )

        # --- Plot target paragraph for each model (highlighted) ---
        for i, (rid, embed_name) in enumerate(all_tags):
            if rid != target_id:
                continue
            style = MODEL_STYLES[embed_name]
            
            ax.scatter(
                reduced[i, 0], reduced[i, 1], reduced[i, 2], # Added Z-axis
                color=style["color"],
                marker=style["marker"],
                s=300,
                zorder=5,
                edgecolors="black",
                linewidths=1.5
            )
            
            # 3D Text Annotation
            ax.text(
                reduced[i, 0], reduced[i, 1], reduced[i, 2] + z_offset, # Slightly raised in Z
                f"{target_id}\n({embed_name.upper()})",
                fontsize=9,
                fontweight="bold",
                color=style["color"],
                zorder=6
            )

        # --- Legend: groups (shape=circle) + models (shape=marker) ---
        group_legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=9,  label="Similar context, diff synonyms"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",    markersize=9,  label="Diff context, similar wording"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",  markersize=9,  label="Everything different"),
        ]
        model_legend = [
            Line2D([0], [0], marker=MODEL_STYLES[m]["marker"], color="w",
                   markerfacecolor=MODEL_STYLES[m]["color"], markersize=10,
                   label=f"{m.upper()} (target)")
            for m in models
        ]
        
        # In 3D plots, moving the legend slightly outside the plot area helps prevent overlapping
        ax.legend(
            handles=group_legend + model_legend,
            loc="upper left", bbox_to_anchor=(1.05, 0.95), 
            fontsize=8, title="Background / Target", title_fontsize=8
        )

        ax.set_title(
            f"3D Paragraph-centric PCA — {target_id}  ({suffix})\n"
            f"Group: {GROUP_CONFIG[target_id][0]}  |  "
            f"Total Variance: {explained[0]+explained[1]+explained[2]:.1f}%",
            fontsize=11
        )
        
        # Set 3D axis labels
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=10)
        ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)", fontsize=10)
        
        # Adjust viewing angle for better perspective (optional)
        ax.view_init(elev=20, azim=45) 

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{target_id}_pca_B_3D_{suffix_file}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.5)
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    my_models = ["minilm", "bge", "e5"]
    my_ids    = [f"R{i}" for i in range(1, 10)]

    EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
    PLOTS_FOLDER      = "progress_meeting/plots_paragraph_B_3D"
    plot_paragraph_centric_B_3D(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=False)

    print("\nNow plotting normalized...\n")

    EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"
    PLOTS_FOLDER      = "progress_meeting/plots_paragraph_B_norm_3D"
    plot_paragraph_centric_B_3D(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=True)