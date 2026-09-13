import os
import glob
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_TYPE = "unnormalized"

# Target number of clusters (will automatically clamp if K > n_samples)
K = 3

EMBEDDING_DIRS = {
    "normalized": "data/processed/embeddings_norm/",
    "unnormalized": "data/processed/embeddings_unnorm/"
}

OUTPUT_DIR = "data/processed/plots"

ALLOWED_EMBEDDING_DIMENSIONS = {384}

KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

PCA_RANDOM_STATE = 42

REGION_STD = 2.0
REGION_ALPHA_2D = 0.15
REGION_ALPHA_3D = 0.12
ELLIPSOID_RESOLUTION = 30

POINT_SIZE = 100
STAR_POINT_SIZE = 250  # Prominent size for ground truth stars
POINT_ALPHA = 0.85
LABEL_FONT_SIZE = 13

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# ============================================================
# 2D COVARIANCE ELLIPSE
# ============================================================

def add_covariance_ellipse(
    ax,
    points,
    color,
    n_std=2.0,
    alpha=0.15
):
    if len(points) < 3:
        return

    center = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0]
        )
    )

    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        linewidth=2,
        alpha=alpha
    )

    ax.add_patch(ellipse)

# ============================================================
# PREPARE OUTPUT DIRECTORY & VALIDATE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

if EMBEDDING_TYPE not in EMBEDDING_DIRS:
    raise ValueError(
        f"Invalid EMBEDDING_TYPE: {EMBEDDING_TYPE}\n"
        f"Available options: {list(EMBEDDING_DIRS.keys())}"
    )

embedding_dir = EMBEDDING_DIRS[EMBEDDING_TYPE]


# ============================================================
# LOAD AND GROUP EMBEDDING FILES (BY PREFIX + DIMENSION)
# ============================================================

print("\n========================================")
print("SCANNING & GROUPING EMBEDDINGS")
print("========================================")
print(f"Embedding type: {EMBEDDING_TYPE}")
print(f"Embedding directory: {embedding_dir}")

embedding_files = sorted(glob.glob(os.path.join(embedding_dir, "*.npy")))

if not embedding_files:
    raise FileNotFoundError(f"No .npy files found in:\n{embedding_dir}")

grouped_files = defaultdict(list)

for filepath in embedding_files:
    filename = os.path.basename(filepath)
    
    # Load vector to inspect its dimension
    embedding = np.load(filepath)
    if embedding.ndim != 1 or embedding.shape[0] not in ALLOWED_EMBEDDING_DIMENSIONS:
        continue
    
    dim = embedding.shape[0]
    
    # Case 1: Ground truth format -> V1_A1_correctAnswer...
    v_match = re.match(r"^V\d+_(A\d+)_correctanswer", filename)
    if v_match:
        target_a_group = v_match.group(1)  # Extracts 'A1', 'A2', etc.
        grouped_files[f"{target_a_group}_{dim}d"].append(filepath)
        continue

    # Case 2: Standard ID prefix (A1, A10, R1, etc.)
    prefix_match = re.match(r"^([A-Za-z]+\d+)", filename)
    if prefix_match:
        prefix = prefix_match.group(1)
        if re.match(r"^R[1-9]$", prefix):
            group_key = f"R_all_{dim}d"
        else:
            group_key = f"{prefix}_{dim}d"
        grouped_files[group_key].append(filepath)
    else:
        grouped_files[f"other_{dim}d"].append(filepath)

print(f"Found {len(grouped_files)} analysis group(s): {list(grouped_files.keys())}")


# ============================================================
# PROCESS EACH GROUP SEPARATELY
# ============================================================

for group_name, files in grouped_files.items():

    print("\n" + "=" * 50)
    print(f"PROCESSING GROUP: {group_name} ({len(files)} files)")
    print("=" * 50)

    embeddings = []
    labels = []
    is_v_point = []

    for filepath in files:
        embedding = np.load(filepath)
        embeddings.append(embedding)

        filename = os.path.basename(filepath)
        point_label = filename.replace(f"_{EMBEDDING_TYPE}", "").replace(".npy", "")
        labels.append(point_label)

        # Mark whether this vector is a ground truth V* point
        is_v_point.append(filename.startswith("V") and "correctanswer" in filename)

    X = np.vstack(embeddings)
    n_samples = X.shape[0]
    group_dim = X.shape[1]
    is_v_point = np.array(is_v_point)

    print(f"Embedding matrix shape: {X.shape} (Dimension: {group_dim}D)")

    group_k = min(K, n_samples)
    if group_k < 2:
        print(f"Skipping {group_name}: Requires at least 2 points for clustering, found {n_samples}.")
        continue

    # --------------------------------------------------------
    # RUN K-MEANS
    # --------------------------------------------------------
    kmeans = KMeans(
        n_clusters=group_k,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT
    )
    cluster_labels = kmeans.fit_predict(X)

    # --------------------------------------------------------
    # RUN PCA (2D)
    # --------------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X)

    cmap = plt.get_cmap("tab10", group_k)

    # --------------------------------------------------------
    # 2D VISUALIZATION
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))

    for cluster_id in range(group_k):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_points = X_pca_2d[indices]
        cluster_color = cmap(cluster_id)

        # Draw Covariance Ellipse
        add_covariance_ellipse(
            ax=ax,
            points=cluster_points,
            color=cluster_color,
            n_std=REGION_STD,
            alpha=REGION_ALPHA_2D
        )

        # Plot standard responses (circles)
        regular_idx = [i for i in indices if not is_v_point[i]]
        if regular_idx:
            ax.scatter(
                X_pca_2d[regular_idx, 0],
                X_pca_2d[regular_idx, 1],
                color=cluster_color,
                marker="o",
                label=f"Cluster {cluster_id + 1}",
                s=POINT_SIZE,
                alpha=POINT_ALPHA
            )

        # Plot V* ground truth responses (stars)
        v_idx = [i for i in indices if is_v_point[i]]
        if v_idx:
            ax.scatter(
                X_pca_2d[v_idx, 0],
                X_pca_2d[v_idx, 1],
                color=cluster_color,
                marker="*",
                edgecolors="black",
                linewidths=1.2,
                label=f"Cluster {cluster_id + 1} (Ground Truth)",
                s=STAR_POINT_SIZE,
                alpha=1.0
            )

    for index, label in enumerate(labels):
        ax.annotate(
            label,
            (X_pca_2d[index, 0], X_pca_2d[index, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=LABEL_FONT_SIZE
        )

    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_title(f"PCA 2D + K-Means [{group_name}] (K={group_k}, {EMBEDDING_TYPE})")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    output_2d = os.path.join(
        OUTPUT_DIR,
        f"pca2d_kmeans_{group_name}_{EMBEDDING_TYPE}_k{group_k}.png"
    )
    plt.savefig(output_2d, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved 2D: {output_2d}")

print("\n========================================")
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("========================================")