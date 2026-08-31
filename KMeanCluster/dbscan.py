"""
PCA + DBSCAN (Density-Based Spatial Clustering) analysis of LLM response embeddings.

Workflow
--------
1. Load individual embedding vectors (supports 384, 768, 1024 dimensions).
2. Group files into subsets by Prefix AND Model/Dimension:
   - All 'R' prefixes (R1 to R9) grouped together per model.
   - Each 'A' prefix (A1, A2, ...) grouped independently along with its corresponding 'V*' ground truth vector.
3. For each group:
   - Combine into an N × D matrix.
   - Run DBSCAN density clustering on the original D-dimensional embeddings.
   - Identify dense clusters and flag noise points (label == -1).
   - Run PCA separately for 2D and 3D visualization.
   - Apply DBSCAN cluster labels to both PCA plots.
   - Plot regular response points as circles ('o'), ground truths as stars ('*'), and noise as gray crosses ('x').
   - Draw covariance ellipses and ellipsoids only for valid dense clusters (>= 3 points).
   - Save the respective 2D and 3D plots.
"""

import os
import glob
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_TYPE = "unnormalized"

# Metric for DBSCAN: 'cosine' (recommended for normalized embeddings) or 'euclidean'
DBSCAN_METRIC = "cosine" if EMBEDDING_TYPE == "normalized" else "euclidean"

# DBSCAN Parameters
# - EPS: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
#   (For cosine distance: typical range 0.15 - 0.40; for unnormalized Euclidean: 3.0 - 15.0 depending on dim)
DBSCAN_EPS = 0.25 if EMBEDDING_TYPE == "normalized" else 7.0

# - MIN_SAMPLES: Minimum points to form a core dense cluster (2 is recommended for small N per group)
DBSCAN_MIN_SAMPLES = 2

EMBEDDING_DIRS = {
    "normalized": "KMeanCluster/data/processed/embeddings_norm/",
    "unnormalized": "KMeanCluster/data/processed/embeddings_unnorm/"
}

OUTPUT_DIR = "KMeanCluster/data/processed/plots_dbscan"

ALLOWED_EMBEDDING_DIMENSIONS = {384, 768, 1024}

PCA_RANDOM_STATE = 42

REGION_STD = 2.0
REGION_ALPHA_2D = 0.15
REGION_ALPHA_3D = 0.12
ELLIPSOID_RESOLUTION = 30

POINT_SIZE = 100
STAR_POINT_SIZE = 250  # Prominent size for ground truth stars
POINT_ALPHA = 0.85
LABEL_FONT_SIZE = 8


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
# 3D COVARIANCE ELLIPSOID
# ============================================================

def plot_covariance_ellipsoid(
    ax,
    points,
    color,
    n_std=2.0,
    alpha=0.12,
    resolution=30
):
    if len(points) < 4:
        return

    center = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1)
    radii = n_std * np.sqrt(eigenvalues)
    ellipsoid = sphere * radii
    ellipsoid = ellipsoid @ eigenvectors.T
    ellipsoid += center

    ax.plot_surface(
        ellipsoid[:, :, 0],
        ellipsoid[:, :, 1],
        ellipsoid[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0
    )


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
print("SCANNING & GROUPING EMBEDDINGS FOR DBSCAN")
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
    
    # Case 1: Ground truth format -> V1_A1_correctanswer_...
    v_match = re.match(r"^V\d+_(A\d+)_correctanswer", filename)
    if v_match:
        target_a_group = v_match.group(1)
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
# PROCESS EACH GROUP SEPARATELY WITH DBSCAN
# ============================================================

for group_name, files in grouped_files.items():

    print("\n" + "=" * 50)
    print(f"PROCESSING DBSCAN GROUP: {group_name} ({len(files)} files)")
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

    if n_samples < 2:
        print(f"Skipping {group_name}: Requires at least 2 points, found {n_samples}.")
        continue

    # --------------------------------------------------------
    # 1. RUN DBSCAN
    # --------------------------------------------------------
    dbscan = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric=DBSCAN_METRIC
    )
    cluster_labels = dbscan.fit_predict(X)

    unique_labels = set(cluster_labels)
    n_clusters_found = len(unique_labels - {-1})
    n_noise_found = np.sum(cluster_labels == -1)

    print(f"DBSCAN Results: {n_clusters_found} cluster(s) discovered, {n_noise_found} noise point(s)")

    # --------------------------------------------------------
    # 2. RUN PCA (2D & 3D)
    # --------------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X)

    n_pca_3d_comp = min(3, n_samples)
    pca_3d = PCA(n_components=n_pca_3d_comp, random_state=PCA_RANDOM_STATE)
    X_pca_3d_raw = pca_3d.fit_transform(X)

    if n_pca_3d_comp < 3:
        X_pca_3d = np.zeros((n_samples, 3))
        X_pca_3d[:, :n_pca_3d_comp] = X_pca_3d_raw
    else:
        X_pca_3d = X_pca_3d_raw

    # Dynamic color palette (excluding noise)
    cmap = plt.get_cmap("tab10", max(1, n_clusters_found))

    # --------------------------------------------------------
    # 3. 2D VISUALIZATION
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))

    for k in unique_labels:
        indices = np.where(cluster_labels == k)[0]
        cluster_points = X_pca_2d[indices]

        if k == -1:
            # Noise points -> Gray 'x'
            color = "#808080"
            label_text = "Noise / Outlier"
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=color,
                marker="x",
                label=label_text,
                s=POINT_SIZE + 20,
                linewidths=2,
                alpha=0.8
            )
        else:
            color = cmap(k)
            label_text = f"Cluster {k + 1}"

            # Draw Covariance Ellipse for dense clusters
            add_covariance_ellipse(
                ax=ax,
                points=cluster_points,
                color=color,
                n_std=REGION_STD,
                alpha=REGION_ALPHA_2D
            )

            # Standard responses (circles)
            reg_idx = [i for i in indices if not is_v_point[i]]
            if reg_idx:
                ax.scatter(
                    X_pca_2d[reg_idx, 0],
                    X_pca_2d[reg_idx, 1],
                    color=color,
                    marker="o",
                    label=label_text,
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA
                )

            # Ground truth responses (stars)
            v_idx = [i for i in indices if is_v_point[i]]
            if v_idx:
                ax.scatter(
                    X_pca_2d[v_idx, 0],
                    X_pca_2d[v_idx, 1],
                    color=color,
                    marker="*",
                    edgecolors="black",
                    linewidths=1.2,
                    label=f"{label_text} (Ground Truth)",
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
    ax.set_title(f"PCA 2D + DBSCAN [{group_name}] (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}, {EMBEDDING_TYPE})")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    output_2d = os.path.join(
        OUTPUT_DIR,
        f"pca2d_dbscan_{group_name}_{EMBEDDING_TYPE}.png"
    )
    plt.savefig(output_2d, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 4. 3D VISUALIZATION
    # --------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for k in unique_labels:
        indices = np.where(cluster_labels == k)[0]
        cluster_points = X_pca_3d[indices]

        if k == -1:
            # Noise points
            color = "#808080"
            label_text = "Noise / Outlier"
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2],
                color=color,
                marker="x",
                label=label_text,
                s=POINT_SIZE + 20,
                linewidths=2,
                alpha=0.8
            )
        else:
            color = cmap(k)
            label_text = f"Cluster {k + 1}"

            # Draw Covariance Ellipsoid
            plot_covariance_ellipsoid(
                ax=ax,
                points=cluster_points,
                color=color,
                n_std=REGION_STD,
                alpha=REGION_ALPHA_3D,
                resolution=ELLIPSOID_RESOLUTION
            )

            # Standard responses (circles)
            reg_idx = [i for i in indices if not is_v_point[i]]
            if reg_idx:
                ax.scatter(
                    X_pca_3d[reg_idx, 0],
                    X_pca_3d[reg_idx, 1],
                    X_pca_3d[reg_idx, 2],
                    color=color,
                    marker="o",
                    label=label_text,
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA
                )

            # Ground truth responses (stars)
            v_idx = [i for i in indices if is_v_point[i]]
            if v_idx:
                ax.scatter(
                    X_pca_3d[v_idx, 0],
                    X_pca_3d[v_idx, 1],
                    X_pca_3d[v_idx, 2],
                    color=color,
                    marker="*",
                    edgecolors="black",
                    linewidths=1.2,
                    label=f"{label_text} (Ground Truth)",
                    s=STAR_POINT_SIZE,
                    alpha=1.0
                )

    for index, label in enumerate(labels):
        ax.text(
            X_pca_3d[index, 0],
            X_pca_3d[index, 1],
            X_pca_3d[index, 2],
            label,
            fontsize=LABEL_FONT_SIZE
        )

    pc3_var = pca_3d.explained_variance_ratio_[2] * 100 if n_pca_3d_comp >= 3 else 0.0
    ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_zlabel(f"PC3 ({pc3_var:.2f}% variance)")
    ax.set_title(f"PCA 3D + DBSCAN [{group_name}] (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}, {EMBEDDING_TYPE})")
    ax.legend()
    plt.tight_layout()

    output_3d = os.path.join(
        OUTPUT_DIR,
        f"pca3d_dbscan_{group_name}_{EMBEDDING_TYPE}.png"
    )
    plt.savefig(output_3d, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved 2D: {output_2d}")
    print(f"✓ Saved 3D: {output_3d}")

print("\n========================================")
print("ALL DBSCAN PLOTS GENERATED")
print("========================================")