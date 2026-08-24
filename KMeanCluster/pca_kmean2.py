"""
PCA + K-Means analysis of LLM response embeddings.

Workflow
--------
1. Load individual embedding vectors (supports 384, 768, 1024 dimensions).
2. Group files into subsets by Prefix AND Model/Dimension:
   - All 'R' prefixes (R1 to R9) grouped together per model.
   - Each 'A' prefix (A1, A2, ...) grouped independently per model.
3. For each group:
   - Combine into an N × D matrix.
   - Run K-Means on the original D-dimensional embeddings.
   - Run PCA separately for 2D and 3D visualization.
   - Apply K-Means cluster labels to both PCA plots.
   - Draw covariance ellipses and ellipsoids as visual cluster regions.
   - Save the respective 2D and 3D plots.
"""

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
    "normalized": "KMeanCluster/data/processed/embeddings_norm/",
    "unnormalized": "KMeanCluster/data/processed/embeddings_unnorm/"
}

OUTPUT_DIR = "KMeanCluster/data/processed/plots"

ALLOWED_EMBEDDING_DIMENSIONS = {384, 768, 1024}

KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

PCA_RANDOM_STATE = 42

REGION_STD = 2.0
REGION_ALPHA_2D = 0.15
REGION_ALPHA_3D = 0.12
ELLIPSOID_RESOLUTION = 30

POINT_SIZE = 100
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
    
    # Match ID prefix (e.g., A1, A10, R1)
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

    for filepath in files:
        embedding = np.load(filepath)
        embeddings.append(embedding)

        filename = os.path.basename(filepath)
        point_label = filename.replace(f"_{EMBEDDING_TYPE}", "").replace(".npy", "")
        labels.append(point_label)

    X = np.vstack(embeddings)
    n_samples = X.shape[0]
    group_dim = X.shape[1]

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
    # RUN PCA (2D & 3D)
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

    cmap = plt.get_cmap("tab10", group_k)

    # --------------------------------------------------------
    # 2D VISUALIZATION
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))

    for cluster_id in range(group_k):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_points = X_pca_2d[indices]
        cluster_color = cmap(cluster_id)

        add_covariance_ellipse(
            ax=ax,
            points=cluster_points,
            color=cluster_color,
            n_std=REGION_STD,
            alpha=REGION_ALPHA_2D
        )

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=cluster_color,
            label=f"Cluster {cluster_id + 1}",
            s=POINT_SIZE,
            alpha=POINT_ALPHA
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

    # --------------------------------------------------------
    # 3D VISUALIZATION
    # --------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_id in range(group_k):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_points = X_pca_3d[indices]
        cluster_color = cmap(cluster_id)

        plot_covariance_ellipsoid(
            ax=ax,
            points=cluster_points,
            color=cluster_color,
            n_std=REGION_STD,
            alpha=REGION_ALPHA_3D,
            resolution=ELLIPSOID_RESOLUTION
        )

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            color=cluster_color,
            label=f"Cluster {cluster_id + 1}",
            s=POINT_SIZE,
            alpha=POINT_ALPHA
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
    ax.set_title(f"PCA 3D + K-Means [{group_name}] (K={group_k}, {EMBEDDING_TYPE})")
    ax.legend()
    plt.tight_layout()

    output_3d = os.path.join(
        OUTPUT_DIR,
        f"pca3d_kmeans_{group_name}_{EMBEDDING_TYPE}_k{group_k}.png"
    )
    plt.savefig(output_3d, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved 2D: {output_2d}")
    print(f"✓ Saved 3D: {output_3d}")

print("\n========================================")
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("========================================")

# """
# PCA + K-Means analysis of LLM response embeddings.

# Workflow
# --------
# 1. Load individual 384-dimensional MiniLM embedding vectors.
# 2. Group files into subsets:
#    - All 'R' prefixes (R1 to R9) grouped together.
#    - Each 'A' prefix (A1, A2, ...) treated as its own independent group.
# 3. For each group:
#    - Combine into an N × 384 matrix.
#    - Run K-Means on the original 384-dimensional embeddings.
#    - Run PCA separately for 2D and 3D visualization.
#    - Apply K-Means cluster labels to both PCA plots.
#    - Draw covariance ellipses and ellipsoids as visual cluster regions.
#    - Save the respective 2D and 3D plots.
# """

# import os
# import glob
# import re
# from collections import defaultdict

# import numpy as np
# import matplotlib.pyplot as plt

# from matplotlib.patches import Ellipse
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA


# # ============================================================
# # CONFIGURATION
# # ============================================================

# EMBEDDING_TYPE = "unnormalized"

# # Target number of clusters (will automatically clamp if K > n_samples)
# K = 3

# EMBEDDING_DIRS = {
#     "normalized": (
#         "KMeanCluster/data/processed/embeddings_norm/"
#     ),
#     "unnormalized": (
#         "KMeanCluster/data/processed/embeddings_unnorm/"
#     )
# }

# OUTPUT_DIR = (
#     "KMeanCluster/data/processed/plots"
# )

# EXPECTED_EMBEDDING_DIMENSION = 384

# KMEANS_RANDOM_STATE = 42
# KMEANS_N_INIT = 10

# PCA_RANDOM_STATE = 42

# REGION_STD = 2.0
# REGION_ALPHA_2D = 0.15
# REGION_ALPHA_3D = 0.12
# ELLIPSOID_RESOLUTION = 30

# POINT_SIZE = 100
# POINT_ALPHA = 0.85
# LABEL_FONT_SIZE = 8


# # ============================================================
# # 2D COVARIANCE ELLIPSE
# # ============================================================

# def add_covariance_ellipse(
#     ax,
#     points,
#     color,
#     n_std=2.0,
#     alpha=0.15
# ):
#     if len(points) < 3:
#         return

#     center = np.mean(points, axis=0)
#     covariance = np.cov(points, rowvar=False)

#     eigenvalues, eigenvectors = np.linalg.eigh(covariance)
#     order = eigenvalues.argsort()[::-1]
#     eigenvalues = eigenvalues[order]
#     eigenvectors = eigenvectors[:, order]

#     eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

#     angle = np.degrees(
#         np.arctan2(
#             eigenvectors[1, 0],
#             eigenvectors[0, 0]
#         )
#     )

#     width = 2 * n_std * np.sqrt(eigenvalues[0])
#     height = 2 * n_std * np.sqrt(eigenvalues[1])

#     ellipse = Ellipse(
#         xy=center,
#         width=width,
#         height=height,
#         angle=angle,
#         edgecolor=color,
#         facecolor=color,
#         linewidth=2,
#         alpha=alpha
#     )

#     ax.add_patch(ellipse)


# # ============================================================
# # 3D COVARIANCE ELLIPSOID
# # ============================================================

# def plot_covariance_ellipsoid(
#     ax,
#     points,
#     color,
#     n_std=2.0,
#     alpha=0.12,
#     resolution=30
# ):
#     if len(points) < 4:
#         return

#     center = np.mean(points, axis=0)
#     covariance = np.cov(points, rowvar=False)

#     eigenvalues, eigenvectors = np.linalg.eigh(covariance)
#     order = eigenvalues.argsort()[::-1]
#     eigenvalues = eigenvalues[order]
#     eigenvectors = eigenvectors[:, order]

#     eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

#     u = np.linspace(0, 2 * np.pi, resolution)
#     v = np.linspace(0, np.pi, resolution)

#     x = np.outer(np.cos(u), np.sin(v))
#     y = np.outer(np.sin(u), np.sin(v))
#     z = np.outer(np.ones_like(u), np.cos(v))

#     sphere = np.stack([x, y, z], axis=-1)
#     radii = n_std * np.sqrt(eigenvalues)
#     ellipsoid = sphere * radii
#     ellipsoid = ellipsoid @ eigenvectors.T
#     ellipsoid += center

#     ax.plot_surface(
#         ellipsoid[:, :, 0],
#         ellipsoid[:, :, 1],
#         ellipsoid[:, :, 2],
#         color=color,
#         alpha=alpha,
#         linewidth=0
#     )


# # ============================================================
# # PREPARE OUTPUT DIRECTORY & VALIDATE
# # ============================================================

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# if EMBEDDING_TYPE not in EMBEDDING_DIRS:
#     raise ValueError(
#         f"Invalid EMBEDDING_TYPE: {EMBEDDING_TYPE}\n"
#         f"Available options: {list(EMBEDDING_DIRS.keys())}"
#     )

# embedding_dir = EMBEDDING_DIRS[EMBEDDING_TYPE]


# # ============================================================
# # LOAD AND GROUP EMBEDDING FILES
# # ============================================================

# print("\n========================================")
# print("SCANNING & GROUPING EMBEDDINGS")
# print("========================================")
# print(f"Embedding type: {EMBEDDING_TYPE}")
# print(f"Embedding directory: {embedding_dir}")

# embedding_files = sorted(glob.glob(os.path.join(embedding_dir, "*.npy")))

# if not embedding_files:
#     raise FileNotFoundError(f"No .npy files found in:\n{embedding_dir}")

# # Group files:
# # - 'R_group': Contains all files starting with R1 to R9 (e.g., R1_*, R2_*)
# # - 'A1', 'A2', ...: Grouped individually
# grouped_files = defaultdict(list)

# for filepath in embedding_files:
#     filename = os.path.basename(filepath)
#     prefix_match = re.match(r"^([A-Za-z]+\d+)", filename)
    
#     if prefix_match:
#         prefix = prefix_match.group(1)
#         if re.match(r"^R[1-9]$", prefix):
#             grouped_files["R_all"].append(filepath)
#         else:
#             grouped_files[prefix].append(filepath)
#     else:
#         grouped_files["other"].append(filepath)

# print(f"Found {len(grouped_files)} analysis group(s): {list(grouped_files.keys())}")


# # ============================================================
# # PROCESS EACH GROUP SEPARATELY
# # ============================================================

# for group_name, files in grouped_files.items():

#     print("\n" + "=" * 50)
#     print(f"PROCESSING GROUP: {group_name} ({len(files)} files)")
#     print("=" * 50)

#     embeddings = []
#     labels = []

#     for filepath in files:
#         embedding = np.load(filepath)

#         if embedding.ndim != 1 or embedding.shape[0] != EXPECTED_EMBEDDING_DIMENSION:
#             raise ValueError(
#                 f"Invalid embedding dimensions in {filepath}. "
#                 f"Expected ({EXPECTED_EMBEDDING_DIMENSION},), got {embedding.shape}"
#             )

#         embeddings.append(embedding)

#         # Retain full identifier (e.g., A1_chatgpt or R1_0) for individual plot points
#         filename = os.path.basename(filepath)
#         point_label = filename.replace(f"_{EMBEDDING_TYPE}", "").replace(".npy", "")
#         labels.append(point_label)

#     X = np.vstack(embeddings)
#     n_samples = X.shape[0]

#     # Ensure K is valid for the group sample size
#     group_k = min(K, n_samples)
#     if group_k < 2:
#         print(f"Skipping {group_name}: Requires at least 2 points for clustering, found {n_samples}.")
#         continue

#     # --------------------------------------------------------
#     # RUN K-MEANS
#     # --------------------------------------------------------
#     kmeans = KMeans(
#         n_clusters=group_k,
#         random_state=KMEANS_RANDOM_STATE,
#         n_init=KMEANS_N_INIT
#     )
#     cluster_labels = kmeans.fit_predict(X)

#     # --------------------------------------------------------
#     # RUN PCA (2D & 3D)
#     # --------------------------------------------------------
#     pca_2d = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
#     X_pca_2d = pca_2d.fit_transform(X)

#     # If samples are fewer than 3, adjust 3D PCA components
#     n_pca_3d_comp = min(3, n_samples)
#     pca_3d = PCA(n_components=n_pca_3d_comp, random_state=PCA_RANDOM_STATE)
#     X_pca_3d_raw = pca_3d.fit_transform(X)

#     if n_pca_3d_comp < 3:
#         # Zero-pad if fewer than 3 dimensions exist
#         X_pca_3d = np.zeros((n_samples, 3))
#         X_pca_3d[:, :n_pca_3d_comp] = X_pca_3d_raw
#     else:
#         X_pca_3d = X_pca_3d_raw

#     cmap = plt.get_cmap("tab10", group_k)

#     # --------------------------------------------------------
#     # 2D VISUALIZATION
#     # --------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(12, 9))

#     for cluster_id in range(group_k):
#         indices = np.where(cluster_labels == cluster_id)[0]
#         cluster_points = X_pca_2d[indices]
#         cluster_color = cmap(cluster_id)

#         add_covariance_ellipse(
#             ax=ax,
#             points=cluster_points,
#             color=cluster_color,
#             n_std=REGION_STD,
#             alpha=REGION_ALPHA_2D
#         )

#         ax.scatter(
#             cluster_points[:, 0],
#             cluster_points[:, 1],
#             color=cluster_color,
#             label=f"Cluster {cluster_id + 1}",
#             s=POINT_SIZE,
#             alpha=POINT_ALPHA
#         )

#     for index, label in enumerate(labels):
#         ax.annotate(
#             label,
#             (X_pca_2d[index, 0], X_pca_2d[index, 1]),
#             xytext=(5, 5),
#             textcoords="offset points",
#             fontsize=LABEL_FONT_SIZE
#         )

#     ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.2f}% variance)")
#     ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.2f}% variance)")
#     ax.set_title(f"MiniLM PCA 2D + K-Means [{group_name}] (K={group_k}, {EMBEDDING_TYPE})")
#     ax.legend()
#     ax.grid(True, alpha=0.2)
#     plt.tight_layout()

#     output_2d = os.path.join(
#         OUTPUT_DIR,
#         f"minilm_{group_name}_{EMBEDDING_TYPE}_pca2d_kmeans_k{group_k}.png"
#     )
#     plt.savefig(output_2d, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     # --------------------------------------------------------
#     # 3D VISUALIZATION
#     # --------------------------------------------------------
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection="3d")

#     for cluster_id in range(group_k):
#         indices = np.where(cluster_labels == cluster_id)[0]
#         cluster_points = X_pca_3d[indices]
#         cluster_color = cmap(cluster_id)

#         plot_covariance_ellipsoid(
#             ax=ax,
#             points=cluster_points,
#             color=cluster_color,
#             n_std=REGION_STD,
#             alpha=REGION_ALPHA_3D,
#             resolution=ELLIPSOID_RESOLUTION
#         )

#         ax.scatter(
#             cluster_points[:, 0],
#             cluster_points[:, 1],
#             cluster_points[:, 2],
#             color=cluster_color,
#             label=f"Cluster {cluster_id + 1}",
#             s=POINT_SIZE,
#             alpha=POINT_ALPHA
#         )

#     for index, label in enumerate(labels):
#         ax.text(
#             X_pca_3d[index, 0],
#             X_pca_3d[index, 1],
#             X_pca_3d[index, 2],
#             label,
#             fontsize=LABEL_FONT_SIZE
#         )

#     pc3_var = pca_3d.explained_variance_ratio_[2] * 100 if n_pca_3d_comp >= 3 else 0.0
#     ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.2f}% variance)")
#     ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.2f}% variance)")
#     ax.set_zlabel(f"PC3 ({pc3_var:.2f}% variance)")
#     ax.set_title(f"MiniLM PCA 3D + K-Means [{group_name}] (K={group_k}, {EMBEDDING_TYPE})")
#     ax.legend()
#     plt.tight_layout()

#     output_3d = os.path.join(
#         OUTPUT_DIR,
#         f"minilm_{group_name}_{EMBEDDING_TYPE}_pca3d_kmeans_k{group_k}.png"
#     )
#     plt.savefig(output_3d, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"✓ Saved 2D: {output_2d}")
#     print(f"✓ Saved 3D: {output_3d}")

# print("\n========================================")
# print("ALL PLOTS GENERATED SUCCESSFULLY")
# print("========================================")