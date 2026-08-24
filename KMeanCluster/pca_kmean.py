"""
PCA + K-Means analysis of LLM response embeddings.

Workflow
--------
1. Load individual 384-dimensional MiniLM embedding vectors.
2. Combine them into an N × 384 matrix.
3. Run K-Means on the original 384-dimensional embeddings.
4. Run PCA separately for 2D and 3D visualization.
5. Apply the K-Means cluster labels to both PCA plots.
6. Draw covariance ellipses and ellipsoids as visual cluster regions.

Important:
----------
K-Means is NOT performed on the PCA-reduced coordinates.

K-Means:
    N × 384 embeddings
        ↓
    Cluster labels

PCA:
    N × 384 embeddings
        ↓
    2D / 3D coordinates

Visualization:
    PCA coordinates + K-Means cluster labels

Each response is stored as an individual .npy file.
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Select embedding type
#
# Options:
#     "normalized"
#     "unnormalized"
# ------------------------------------------------------------

EMBEDDING_TYPE = "unnormalized"


# ------------------------------------------------------------
# Number of K-Means clusters
#
# Change only this value to test different K values.
#
# Examples:
#     K = 3
#     K = 4
#     K = 5
# ------------------------------------------------------------

K = 3


# ------------------------------------------------------------
# Input embedding directories
# ------------------------------------------------------------

EMBEDDING_DIRS = {
    "normalized": (
        "KMeanCluster/data/processed/embeddings_norm/"
    ),
    "unnormalized": (
        "KMeanCluster/data/processed/embeddings_unnorm/"
    )
}


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    "KMeanCluster/data/processed/plots"
)


# ------------------------------------------------------------
# Expected MiniLM embedding dimension
# ------------------------------------------------------------

EXPECTED_EMBEDDING_DIMENSION = 384


# ------------------------------------------------------------
# K-Means configuration
# ------------------------------------------------------------

KMEANS_RANDOM_STATE = 42

KMEANS_N_INIT = 10


# ------------------------------------------------------------
# PCA configuration
# ------------------------------------------------------------

PCA_RANDOM_STATE = 42


# ------------------------------------------------------------
# Visual region configuration
#
# REGION_STD controls the size of the covariance
# ellipse / ellipsoid.
#
# 2.0 means approximately two standard deviations.
# ------------------------------------------------------------

REGION_STD = 2.0

REGION_ALPHA_2D = 0.15

REGION_ALPHA_3D = 0.12

ELLIPSOID_RESOLUTION = 30


# ------------------------------------------------------------
# Plot point configuration
# ------------------------------------------------------------

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
    """
    Draw a covariance ellipse around a 2D cluster.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the ellipse.

    points : numpy.ndarray
        Cluster coordinates with shape:
            (number_of_points, 2)

    color
        Color of the cluster.

    n_std : float
        Number of standard deviations controlling
        the ellipse size.

    alpha : float
        Transparency of the ellipse.
    """

    # --------------------------------------------------------
    # At least 3 points are required for a meaningful
    # 2D covariance region.
    # --------------------------------------------------------

    if len(points) < 3:
        return

    # --------------------------------------------------------
    # Calculate cluster center
    # --------------------------------------------------------

    center = np.mean(
        points,
        axis=0
    )

    # --------------------------------------------------------
    # Calculate covariance matrix
    #
    # Shape:
    #     2 × 2
    # --------------------------------------------------------

    covariance = np.cov(
        points,
        rowvar=False
    )

    # --------------------------------------------------------
    # Eigenvalue decomposition
    #
    # Eigenvalues:
    #     Determine ellipse axis lengths.
    #
    # Eigenvectors:
    #     Determine ellipse orientation.
    # --------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Sort largest eigenvalue first.

    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]

    eigenvectors = eigenvectors[:, order]

    # --------------------------------------------------------
    # Prevent numerical issues caused by tiny negative
    # eigenvalues from floating-point precision.
    # --------------------------------------------------------

    eigenvalues = np.clip(
        eigenvalues,
        a_min=0,
        a_max=None
    )

    # --------------------------------------------------------
    # Determine ellipse orientation.
    # --------------------------------------------------------

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0]
        )
    )

    # --------------------------------------------------------
    # Calculate ellipse width and height.
    # --------------------------------------------------------

    width = (
        2
        * n_std
        * np.sqrt(eigenvalues[0])
    )

    height = (
        2
        * n_std
        * np.sqrt(eigenvalues[1])
    )

    # --------------------------------------------------------
    # Create covariance ellipse.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Add ellipse to the plot.
    # --------------------------------------------------------

    ax.add_patch(
        ellipse
    )


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
    """
    Draw a covariance ellipsoid around a 3D cluster.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis on which to draw the ellipsoid.

    points : numpy.ndarray
        Cluster coordinates with shape:
            (number_of_points, 3)

    color
        Color of the cluster.

    n_std : float
        Number of standard deviations controlling
        ellipsoid size.

    alpha : float
        Transparency of the ellipsoid.

    resolution : int
        Surface resolution.
    """

    # --------------------------------------------------------
    # A 3D covariance estimate requires at least 4 points.
    # --------------------------------------------------------

    if len(points) < 4:
        return

    # --------------------------------------------------------
    # Calculate cluster center.
    # --------------------------------------------------------

    center = np.mean(
        points,
        axis=0
    )

    # --------------------------------------------------------
    # Calculate 3 × 3 covariance matrix.
    # --------------------------------------------------------

    covariance = np.cov(
        points,
        rowvar=False
    )

    # --------------------------------------------------------
    # Eigenvalue decomposition.
    # --------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Sort from largest to smallest.

    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]

    eigenvectors = eigenvectors[:, order]

    # Prevent numerical precision problems.

    eigenvalues = np.clip(
        eigenvalues,
        a_min=0,
        a_max=None
    )

    # --------------------------------------------------------
    # Generate a unit sphere.
    # --------------------------------------------------------

    u = np.linspace(
        0,
        2 * np.pi,
        resolution
    )

    v = np.linspace(
        0,
        np.pi,
        resolution
    )

    x = np.outer(
        np.cos(u),
        np.sin(v)
    )

    y = np.outer(
        np.sin(u),
        np.sin(v)
    )

    z = np.outer(
        np.ones_like(u),
        np.cos(v)
    )

    # --------------------------------------------------------
    # Combine sphere coordinates.
    #
    # Shape:
    #     resolution × resolution × 3
    # --------------------------------------------------------

    sphere = np.stack(
        [x, y, z],
        axis=-1
    )

    # --------------------------------------------------------
    # Calculate ellipsoid radii.
    #
    # Eigenvalues determine the variance along
    # each principal direction.
    # --------------------------------------------------------

    radii = (
        n_std
        * np.sqrt(eigenvalues)
    )

    # --------------------------------------------------------
    # Scale unit sphere.
    # --------------------------------------------------------

    ellipsoid = sphere * radii

    # --------------------------------------------------------
    # Rotate ellipsoid using eigenvectors.
    # --------------------------------------------------------

    ellipsoid = (
        ellipsoid
        @ eigenvectors.T
    )

    # --------------------------------------------------------
    # Move ellipsoid to the cluster center.
    # --------------------------------------------------------

    ellipsoid += center

    # --------------------------------------------------------
    # Draw ellipsoid surface.
    # --------------------------------------------------------

    ax.plot_surface(
        ellipsoid[:, :, 0],
        ellipsoid[:, :, 1],
        ellipsoid[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0
    )


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VALIDATE EMBEDDING TYPE
# ============================================================

if EMBEDDING_TYPE not in EMBEDDING_DIRS:

    raise ValueError(
        f"Invalid EMBEDDING_TYPE: {EMBEDDING_TYPE}\n"
        f"Available options: "
        f"{list(EMBEDDING_DIRS.keys())}"
    )


embedding_dir = EMBEDDING_DIRS[
    EMBEDDING_TYPE
]


# ============================================================
# LOAD INDIVIDUAL EMBEDDING FILES
# ============================================================

print("\n========================================")
print("LOADING EMBEDDINGS")
print("========================================")

print(
    f"Embedding type: {EMBEDDING_TYPE}"
)

print(
    f"Embedding directory: {embedding_dir}"
)


embedding_files = sorted(
    glob.glob(
        os.path.join(
            embedding_dir,
            "*.npy"
        )
    )
)


# ============================================================
# VALIDATE FILES
# ============================================================

if not embedding_files:

    raise FileNotFoundError(
        f"No .npy files found in:\n"
        f"{embedding_dir}"
    )


print(
    f"Number of embedding files: "
    f"{len(embedding_files)}"
)


# ============================================================
# LOAD ALL INDIVIDUAL VECTORS
# ============================================================

embeddings = []

labels = []


for filepath in embedding_files:

    # --------------------------------------------------------
    # Load individual embedding.
    # --------------------------------------------------------

    embedding = np.load(
        filepath
    )

    # --------------------------------------------------------
    # Validate embedding shape.
    # --------------------------------------------------------

    if embedding.ndim != 1:

        raise ValueError(
            f"Expected a 1D embedding vector.\n"
            f"File: {filepath}\n"
            f"Actual shape: {embedding.shape}"
        )


    if (
        embedding.shape[0]
        != EXPECTED_EMBEDDING_DIMENSION
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_EMBEDDING_DIMENSION} "
            f"dimensions.\n"
            f"File: {filepath}\n"
            f"Actual dimension: "
            f"{embedding.shape[0]}"
        )


    # --------------------------------------------------------
    # Add embedding to list.
    # --------------------------------------------------------

    embeddings.append(
        embedding
    )


    # --------------------------------------------------------
    # Extract filename without extension.
    # --------------------------------------------------------

    filename = os.path.basename(
        filepath
    )

    # label = os.path.splitext(
    #     filename
    # )[0]

    label = os.path.splitext(filename)[0].split('_')[0]

    labels.append(
        label
    )


# ============================================================
# CREATE N × 384 EMBEDDING MATRIX
# ============================================================

X = np.vstack(
    embeddings
)


print("\nEmbedding matrix loaded.")

print(
    f"Matrix shape: {X.shape}"
)


# ============================================================
# VALIDATE K
# ============================================================

n_samples = X.shape[0]


if K < 2:

    raise ValueError(
        "K must be at least 2."
    )


if K > n_samples:

    raise ValueError(
        f"K={K} cannot be larger than "
        f"the number of responses "
        f"({n_samples})."
    )


# ============================================================
# RUN K-MEANS ON ORIGINAL 384-D EMBEDDINGS
# ============================================================

print("\n========================================")
print("RUNNING K-MEANS")
print("========================================")

print(
    f"Input shape: {X.shape}"
)

print(
    f"Number of clusters K: {K}"
)


kmeans = KMeans(
    n_clusters=K,
    random_state=KMEANS_RANDOM_STATE,
    n_init=KMEANS_N_INIT
)


cluster_labels = kmeans.fit_predict(
    X
)


unique_clusters = np.unique(
    cluster_labels
)


print(
    f"Clusters generated: "
    f"{len(unique_clusters)}"
)


# ============================================================
# PRINT CLUSTER ASSIGNMENTS
# ============================================================

print("\n========================================")
print("CLUSTER ASSIGNMENTS")
print("========================================")


for cluster_id in range(K):

    print(
        f"\nCluster {cluster_id + 1}:"
    )


    indices = np.where(
        cluster_labels == cluster_id
    )[0]


    for index in indices:

        print(
            f"    {labels[index]}"
        )


# ============================================================
# RUN PCA FOR 2D VISUALIZATION
# ============================================================

print("\n========================================")
print("RUNNING PCA → 2D")
print("========================================")


pca_2d = PCA(
    n_components=2,
    random_state=PCA_RANDOM_STATE
)


X_pca_2d = pca_2d.fit_transform(
    X
)


explained_variance_2d = (
    pca_2d
    .explained_variance_ratio_
    .sum()
)


print(
    f"PCA 2D shape: "
    f"{X_pca_2d.shape}"
)


print(
    f"Total explained variance: "
    f"{explained_variance_2d:.4f}"
)


# ============================================================
# RUN PCA FOR 3D VISUALIZATION
# ============================================================

print("\n========================================")
print("RUNNING PCA → 3D")
print("========================================")


pca_3d = PCA(
    n_components=3,
    random_state=PCA_RANDOM_STATE
)


X_pca_3d = pca_3d.fit_transform(
    X
)


explained_variance_3d = (
    pca_3d
    .explained_variance_ratio_
    .sum()
)


print(
    f"PCA 3D shape: "
    f"{X_pca_3d.shape}"
)


print(
    f"Total explained variance: "
    f"{explained_variance_3d:.4f}"
)


# ============================================================
# DYNAMIC CLUSTER COLORS
# ============================================================

# Colors are dynamically generated based on K.

cmap = plt.get_cmap(
    "tab10",
    K
)


# ============================================================
# CREATE 2D PCA + K-MEANS VISUALIZATION
# ============================================================

print("\n========================================")
print("CREATING 2D VISUALIZATION")
print("========================================")


fig, ax = plt.subplots(
    figsize=(12, 9)
)


# ------------------------------------------------------------
# Plot each cluster.
# ------------------------------------------------------------

for cluster_id in range(K):

    indices = np.where(
        cluster_labels == cluster_id
    )[0]


    cluster_points = X_pca_2d[
        indices
    ]


    cluster_color = cmap(
        cluster_id
    )


    # --------------------------------------------------------
    # Draw covariance ellipse.
    # --------------------------------------------------------

    add_covariance_ellipse(
        ax=ax,
        points=cluster_points,
        color=cluster_color,
        n_std=REGION_STD,
        alpha=REGION_ALPHA_2D
    )


    # --------------------------------------------------------
    # Plot cluster points.
    # --------------------------------------------------------

    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        color=cluster_color,
        label=f"Cluster {cluster_id + 1}",
        s=POINT_SIZE,
        alpha=POINT_ALPHA
    )


# ------------------------------------------------------------
# Add labels to each point.
# ------------------------------------------------------------

for index, label in enumerate(
    labels
):

    ax.annotate(
        label,
        (
            X_pca_2d[index, 0],
            X_pca_2d[index, 1]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=LABEL_FONT_SIZE
    )


# ------------------------------------------------------------
# Plot labels and title.
# ------------------------------------------------------------

ax.set_xlabel(
    f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.2f}% variance)"
)


ax.set_ylabel(
    f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.2f}% variance)"
)


ax.set_title(
    f"MiniLM PCA 2D + K-Means "
    f"(K={K}, {EMBEDDING_TYPE})"
)


ax.legend()


ax.grid(
    True,
    alpha=0.2
)


plt.tight_layout()


# ------------------------------------------------------------
# Save 2D plot.
# ------------------------------------------------------------

output_2d = os.path.join(
    OUTPUT_DIR,
    f"minilm_{EMBEDDING_TYPE}_"
    f"pca2d_kmeans_k{K}.png"
)


plt.savefig(
    output_2d,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


plt.close(
    fig
)


# ============================================================
# CREATE 3D PCA + K-MEANS VISUALIZATION
# ============================================================

print("\n========================================")
print("CREATING 3D VISUALIZATION")
print("========================================")


fig = plt.figure(
    figsize=(12, 9)
)


ax = fig.add_subplot(
    111,
    projection="3d"
)


# ------------------------------------------------------------
# Plot each cluster.
# ------------------------------------------------------------

for cluster_id in range(K):

    indices = np.where(
        cluster_labels == cluster_id
    )[0]


    cluster_points = X_pca_3d[
        indices
    ]


    cluster_color = cmap(
        cluster_id
    )


    # --------------------------------------------------------
    # Draw covariance ellipsoid.
    # --------------------------------------------------------

    plot_covariance_ellipsoid(
        ax=ax,
        points=cluster_points,
        color=cluster_color,
        n_std=REGION_STD,
        alpha=REGION_ALPHA_3D,
        resolution=ELLIPSOID_RESOLUTION
    )


    # --------------------------------------------------------
    # Plot cluster points.
    # --------------------------------------------------------

    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        cluster_points[:, 2],
        color=cluster_color,
        label=f"Cluster {cluster_id + 1}",
        s=POINT_SIZE,
        alpha=POINT_ALPHA
    )


# ------------------------------------------------------------
# Add text label to each point.
# ------------------------------------------------------------

for index, label in enumerate(
    labels
):

    ax.text(
        X_pca_3d[index, 0],
        X_pca_3d[index, 1],
        X_pca_3d[index, 2],
        label,
        fontsize=LABEL_FONT_SIZE
    )


# ------------------------------------------------------------
# Plot labels and title.
# ------------------------------------------------------------

ax.set_xlabel(
    f"PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.2f}% variance)"
)


ax.set_ylabel(
    f"PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.2f}% variance)"
)


ax.set_zlabel(
    f"PC3 ({pca_3d.explained_variance_ratio_[2] * 100:.2f}% variance)"
)


ax.set_title(
    f"MiniLM PCA 3D + K-Means "
    f"(K={K}, {EMBEDDING_TYPE})"
)


ax.legend()


plt.tight_layout()


# ------------------------------------------------------------
# Save 3D plot.
# ------------------------------------------------------------

output_3d = os.path.join(
    OUTPUT_DIR,
    f"minilm_{EMBEDDING_TYPE}_"
    f"pca3d_kmeans_k{K}.png"
)


plt.savefig(
    output_3d,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


plt.close(
    fig
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("PCA + K-MEANS ANALYSIS COMPLETE")
print("========================================")


print(
    f"Embedding type: "
    f"{EMBEDDING_TYPE}"
)


print(
    f"Number of responses: "
    f"{n_samples}"
)


print(
    f"Embedding dimension: "
    f"{X.shape[1]}"
)


print(
    f"K: {K}"
)


print(
    f"2D PCA explained variance: "
    f"{explained_variance_2d:.4f}"
)


print(
    f"3D PCA explained variance: "
    f"{explained_variance_3d:.4f}"
)


print(
    f"\n2D plot saved to:"
)


print(
    output_2d
)


print(
    f"\n3D plot saved to:"
)


print(
    output_3d
)

# """
# PCA + K-Means analysis of LLM response embeddings.

# K-Means is performed on the ORIGINAL 384-dimensional
# embedding vectors.

# PCA is used only for visualization:
#     - 2D PCA
#     - 3D PCA

# Each response has its own .npy embedding file.

# The value of K is controlled from one location.
# """

# import os
# import glob
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA


# # ============================================================
# # CONFIGURATION
# # ============================================================

# # ------------------------------------------------------------
# # Select embedding type
# # ------------------------------------------------------------
# # Options:
# #     "normalized"
# #     "unnormalized"
# # ------------------------------------------------------------

# EMBEDDING_TYPE = "unnormalized"


# # ------------------------------------------------------------
# # K-Means parameter
# # ------------------------------------------------------------

# K = 5


# # ------------------------------------------------------------
# # Input directories
# # ------------------------------------------------------------

# EMBEDDING_DIRS = {
#     "normalized": "KMeanCluster/data/processed/embeddings_norm",
#     "unnormalized": "KMeanCluster/data/processed/embeddings",
# }


# # ------------------------------------------------------------
# # Output directory
# # ------------------------------------------------------------

# OUTPUT_DIR = "KMeanCluster/data/processed/plots"


# # ------------------------------------------------------------
# # Reproducibility
# # ------------------------------------------------------------

# KMEANS_RANDOM_STATE = 42
# PCA_RANDOM_STATE = 42


# # ============================================================
# # PREPARE DIRECTORIES
# # ============================================================

# os.makedirs(
#     OUTPUT_DIR,
#     exist_ok=True
# )


# # ============================================================
# # SELECT EMBEDDING DIRECTORY
# # ============================================================

# if EMBEDDING_TYPE not in EMBEDDING_DIRS:
#     raise ValueError(
#         f"Invalid EMBEDDING_TYPE: {EMBEDDING_TYPE}. "
#         f"Choose from: {list(EMBEDDING_DIRS.keys())}"
#     )


# embedding_dir = EMBEDDING_DIRS[EMBEDDING_TYPE]


# # ============================================================
# # LOAD INDIVIDUAL EMBEDDINGS
# # ============================================================

# print("\n========================================")
# print("Loading embeddings")
# print("========================================")

# embedding_files = sorted(
#     glob.glob(
#         os.path.join(
#             embedding_dir,
#             "*.npy"
#         )
#     )
# )

# if not embedding_files:
#     raise FileNotFoundError(
#         f"No .npy files found in:\n{embedding_dir}"
#     )


# print(
#     f"Embedding type: {EMBEDDING_TYPE}"
# )

# print(
#     f"Embedding directory: {embedding_dir}"
# )

# print(
#     f"Number of embedding files: "
#     f"{len(embedding_files)}"
# )


# # ============================================================
# # LOAD VECTORS
# # ============================================================

# embeddings = []
# labels = []


# for filepath in embedding_files:

#     embedding = np.load(filepath)

#     # --------------------------------------------------------
#     # Validate individual embedding
#     # --------------------------------------------------------

#     if embedding.ndim != 1:
#         raise ValueError(
#             f"Expected a 1D embedding vector, "
#             f"but got shape {embedding.shape} "
#             f"for {filepath}"
#         )

#     if embedding.shape[0] != 384:
#         raise ValueError(
#             f"Expected 384 dimensions, "
#             f"but got {embedding.shape[0]} "
#             f"for {filepath}"
#         )

#     embeddings.append(embedding)

#     # --------------------------------------------------------
#     # Keep filename as identity
#     # --------------------------------------------------------

#     filename = os.path.basename(filepath)

#     labels.append(
#         os.path.splitext(filename)[0]
#     )


# # ============================================================
# # CREATE N × 384 MATRIX
# # ============================================================

# X = np.vstack(embeddings)


# print("\nEmbedding matrix:")
# print(
#     f"Shape: {X.shape}"
# )

# print(
#     f"Expected dimension: 384"
# )


# # ============================================================
# # VALIDATE K
# # ============================================================

# n_samples = X.shape[0]

# if K < 2:
#     raise ValueError(
#         "K must be at least 2."
#     )

# if K > n_samples:
#     raise ValueError(
#         f"K={K} is larger than the number "
#         f"of responses ({n_samples})."
#     )


# # ============================================================
# # K-MEANS
# # ============================================================

# print("\n========================================")
# print("Running K-Means")
# print("========================================")

# kmeans = KMeans(
#     n_clusters=K,
#     random_state=KMEANS_RANDOM_STATE,
#     n_init=10
# )

# cluster_labels = kmeans.fit_predict(X)


# print(
#     f"K = {K}"
# )

# print(
#     f"Clusters generated: "
#     f"{len(np.unique(cluster_labels))}"
# )


# # ============================================================
# # PRINT CLUSTER ASSIGNMENTS
# # ============================================================

# print("\nCluster assignments:")

# for cluster_id in range(K):

#     print(
#         f"\nCluster {cluster_id + 1}:"
#     )

#     indices = np.where(
#         cluster_labels == cluster_id
#     )[0]

#     for index in indices:

#         print(
#             f"    {labels[index]}"
#         )


# # ============================================================
# # PCA — 2D
# # ============================================================

# print("\n========================================")
# print("Running 2D PCA")
# print("========================================")

# pca_2d = PCA(
#     n_components=2,
#     random_state=PCA_RANDOM_STATE
# )

# X_pca_2d = pca_2d.fit_transform(X)


# explained_variance_2d = (
#     pca_2d.explained_variance_ratio_.sum()
# )


# print(
#     f"2D PCA shape: {X_pca_2d.shape}"
# )

# print(
#     f"Explained variance: "
#     f"{explained_variance_2d:.4f}"
# )


# # ============================================================
# # PCA — 3D
# # ============================================================

# print("\n========================================")
# print("Running 3D PCA")
# print("========================================")

# pca_3d = PCA(
#     n_components=3,
#     random_state=PCA_RANDOM_STATE
# )

# X_pca_3d = pca_3d.fit_transform(X)


# explained_variance_3d = (
#     pca_3d.explained_variance_ratio_.sum()
# )


# print(
#     f"3D PCA shape: {X_pca_3d.shape}"
# )

# print(
#     f"Explained variance: "
#     f"{explained_variance_3d:.4f}"
# )


# # ============================================================
# # DYNAMIC CLUSTER VISUALIZATION SETTINGS
# # ============================================================

# # We deliberately do NOT hard-code colors for K=3,
# # K=4, K=5, etc.

# # Matplotlib will automatically provide K distinct colors
# # from the selected colormap.

# cmap = plt.get_cmap(
#     "tab10",
#     K
# )


# # ============================================================
# # 2D PCA + K-MEANS PLOT
# # ============================================================

# print("\nCreating 2D plot...")

# fig, ax = plt.subplots(
#     figsize=(12, 9)
# )


# for cluster_id in range(K):

#     indices = np.where(
#         cluster_labels == cluster_id
#     )[0]

#     ax.scatter(
#         X_pca_2d[indices, 0],
#         X_pca_2d[indices, 1],
#         color=cmap(cluster_id),
#         label=f"Cluster {cluster_id + 1}",
#         s=100,
#         alpha=0.85
#     )


# # ------------------------------------------------------------
# # Add LLM labels
# # ------------------------------------------------------------

# for index, label in enumerate(labels):

#     ax.annotate(
#         label,
#         (
#             X_pca_2d[index, 0],
#             X_pca_2d[index, 1]
#         ),
#         xytext=(5, 5),
#         textcoords="offset points",
#         fontsize=8
#     )


# ax.set_xlabel(
#     "Principal Component 1"
# )

# ax.set_ylabel(
#     "Principal Component 2"
# )

# ax.set_title(
#     f"MiniLM — PCA 2D + K-Means "
#     f"(K={K}, {EMBEDDING_TYPE})"
# )

# ax.legend()

# ax.grid(
#     True,
#     alpha=0.2
# )

# plt.tight_layout()


# output_2d = os.path.join(
#     OUTPUT_DIR,
#     f"minilm_{EMBEDDING_TYPE}_pca2d_kmeans_k{K}.png"
# )

# plt.savefig(
#     output_2d,
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# plt.close(fig)


# # ============================================================
# # 3D PCA + K-MEANS PLOT
# # ============================================================

# print("\nCreating 3D plot...")

# fig = plt.figure(
#     figsize=(12, 9)
# )

# ax = fig.add_subplot(
#     111,
#     projection="3d"
# )


# for cluster_id in range(K):

#     indices = np.where(
#         cluster_labels == cluster_id
#     )[0]

#     ax.scatter(
#         X_pca_3d[indices, 0],
#         X_pca_3d[indices, 1],
#         X_pca_3d[indices, 2],
#         color=cmap(cluster_id),
#         label=f"Cluster {cluster_id + 1}",
#         s=100,
#         alpha=0.85
#     )


# # ------------------------------------------------------------
# # Add LLM labels
# # ------------------------------------------------------------

# for index, label in enumerate(labels):

#     ax.text(
#         X_pca_3d[index, 0],
#         X_pca_3d[index, 1],
#         X_pca_3d[index, 2],
#         label,
#         fontsize=8
#     )


# ax.set_xlabel(
#     "Principal Component 1"
# )

# ax.set_ylabel(
#     "Principal Component 2"
# )

# ax.set_zlabel(
#     "Principal Component 3"
# )

# ax.set_title(
#     f"MiniLM — PCA 3D + K-Means "
#     f"(K={K}, {EMBEDDING_TYPE})"
# )

# ax.legend()

# plt.tight_layout()


# output_3d = os.path.join(
#     OUTPUT_DIR,
#     f"minilm_{EMBEDDING_TYPE}_pca3d_kmeans_k{K}.png"
# )

# plt.savefig(
#     output_3d,
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# plt.close(fig)


# # ============================================================
# # FINAL SUMMARY
# # ============================================================

# print("\n========================================")
# print("PCA + K-Means analysis complete")
# print("========================================")

# print(
#     f"Embedding type: {EMBEDDING_TYPE}"
# )

# print(
#     f"Number of responses: {n_samples}"
# )

# print(
#     f"Embedding dimensions: {X.shape[1]}"
# )

# print(
#     f"K: {K}"
# )

# print(
#     f"2D explained variance: "
#     f"{explained_variance_2d:.4f}"
# )

# print(
#     f"3D explained variance: "
#     f"{explained_variance_3d:.4f}"
# )

# print(
#     f"\n2D plot saved to:\n{output_2d}"
# )

# print(
#     f"\n3D plot saved to:\n{output_3d}"
# )