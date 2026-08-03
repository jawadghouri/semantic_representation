import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fisher_project(embeddings: np.ndarray, labels: list):
    """
    Fit Linear Discriminant Analysis on G/H-labelled embeddings and
    project them to 1D Fisher space.

    Returns:
        projections: (N,) array of 1D Fisher-space coordinates
        lda: fitted LDA object (for re-use on new points)
    """
    binary = np.array([0 if l == "G" else 1 for l in labels])

    if len(np.unique(binary)) < 2:
        # Only one class — return zeros; can't fit LDA
        return np.zeros(len(labels)), None

    lda = LinearDiscriminantAnalysis(n_components=1)
    projections = lda.fit_transform(embeddings, binary).ravel()
    return projections, lda


def fisher_project_groups(projections: np.ndarray, labels: list):
    """Split 1D projections into G and H arrays."""
    proj_g = projections[[i for i, l in enumerate(labels) if l == "G"]]
    proj_h = projections[[i for i, l in enumerate(labels) if l == "H"]]
    return proj_g, proj_h
