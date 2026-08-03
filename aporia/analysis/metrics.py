import numpy as np


def mean_intra_distance(distances: np.ndarray) -> float:
    """Mean of an array of intra-class pairwise distances."""
    if len(distances) == 0:
        return float("nan")
    return float(np.mean(distances))


def inter_intra_ratio(d_gh: np.ndarray, d_gg: np.ndarray, d_hh: np.ndarray) -> float:
    """
    Ratio of mean inter-class distance to mean intra-class distance.
    Higher ratio → better separability between G and H.
    APORIA reports 1.13× in raw embedding space, 7.26× after Fisher projection.
    """
    inter = np.mean(d_gh) if len(d_gh) > 0 else float("nan")
    intra_mean = np.mean(
        np.concatenate([d_gg, d_hh])
    ) if (len(d_gg) + len(d_hh)) > 0 else float("nan")

    if intra_mean == 0 or np.isnan(intra_mean):
        return float("nan")
    return float(inter / intra_mean)


def fisher_inter_intra_ratio(proj_g: np.ndarray, proj_h: np.ndarray) -> float:
    """
    Inter-to-intra ratio in 1D Fisher space.
    Inter: distance between class means.
    Intra: average of within-class standard deviations.
    """
    if len(proj_g) == 0 or len(proj_h) == 0:
        return float("nan")

    mean_g = np.mean(proj_g)
    mean_h = np.mean(proj_h)
    inter = abs(mean_g - mean_h)

    std_g = np.std(proj_g)
    std_h = np.std(proj_h)
    intra = (std_g + std_h) / 2

    if intra == 0:
        return float("nan")
    return float(inter / intra)
