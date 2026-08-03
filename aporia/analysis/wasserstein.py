import numpy as np
from scipy.stats import wasserstein_distance as _wd


def compute_wasserstein(d_gg: np.ndarray, d_hh: np.ndarray) -> float:
    """
    Wasserstein-1 distance between the intra-genuine (D_GG) and
    intra-hallucinated (D_HH) distance distributions.
    A larger value means the two distributions are more separated,
    indicating that genuine responses cluster more tightly.
    """
    if len(d_gg) == 0 or len(d_hh) == 0:
        return float("nan")
    return float(_wd(d_gg, d_hh))
