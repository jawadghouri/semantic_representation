import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_distance_distributions(
    d_gg: np.ndarray,
    d_hh: np.ndarray,
    d_gh: np.ndarray,
    title: str,
    output_path: str,
):
    """Overlaid KDE plots of D_GG, D_HH, D_GH distance distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    def kde_plot(data, color, label):
        if len(data) < 2:
            return
        kde = gaussian_kde(data)
        x = np.linspace(data.min() - 0.1, data.max() + 0.1, 300)
        ax.fill_between(x, kde(x), alpha=0.25, color=color)
        ax.plot(x, kde(x), color=color, linewidth=2, label=label)
        ax.axvline(np.mean(data), color=color, linestyle="--", linewidth=1, alpha=0.7)

    kde_plot(d_gg, "green", f"D_GG (intra-genuine, n={len(d_gg)})")
    kde_plot(d_hh, "red", f"D_HH (intra-hallucinated, n={len(d_hh)})")
    kde_plot(d_gh, "gray", f"D_GH (inter-class, n={len(d_gh)})")

    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot -> {output_path}")
