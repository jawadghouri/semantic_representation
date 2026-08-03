import numpy as np
import matplotlib.pyplot as plt


def plot_fisher(proj_g: np.ndarray, proj_h: np.ndarray, title: str, output_path: str):
    """1D histogram of G and H projections in Fisher discriminant space."""
    fig, ax = plt.subplots(figsize=(8, 4))

    common_kw = dict(alpha=0.6, edgecolor="black", linewidth=0.5)
    bins = 10

    if len(proj_g) > 0:
        ax.hist(proj_g, bins=bins, color="green", label=f"Genuine (n={len(proj_g)})", **common_kw)
        ax.axvline(np.mean(proj_g), color="green", linestyle="--", linewidth=1.5, label="G mean")

    if len(proj_h) > 0:
        ax.hist(proj_h, bins=bins, color="red", label=f"Hallucinated (n={len(proj_h)})", **common_kw)
        ax.axvline(np.mean(proj_h), color="red", linestyle="--", linewidth=1.5, label="H mean")

    ax.set_xlabel("Fisher Discriminant Score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Fisher plot -> {output_path}")
