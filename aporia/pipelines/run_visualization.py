import os
import numpy as np

from utils.io_utils import load_json, load_numpy
from aporia.visualization.heatmap import plot_heatmap
from aporia.visualization.distribution import plot_distance_distributions
from aporia.visualization.pca_plot import plot_pca
from aporia.visualization.fisher_plot import plot_fisher

EMB_DIR = "data/aporia/embeddings"
LABELS_DIR = "data/aporia/labels"
DISTANCES_DIR = "data/aporia/results/distances"
FIGS_DIR = "data/aporia/results/figures"

LLM_NAMES = ["llama", "mistral", "phi"]
EMBEDDER_NAMES = ["minilm", "bge", "e5"]
PROMPT_IDS = ["P1", "P2", "P3", "P4", "P5"]


def run_pipeline():
    for llm in LLM_NAMES:
        labels_data = load_json(f"{LABELS_DIR}/{llm}_labels.json")
        labels_by_pid = {item["prompt_id"]: item["labels"] for item in labels_data}

        for emb_name in EMBEDDER_NAMES:
            for pid in PROMPT_IDS:
                emb_path = f"{EMB_DIR}/{llm}_{emb_name}_{pid}.npy"
                dist_path = f"{DISTANCES_DIR}/{llm}_{emb_name}_{pid}.json"

                if not os.path.exists(emb_path) or not os.path.exists(dist_path):
                    print(f"Skipping {llm}/{emb_name}/{pid} — files missing.")
                    continue

                embeddings = load_numpy(emb_path)
                labels = labels_by_pid.get(pid, [])
                dist_data = load_json(dist_path)

                d_gg = np.array(dist_data["d_gg"])
                d_hh = np.array(dist_data["d_hh"])
                d_gh = np.array(dist_data["d_gh"])
                proj_g = np.array(dist_data["proj_g"])
                proj_h = np.array(dist_data["proj_h"])

                tag = f"{llm}_{emb_name}_{pid}"
                base_title = f"{llm.upper()} | {emb_name.upper()} | {pid}"

                # Heatmap
                plot_heatmap(
                    embeddings, labels,
                    title=f"Pairwise Distance Heatmap — {base_title}",
                    output_path=f"{FIGS_DIR}/heatmaps/{tag}.png",
                )

                # Distance distributions
                plot_distance_distributions(
                    d_gg, d_hh, d_gh,
                    title=f"Distance Distributions — {base_title}",
                    output_path=f"{FIGS_DIR}/distributions/{tag}.png",
                )

                # PCA
                plot_pca(
                    embeddings, labels,
                    title=f"PCA Scatter — {base_title}",
                    output_path=f"{FIGS_DIR}/pca/{tag}.png",
                )

                # Fisher 1D
                plot_fisher(
                    proj_g, proj_h,
                    title=f"Fisher Projection — {base_title}",
                    output_path=f"{FIGS_DIR}/fisher/{tag}.png",
                )

    print("\nVisualization complete.")


if __name__ == "__main__":
    run_pipeline()
