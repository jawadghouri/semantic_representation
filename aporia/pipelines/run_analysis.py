import os
import numpy as np

from utils.io_utils import load_json, save_json, load_numpy
from aporia.analysis.pairwise import compute_pairwise_distances, extract_distance_sets
from aporia.analysis.wasserstein import compute_wasserstein
from aporia.analysis.fisher import fisher_project, fisher_project_groups
from aporia.analysis.metrics import (
    mean_intra_distance,
    inter_intra_ratio,
    fisher_inter_intra_ratio,
)

EMB_DIR = "data/aporia/embeddings"
LABELS_DIR = "data/aporia/labels"
RESULTS_DIR = "data/aporia/results"
DISTANCES_DIR = f"{RESULTS_DIR}/distances"
METRICS_DIR = f"{RESULTS_DIR}/metrics"

LLM_NAMES = ["llama", "mistral", "phi"]
EMBEDDER_NAMES = ["minilm", "bge", "e5"]
PROMPT_IDS = ["P1", "P2", "P3", "P4", "P5"]


def run_pipeline():
    all_metrics = {}

    for llm in LLM_NAMES:
        labels_data = load_json(f"{LABELS_DIR}/{llm}_labels.json")
        labels_by_pid = {item["prompt_id"]: item["labels"] for item in labels_data}

        for emb_name in EMBEDDER_NAMES:
            key = f"{llm}_{emb_name}"
            all_metrics[key] = {}

            for pid in PROMPT_IDS:
                emb_path = f"{EMB_DIR}/{llm}_{emb_name}_{pid}.npy"
                if not os.path.exists(emb_path):
                    print(f"Missing: {emb_path}, skipping.")
                    continue

                embeddings = load_numpy(emb_path)
                labels = labels_by_pid.get(pid, [])

                if len(labels) != len(embeddings):
                    print(f"Label/embedding mismatch for {key}/{pid}, skipping.")
                    continue

                # Phase 1: pairwise distances
                dist_matrix = compute_pairwise_distances(embeddings)
                d_gg, d_hh, d_gh = extract_distance_sets(dist_matrix, labels)

                # Wasserstein distance between D_GG and D_HH
                w_dist = compute_wasserstein(d_gg, d_hh)

                # Inter/intra ratio in raw embedding space
                ratio = inter_intra_ratio(d_gh, d_gg, d_hh)

                # Phase 2: Fisher projection
                projections, lda = fisher_project(embeddings, labels)
                proj_g, proj_h = fisher_project_groups(projections, labels)
                fisher_ratio = fisher_inter_intra_ratio(proj_g, proj_h)

                # Save distance arrays
                dist_out = {
                    "d_gg": d_gg.tolist(),
                    "d_hh": d_hh.tolist(),
                    "d_gh": d_gh.tolist(),
                    "projections": projections.tolist(),
                    "proj_g": proj_g.tolist(),
                    "proj_h": proj_h.tolist(),
                    "labels": labels,
                }
                save_json(dist_out, f"{DISTANCES_DIR}/{llm}_{emb_name}_{pid}.json")

                # Collect metrics
                all_metrics[key][pid] = {
                    "n_genuine": labels.count("G"),
                    "n_hallucinated": labels.count("H"),
                    "mean_d_gg": mean_intra_distance(d_gg),
                    "mean_d_hh": mean_intra_distance(d_hh),
                    "wasserstein_gg_hh": w_dist,
                    "inter_intra_ratio_raw": ratio,
                    "inter_intra_ratio_fisher": fisher_ratio,
                }

                print(
                    f"{key}/{pid}: G={labels.count('G')} H={labels.count('H')} "
                    f"W={w_dist:.4f} ratio_raw={ratio:.3f} ratio_fisher={fisher_ratio:.3f}"
                )

    save_json(all_metrics, f"{METRICS_DIR}/all_metrics.json")
    print(f"\nMetrics saved -> {METRICS_DIR}/all_metrics.json")
    print("Analysis complete.")


if __name__ == "__main__":
    run_pipeline()
