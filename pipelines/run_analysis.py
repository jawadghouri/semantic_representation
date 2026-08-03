import json
from pathlib import Path

import numpy as np

from analysis.prompt_similarity import compare_prompt_to_responses

EMBEDDING_DIR = Path("data/processed/embeddings")
OUTPUT_DIR = Path("results/similarities")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODELS = ["minilm", "e5", "bge"]

for embedding_model in EMBEDDING_MODELS:

    print(f"\nProcessing {embedding_model}")
    prompt_file = (EMBEDDING_DIR / f"prompt_{embedding_model}.npy")
    prompt_embedding = np.load(prompt_file)
    response_embeddings = {}

    for file in EMBEDDING_DIR.glob(f"*_{embedding_model}.npy"):
        model_name = file.stem.replace(f"_{embedding_model}", "")
        if model_name == "prompt":
            continue
        response_embeddings[model_name] = np.load(file)

    results = compare_prompt_to_responses(prompt_embedding, response_embeddings)

    output = {"embedding_model":embedding_model, "results":results}

    with open(OUTPUT_DIR / f"{embedding_model}.json", "w") as f:

        json.dump(output, f, indent=4)

    print(f"Saved {embedding_model}.json")

# from pathlib import Path
# import numpy as np

# from analysis.distance import compute_distance_matrix
# from analysis.statistics import compute_statistics
# from analysis.heatmap import save_heatmap
# from analysis.dendrogram import save_dendrogram
# from analysis.umap_plot import save_umap
# from analysis.report import save_report



# EMBEDDING_DIR = Path(
#     "data/processed/embeddings"
# )

# FIGURE_DIR = Path(
#     "results/figures"
# )

# METRIC_DIR = Path(
#     "results/metrics"
# )

# SUMMARY_DIR = Path(
#     "results/summaries"
# )


# embedding_models = [
#     "minilm",
#     "e5",
#     "bge"
# ]


# for embedding_model in embedding_models:

#     embeddings = []

#     labels = []

#     for file in EMBEDDING_DIR.glob(
#             f"*_{embedding_model}.npy"
#     ):

#         labels.append(
#             file.stem.replace(
#                 f"_{embedding_model}",
#                 ""
#             )
#         )

#         embeddings.append(
#             np.load(file)
#         )

#     embeddings = np.vstack(
#         embeddings
#     )

#     statistics = (
#         compute_statistics(
#             embeddings
#         )
#     )

#     distance_matrix = (
#         compute_distance_matrix(
#             embeddings
#         )
#     )

#     np.save(
#         METRIC_DIR /
#         f"{embedding_model}_distance.npy",

#         distance_matrix
#     )

#     save_heatmap(
#         distance_matrix,
#         labels,
#         FIGURE_DIR /
#         f"{embedding_model}_heatmap.png"
#     )

#     save_dendrogram(
#         embeddings,
#         labels,
#         FIGURE_DIR /
#         f"{embedding_model}_dendrogram.png"
#     )

#     save_umap(
#         embeddings,
#         labels,
#         FIGURE_DIR /
#         f"{embedding_model}_umap.png"
#     )

#     save_report(
#         statistics,
#         labels,
#         distance_matrix,
#         SUMMARY_DIR /
#         f"{embedding_model}.json"
#     )

#     print(
#         f"Finished {embedding_model}"
#     )

# # import json
# # from pathlib import Path
# # import numpy as np

# # from analysis.clustering_analysis import cluster_embeddings
# # from analysis.cross_model import compute_cross_model_agreement
# # from analysis.embedding_statistics import compute_statistics
# # from analysis.similarity_analysis import compute_distance_matrix, save_euclidean_heatmap
# # from analysis.visualization import generate_umap, save_umap_plot

# # EMBEDDING_DIR = Path("data/processed/embeddings")
# # FIGURE_DIR = Path("results/figures")
# # METRIC_DIR = Path("results/metrics")
# # SUMMARY_DIR = Path("results/summaries")

# # for directory in [FIGURE_DIR, METRIC_DIR, SUMMARY_DIR]:
# #     directory.mkdir(parents=True, exist_ok=True)

# # # Tracks distance matrices by LLM for cross-model agreement calculations
# # llm_groups = {"llama": {}, "mistral": {}, "phi": {}}

# # # ---------------------------------------------------
# # # STEP 1: COMPUTE PER-MODEL METRICS & VISUALS
# # # ---------------------------------------------------
# # for file in EMBEDDING_DIR.glob("*.npy"):
# #     print(f"\nProcessing {file.name}")
# #     embeddings = np.load(file)

# #     parts = file.stem.split("_")
# #     llm_name = parts[0]
# #     embed_name = parts[1]

# #     # Calculate raw vector distribution properties
# #     stats = compute_statistics(embeddings)
# #     with open(SUMMARY_DIR / f"{file.stem}.json", "w") as f:
# #         json.dump(stats, f, indent=4)

# #     # Calculate raw Euclidean distances and save Seaborn graphic maps
# #     dist_matrix = compute_distance_matrix(embeddings)
# #     np.save(METRIC_DIR / f"{file.stem}_distance.npy", dist_matrix)
# #     save_euclidean_heatmap(dist_matrix, FIGURE_DIR / f"{file.stem}_heatmap.png")

# #     # Retain the matrix in memory for cross-model rank steps next
# #     llm_groups[llm_name][embed_name] = dist_matrix

# #     # KMeans partitioning
# #     labels = cluster_embeddings(embeddings, n_clusters=5)
# #     np.save(METRIC_DIR / f"{file.stem}_clusters.npy", labels)

# #     # High-dimensional compression plots
# #     reduced = generate_umap(embeddings)
# #     save_umap_plot(reduced, labels, FIGURE_DIR / f"{file.stem}_umap.png")

# # # ---------------------------------------------------
# # # STEP 2: CROSS-MODEL RANK CORRELATION (SPEARMAN R)
# # # ---------------------------------------------------
# # print("\nCalculating Cross-Model Spearman r Agreement (Euclidean Space)...")
# # agreement_results = {}

# # for llm_name, models in llm_groups.items():
# #     agreement_results[llm_name] = {}
# #     model_names = list(models.keys())

# #     for i in range(len(model_names)):
# #         for j in range(i + 1, len(model_names)):
# #             m1, m2 = model_names[i], model_names[j]

# #             agreement = compute_cross_model_agreement(models[m1], models[m2])
# #             pair_key = f"{m1}_vs_{m2}"
# #             agreement_results[llm_name][pair_key] = agreement

# # with open(SUMMARY_DIR / "cross_model_agreement.json", "w") as f:
# #     json.dump(agreement_results, f, indent=4)

# # print("Saved cross-model evaluation to summaries/cross_model_agreement.json")
# # print("\nAnalysis completed successfully.")