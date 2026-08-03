##
# This script loads pre-computed text embeddings (e.g., MiniLM, BGE, E5)
# from .npy files and calculates the full pairwise Euclidean distance matrix
# for each model. It generates and saves high-resolution, annotated heatmaps 
# of these distances using Seaborn, visually grouping the text responses into 
# specific semantic categories (e.g., similar context vs. different context) 
# with boundary lines and custom labels. The entire pipeline is executed twice 
# to produce comparison plots for both unnormalized and normalized embedding vector spaces.
##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import os


def plot_distance_heatmap(data_dir, output_dir, models, group_config, normalized):
    """
    group_config: list of dicts, each defining one group.
    Example:
        [
            {"name": "Similar context, diff synonyms", "ids": ["R1", "R2", "R3"]},
            {"name": "Diff context, similar wording",  "ids": ["R4", "R5", "R6"]},
            {"name": "Everything different",           "ids": ["R7", "R8", "R9"]},
        ]
    Groups can have different sizes. Order of groups = order in heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    dir_path = Path(data_dir)

    # --- Flatten group_config into ordered id list and metadata ---
    all_ids       = []
    boundaries    = []   # row/col index where each group ends
    group_names   = []
    group_spans   = []   # (start, end) index per group

    cursor = 0
    for group in group_config:
        start = cursor
        for rid in group["ids"]:
            all_ids.append(rid)
            cursor += 1
        boundaries.append(cursor)        # end index of this group
        group_names.append(group["name"])
        group_spans.append((start, cursor))

    n = len(all_ids)

    for embed_name in models:
        embeddings = []
        valid_ids  = []

        # --- Load embeddings in group order ---
        for rid in all_ids:
            fname = f"{rid}_{embed_name}_norm.npy" if normalized else f"{rid}_{embed_name}.npy"
            fpath = dir_path / fname
            if fpath.exists():
                try:
                    vec = np.load(fpath)
                    if vec.ndim > 1:
                        vec = vec.flatten()
                    embeddings.append(vec)
                    valid_ids.append(rid)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
            else:
                print(f"Warning: {fpath} not found — skipping {rid}")

        if len(embeddings) < 2:
            print(f"Skipping {embed_name} — not enough data.")
            continue

        # --- Compute full pairwise distance matrix ---
        matrix      = np.stack(embeddings)
        dist_matrix = euclidean_distances(matrix, matrix)

        actual_n = len(valid_ids)

        # --- Plot ---
        fig_size = max(8, actual_n * 0.6)
        fig, ax  = plt.subplots(figsize=(fig_size + 2, fig_size))

        sns.heatmap(
            dist_matrix,
            ax=ax,
            xticklabels=valid_ids,
            yticklabels=valid_ids,
            cmap="rocket_r",
            vmin=0,
            vmax=dist_matrix.max(),
            annot=True,
            fmt=".2f",
            annot_kws={"size": max(5, 9 - actual_n // 5)},
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Euclidean Distance (L2)"}
        )

        # --- Map each id to its actual row position among loaded ids ---
        id_to_pos = {rid: i for i, rid in enumerate(valid_ids)}

        # --- Draw group separator lines (based on loaded rows) ---
        for group_name, (start, end) in zip(group_names, group_spans):
            positions = [id_to_pos[rid] for rid in all_ids[start:end] if rid in id_to_pos]
            if not positions:
                continue
            b = max(positions) + 1   # boundary after this group's last loaded row
            if 0 < b < actual_n:
                ax.axhline(b, color="white", linewidth=2.5)
                ax.axvline(b, color="white", linewidth=2.5)

        # --- Group labels on the LEFT, bracketed to their rows ---
        for group_name, (start, end) in zip(group_names, group_spans):
            positions = [id_to_pos[rid] for rid in all_ids[start:end] if rid in id_to_pos]
            if not positions:
                continue
            top_pos = min(positions)          # 0 = top row
            bot_pos = max(positions) + 1      # one past last row

            y_top = 1 - top_pos / actual_n
            y_bot = 1 - bot_pos / actual_n
            y_mid = (y_top + y_bot) / 2

            bx = -0.045                        # bracket x, just left of the R# ticks
            ax.plot([bx, bx], [y_top, y_bot], transform=ax.transAxes,
                    color="dimgray", lw=1.2, clip_on=False)
            ax.plot([bx, bx + 0.008], [y_top, y_top], transform=ax.transAxes,
                    color="dimgray", lw=1.2, clip_on=False)
            ax.plot([bx, bx + 0.008], [y_bot, y_bot], transform=ax.transAxes,
                    color="dimgray", lw=1.2, clip_on=False)

            ax.annotate(group_name, xy=(bx - 0.015, y_mid), xycoords="axes fraction",
                        fontsize=8, ha="center", va="center", color="dimgray",
                        rotation=90, annotation_clip=False)

        suffix = "Normalized" if normalized else "Unnormalized"
        ax.set_title(
            f"Pairwise Euclidean Distance Matrix — {embed_name.upper()} ({suffix})\n"
            f"Full {matrix.shape[1]}D embedding space  |  N={actual_n} responses  |  "
            f"{len(group_config)} groups",
            fontsize=12, pad=12
        )
        ax.set_xlabel("Response ID", fontsize=10)
        # y-axis label removed — it would collide with the rotated group labels.
        # The R# ticks already make the y-axis self-explanatory.
        ax.set_ylabel("")
        plt.tight_layout()

        suffix_file = "norm" if normalized else "unnorm"
        save_path   = os.path.join(output_dir, f"{embed_name}_heatmap_{suffix_file}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":

    # --- Configure your groups here ---
    # Any number of groups, any group size, any names
    GROUP_CONFIG = [
        {
            "name": "Similar context, diff synonyms",
            "ids":  ["R1", "R2", "R3"]
        },
        {
            "name": "Diff context, similar wording",
            "ids":  ["R4", "R5", "R6"]
        },
        {
            "name": "Everything different",
            "ids":  ["R7", "R8", "R9"]
        },
    ]

    my_models = ["minilm", "bge", "e5"]

    # --- Unnormalized ---
    plot_distance_heatmap(
        data_dir     = "progress_meeting/embeddings",
        output_dir   = "progress_meeting/plots_heatmap",
        models       = my_models,
        group_config = GROUP_CONFIG,
        normalized   = False
    )

    print("\nNow plotting normalized...\n")

    # --- Normalized ---
    plot_distance_heatmap(
        data_dir     = "progress_meeting/embeddings_norm",
        output_dir   = "progress_meeting/plots_heatmap_norm",
        models       = my_models,
        group_config = GROUP_CONFIG,
        normalized   = True
    )

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import euclidean_distances
# from pathlib import Path
# import os


# def plot_distance_heatmap(data_dir, output_dir, models, group_config, normalized):
#     """
#     group_config: list of dicts, each defining one group.
#     Example:
#         [
#             {"name": "Similar context, diff synonyms", "ids": ["R1", "R2", "R3"]},
#             {"name": "Diff context, similar wording",  "ids": ["R4", "R5", "R6"]},
#             {"name": "Everything different",           "ids": ["R7", "R8", "R9"]},
#         ]
#     Groups can have different sizes. Order of groups = order in heatmap.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     dir_path = Path(data_dir)

#     # --- Flatten group_config into ordered id list and metadata ---
#     all_ids       = []
#     boundaries    = []   # row/col index where each group ends
#     group_names   = []
#     group_spans   = []   # (start, end) index per group

#     cursor = 0
#     for group in group_config:
#         start = cursor
#         for rid in group["ids"]:
#             all_ids.append(rid)
#             cursor += 1
#         boundaries.append(cursor)        # end index of this group
#         group_names.append(group["name"])
#         group_spans.append((start, cursor))

#     n = len(all_ids)

#     for embed_name in models:
#         embeddings = []
#         valid_ids  = []

#         # --- Load embeddings in group order ---
#         for rid in all_ids:
#             fname = f"{rid}_{embed_name}_norm.npy" if normalized else f"{rid}_{embed_name}.npy"
#             fpath = dir_path / fname
#             if fpath.exists():
#                 try:
#                     vec = np.load(fpath)
#                     if vec.ndim > 1:
#                         vec = vec.flatten()
#                     embeddings.append(vec)
#                     valid_ids.append(rid)
#                 except Exception as e:
#                     print(f"Error loading {fname}: {e}")
#             else:
#                 print(f"Warning: {fpath} not found — skipping {rid}")

#         if len(embeddings) < 2:
#             print(f"Skipping {embed_name} — not enough data.")
#             continue

#         # --- Compute full pairwise distance matrix ---
#         matrix      = np.stack(embeddings)
#         dist_matrix = euclidean_distances(matrix, matrix)

#         actual_n = len(valid_ids)

#         # --- Plot ---
#         fig_size = max(8, actual_n * 0.6)
#         fig, ax  = plt.subplots(figsize=(fig_size + 2, fig_size))

#         sns.heatmap(
#             dist_matrix,
#             ax=ax,
#             xticklabels=valid_ids,
#             yticklabels=valid_ids,
#             cmap="rocket_r",
#             vmin=0,
#             vmax=dist_matrix.max(),
#             annot=True,
#             fmt=".2f",
#             annot_kws={"size": max(5, 9 - actual_n // 5)},
#             linewidths=0.3,
#             linecolor="white",
#             cbar_kws={"label": "Euclidean Distance (L2)"}
#         )

#         # --- Draw group separator lines ---
#         for b in boundaries[:-1]:   # skip last boundary (edge of matrix)
#             if b < actual_n:
#                 ax.axhline(b, color="white", linewidth=2.5)
#                 ax.axvline(b, color="white", linewidth=2.5)

#         # --- Group labels on the right side ---
#         for i, (group_name, (start, end)) in enumerate(zip(group_names, group_spans)):
#             if start >= actual_n:
#                 continue
#             actual_end = min(end, actual_n)
#             mid        = (start + actual_end) / 2

#             ax.annotate(
#                 group_name,
#                 xy=(1.22, 1 - mid / actual_n),
#                 xycoords="axes fraction",
#                 fontsize=8,
#                 ha="left",
#                 va="center",
#                 color="dimgray",
#                 annotation_clip=False
#             )

#         suffix = "Normalized" if normalized else "Unnormalized"
#         ax.set_title(
#             f"Pairwise Euclidean Distance Matrix — {embed_name.upper()} ({suffix})\n"
#             f"Full {matrix.shape[1]}D embedding space  |  N={actual_n} responses  |  "
#             f"{len(group_config)} groups",
#             fontsize=12, pad=12
#         )
#         ax.set_xlabel("Response ID", fontsize=10)
#         ax.set_ylabel("Response ID", fontsize=10)
#         plt.tight_layout()

#         suffix_file = "norm" if normalized else "unnorm"
#         save_path   = os.path.join(output_dir, f"{embed_name}_heatmap_{suffix_file}.png")
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Saved: {save_path}")
#         plt.close()


# if __name__ == "__main__":

#     # --- Configure your groups here ---
#     # Any number of groups, any group size, any names
#     GROUP_CONFIG = [
#         {
#             "name": "Similar context, diff synonyms",
#             "ids":  ["R1", "R2", "R3"]
#         },
#         {
#             "name": "Diff context, similar wording",
#             "ids":  ["R4", "R5", "R6"]
#         },
#         {
#             "name": "Everything different",
#             "ids":  ["R7", "R8", "R9"]
#         },
#     ]

#     my_models = ["minilm", "bge", "e5"]

#     # --- Unnormalized ---
#     plot_distance_heatmap(
#         data_dir     = "progress_meeting/embeddings",
#         output_dir   = "progress_meeting/plots_heatmap",
#         models       = my_models,
#         group_config = GROUP_CONFIG,
#         normalized   = False
#     )

#     print("\nNow plotting normalized...\n")

#     # --- Normalized ---
#     plot_distance_heatmap(
#         data_dir     = "progress_meeting/embeddings_norm",
#         output_dir   = "progress_meeting/plots_heatmap_norm",
#         models       = my_models,
#         group_config = GROUP_CONFIG,
#         normalized   = True
#     )

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.metrics.pairwise import euclidean_distances
# # from pathlib import Path
# # import os


# # def plot_distance_heatmap(data_dir: str, output_dir: str, models: list, ids: list, normalized: bool):
# #     """
# #     Loads all embeddings for each model, computes full pairwise
# #     Euclidean distance matrix in 768D space, saves as annotated heatmap.
# #     """
# #     os.makedirs(output_dir, exist_ok=True)
# #     dir_path = Path(data_dir)

# #     # Color bar for the group separators
# #     group_boundaries = [6, 12]  # after R6 and R12 draw a separator line

# #     for embed_name in models:
# #         embeddings = []
# #         valid_ids = []

# #         for current_id in ids:
# #             file_name = f"{current_id}_{embed_name}_norm.npy" if normalized else f"{current_id}_{embed_name}.npy"
# #             file_path = dir_path / file_name

# #             if file_path.exists():
# #                 try:
# #                     vec = np.load(file_path)
# #                     if vec.ndim > 1:
# #                         vec = vec.flatten()
# #                     embeddings.append(vec)
# #                     valid_ids.append(current_id)
# #                 except Exception as e:
# #                     print(f"Error loading {file_name}: {e}")
# #             else:
# #                 print(f"Warning: Could not find {file_path}")

# #         if len(embeddings) < 2:
# #             print(f"Skipping {embed_name} - not enough data.")
# #             continue

# #         # --- Compute full pairwise distance matrix in full D space ---
# #         matrix = np.stack(embeddings)                        # (N, D)
# #         dist_matrix = euclidean_distances(matrix, matrix)    # (N, N)

# #         n = len(valid_ids)

# #         # --- Plot heatmap ---
# #         fig, ax = plt.subplots(figsize=(10, 8))

# #         sns.heatmap(
# #             dist_matrix,
# #             ax=ax,
# #             xticklabels=valid_ids,
# #             yticklabels=valid_ids,
# #             cmap="rocket_r",           # dark = close, light = far
# #             vmin=0,
# #             vmax=dist_matrix.max(),
# #             annot=True,                # show values in each cell
# #             fmt=".2f",
# #             annot_kws={"size": 7},
# #             linewidths=0.3,
# #             linecolor="white",
# #             cbar_kws={"label": "Euclidean Distance (L2)"}
# #         )

# #         # --- Draw group separator lines ---
# #         for boundary in group_boundaries:
# #             if boundary < n:
# #                 ax.axhline(boundary, color="white", linewidth=2.5)
# #                 ax.axvline(boundary, color="white", linewidth=2.5)

# #         # --- Group labels on the side ---
# #         group_labels = {
# #             (0, 6):   "Similar context\ndiff synonyms",
# #             (6, 12):  "Diff context\nsimilar wording",
# #             (12, n):  "Everything\ndifferent",
# #         }
# #         for (start, end), label in group_labels.items():
# #             if start < n:
# #                 mid = (start + end) / 2
# #                 ax.annotate(
# #                     label,
# #                     xy=(1.18, 1 - mid / n),
# #                     xycoords="axes fraction",
# #                     fontsize=8,
# #                     ha="left",
# #                     va="center",
# #                     color="dimgray"
# #                 )

# #         suffix = "Normalized" if normalized else "Unnormalized"
# #         ax.set_title(
# #             f"Pairwise Euclidean Distance Matrix — {embed_name.upper()} ({suffix})\n"
# #             f"Full {matrix.shape[1]}D embedding space  |  N={n} responses",
# #             fontsize=12, pad=12
# #         )
# #         ax.set_xlabel("Response ID", fontsize=10)
# #         ax.set_ylabel("Response ID", fontsize=10)
# #         plt.tight_layout()

# #         suffix_file = "norm" if normalized else "unnorm"
# #         save_path = os.path.join(output_dir, f"{embed_name}_heatmap_{suffix_file}.png")
# #         plt.savefig(save_path, dpi=300, bbox_inches="tight")
# #         print(f"Saved: {save_path}")
# #         plt.close()


# # if __name__ == "__main__":
# #     EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
# #     PLOTS_FOLDER = "progress_meeting/plots_heatmap"

# #     my_models = ["minilm", "bge", "e5"]
# #     my_ids = [f"R{i}" for i in range(1, 19)]

# #     plot_distance_heatmap(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=False)

# #     print("\nNow plotting normalized embeddings...\n")

# #     EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"
# #     PLOTS_FOLDER = "progress_meeting/plots_heatmap_norm"

# #     plot_distance_heatmap(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=True)