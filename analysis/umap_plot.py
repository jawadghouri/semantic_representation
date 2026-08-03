import umap
import matplotlib.pyplot as plt


def save_umap(
        embeddings,
        labels,
        output_file
):
    # 1. Determine the number of data samples in your stack
    n_samples = embeddings.shape[0]

    # 2. Configure safeguards dynamically if the dataset is small
    if n_samples < 15:
        # Clamp neighbors to the total available pairs (max size - 1)
        n_neighbors = min(2, n_samples - 1)
        # Use random initialization to skip crashing scipy linear algebra operations
        init_strategy = "random"
    else:
        n_neighbors = 15          # UMAP standard default
        init_strategy = "spectral" # UMAP standard default

    # 3. Initialize the reducer with dynamic safety switches
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        init=init_strategy,
        random_state=42
    )

    reduced = reducer.fit_transform(
        embeddings
    )

    # --- Your original visualization logic below remains identical ---
    plt.figure(
        figsize=(8, 6)
    )

    for i, label in enumerate(labels):

        plt.scatter(
            reduced[i, 0],
            reduced[i, 1]
        )

        plt.text(
            reduced[i, 0],
            reduced[i, 1],
            label
        )

    plt.savefig(
        output_file
    )

    plt.close()
# import umap
# import matplotlib.pyplot as plt


# def save_umap(
#         embeddings,
#         labels,
#         output_file
# ):

#     reducer = umap.UMAP(
#         random_state=42
#     )

#     reduced = reducer.fit_transform(
#         embeddings
#     )

#     plt.figure(
#         figsize=(8, 6)
#     )

#     for i, label in enumerate(labels):

#         plt.scatter(
#             reduced[i,0],
#             reduced[i,1]
#         )

#         plt.text(
#             reduced[i,0],
#             reduced[i,1],
#             label
#         )

#     plt.savefig(
#         output_file
#     )

#     plt.close()