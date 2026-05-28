from sklearn.cluster import KMeans


def run_kmeans(
        embeddings,
        n_clusters=5
):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        embeddings
    )

    return labels