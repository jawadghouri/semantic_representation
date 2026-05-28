from sklearn.cluster import KMeans


def cluster_embeddings(embeddings, n_clusters=5):
    """Groups embedding vectors into clusters using KMeans, dynamically adjusting

    n_clusters if the number of samples is too small.
    """
    n_samples = embeddings.shape[0]

    # Dynamic protection: Ensure n_clusters never exceeds the number of available samples
    actual_clusters = min(n_clusters, n_samples)

    # Edge case: If there is only 1 vector, clustering is trivial (everyone belongs to label 0)
    if actual_clusters <= 1:
        import numpy as np

        return np.zeros(n_samples, dtype=int)

    model = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)

    labels = model.fit_predict(embeddings)
    return labels