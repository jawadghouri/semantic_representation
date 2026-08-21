import sys
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load both normalized and unnormalized embeddings."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    # Load unnormalized
    bge_unnorm = np.load(embedding_dir / "new_dataset_bge_unnormalized.npy")
    e5_unnorm = np.load(embedding_dir / "new_dataset_e5_unnormalized.npy")
    minilm_unnorm = np.load(embedding_dir / "new_dataset_minilm_unnormalized.npy")

    # Load normalized
    bge_norm = np.load(embedding_dir / "new_dataset_bge_normalized.npy")
    e5_norm = np.load(embedding_dir / "new_dataset_e5_normalized.npy")
    minilm_norm = np.load(embedding_dir / "new_dataset_minilm_normalized.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'unnormalized': {'bge': bge_unnorm, 'e5': e5_unnorm, 'minilm': minilm_unnorm},
        'normalized': {'bge': bge_norm, 'e5': e5_norm, 'minilm': minilm_norm}
    }, metadata

def find_optimal_k(embeddings, k_range=range(2, 11)):
    """Find optimal K using multiple metrics."""
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    inertias = []

    print(f"\nTesting K values: {list(k_range)}")
    print("-" * 70)
    print(f"{'K':>3} | {'Silhouette':>12} | {'Davies-Bouldin':>15} | {'Calinski-Harabasz':>18} | {'Inertia':>12}")
    print("-" * 70)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        sil_score = silhouette_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)
        calinski_harabasz_scores.append(ch_score)
        inertias.append(inertia)

        print(f"{k:3d} | {sil_score:12.4f} | {db_score:15.4f} | {ch_score:18.2f} | {inertia:12.2f}")

    print("-" * 70)

    optimal_k_sil = k_range[np.argmax(silhouette_scores)]
    optimal_k_db = k_range[np.argmin(davies_bouldin_scores)]
    optimal_k_ch = k_range[np.argmax(calinski_harabasz_scores)]

    print(f"\nOptimal K recommendations:")
    print(f"  - Silhouette Score (higher is better): K = {optimal_k_sil} (score: {max(silhouette_scores):.4f})")
    print(f"  - Davies-Bouldin Index (lower is better): K = {optimal_k_db} (score: {min(davies_bouldin_scores):.4f})")
    print(f"  - Calinski-Harabasz Index (higher is better): K = {optimal_k_ch} (score: {max(calinski_harabasz_scores):.2f})")

    return {
        'k_values': list(k_range),
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'inertias': inertias,
        'optimal_k_silhouette': optimal_k_sil,
        'optimal_k_db': optimal_k_db,
        'optimal_k_ch': optimal_k_ch
    }

def cluster_with_k(embeddings, k, random_state=42):
    """Perform K-means clustering with specified K."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

def analyze_clusters_by_embedding_model(labels, metadata):
    """Analyze cluster distribution by embedding model."""
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS BY EMBEDDING MODEL")
    print("="*70)

    for cluster_id in sorted(set(labels)):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]

        embedding_dist = {}
        for response in cluster_responses:
            emb = response['model']
            embedding_dist[emb] = embedding_dist.get(emb, 0) + 1

        print(f"\nCluster {cluster_id} (n={np.sum(cluster_mask)})")
        print(f"  Embedding models:")
        for model, count in sorted(embedding_dist.items()):
            pct = (count / np.sum(cluster_mask)) * 100
            print(f"    {model:15} - {count:2d} responses ({pct:5.1f}%)")

def analyze_clusters_by_llm_model(labels, metadata):
    """Analyze cluster distribution by LLM model."""
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS BY LLM MODEL")
    print("="*70)

    for cluster_id in sorted(set(labels)):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]

        llm_dist = {}
        for response in cluster_responses:
            llm = response['model']
            llm_dist[llm] = llm_dist.get(llm, 0) + 1

        print(f"\nCluster {cluster_id} (n={np.sum(cluster_mask)})")
        print(f"  LLM models:")
        for llm, count in sorted(llm_dist.items()):
            pct = (count / np.sum(cluster_mask)) * 100
            print(f"    {llm:15} - {count:2d} responses ({pct:5.1f}%)")

def save_clustering_results(results, output_dir, normalization_type):
    """Save clustering results to JSON."""
    output_file = output_dir / f"kmeans_{normalization_type}_analysis.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in results.items()}
        json.dump(results_serializable, f, indent=2)
    print(f"\n✓ Saved analysis to: {output_file}")

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "new/clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*70)

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    embedding_models = ['bge', 'e5', 'minilm']
    normalization_types = ['unnormalized', 'normalized']

    # Analyze each embedding model with each normalization type
    for emb_model in embedding_models:
        print("\n\n" + "="*70)
        print(f"EMBEDDING MODEL: {emb_model.upper()}")
        print("="*70)

        for norm_type in normalization_types:
            print(f"\n\n{'─'*70}")
            print(f"{norm_type.upper()} EMBEDDINGS")
            print(f"{'─'*70}")

            embeddings = embeddings_dict[norm_type][emb_model]

            # Find optimal K
            metrics = find_optimal_k(embeddings, k_range=range(2, 11))

            # Use silhouette score optimal K
            optimal_k = metrics['optimal_k_silhouette']
            print(f"\n→ Using K = {optimal_k} (based on Silhouette Score)")

            # Perform clustering
            labels, kmeans = cluster_with_k(embeddings, optimal_k)

            # Analyze by embedding model
            analyze_clusters_by_embedding_model(labels, metadata)

            # Analyze by LLM model
            analyze_clusters_by_llm_model(labels, metadata)

            # Save results
            results = {
                'optimal_k': optimal_k,
                'metrics': metrics,
                'cluster_labels': labels,
                'centroids': kmeans.cluster_centers_.tolist()
            }
            save_clustering_results(results, output_dir, f"{emb_model}_{norm_type}")

    print("\n" + "="*70)
    print("✓ K-MEANS CLUSTERING ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
