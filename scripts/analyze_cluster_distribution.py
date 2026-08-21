import sys
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load embeddings and metadata."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    bge_unnorm = np.load(embedding_dir / "new_dataset_bge_unnormalized.npy")
    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return bge_unnorm, metadata

def analyze_cluster_llm_distribution():
    """Analyze which LLMs are in each cluster."""
    embeddings, metadata = load_embeddings_and_metadata()

    # K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    print("\n" + "="*80)
    print("CLUSTER LLM DISTRIBUTION ANALYSIS")
    print("="*80)

    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]

        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id} - Total: {len(cluster_responses)} responses")
        print(f"{'─'*80}")

        llm_counts = {}
        for response in cluster_responses:
            llm = response['model']
            llm_counts[llm] = llm_counts.get(llm, 0) + 1

        for llm in sorted(llm_counts.keys()):
            count = llm_counts[llm]
            pct = (count / len(cluster_responses)) * 100
            print(f"  {llm:15} - {count:2d} responses ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("OVERALL DATA DISTRIBUTION")
    print("="*80)

    overall_llm_counts = {}
    for m in metadata:
        llm = m['model']
        overall_llm_counts[llm] = overall_llm_counts.get(llm, 0) + 1

    for llm in sorted(overall_llm_counts.keys()):
        count = overall_llm_counts[llm]
        pct = (count / len(metadata)) * 100
        print(f"  {llm:15} - {count:2d} responses ({pct:5.1f}%)")

if __name__ == "__main__":
    analyze_cluster_llm_distribution()
