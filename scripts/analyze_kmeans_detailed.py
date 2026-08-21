import sys
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load both normalized and unnormalized embeddings."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    bge_unnorm = np.load(embedding_dir / "new_dataset_bge_unnormalized.npy")
    e5_unnorm = np.load(embedding_dir / "new_dataset_e5_unnormalized.npy")
    minilm_unnorm = np.load(embedding_dir / "new_dataset_minilm_unnormalized.npy")

    bge_norm = np.load(embedding_dir / "new_dataset_bge_normalized.npy")
    e5_norm = np.load(embedding_dir / "new_dataset_e5_normalized.npy")
    minilm_norm = np.load(embedding_dir / "new_dataset_minilm_normalized.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'unnormalized': {'bge': bge_unnorm, 'e5': e5_unnorm, 'minilm': minilm_unnorm},
        'normalized': {'bge': bge_norm, 'e5': e5_norm, 'minilm': minilm_norm}
    }, metadata

def analyze_clustering_by_llm_model(labels, metadata, k):
    """Analyze cluster composition by LLM model."""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS BY LLM MODEL")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))

    # Create summary table
    data = []
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]
        cluster_size = len(cluster_responses)

        row = {'Cluster': cluster_id, 'Size': cluster_size}

        for llm in llm_models:
            count = sum(1 for r in cluster_responses if r['model'] == llm)
            pct = (count / cluster_size * 100) if cluster_size > 0 else 0
            row[llm] = f"{count} ({pct:.1f}%)"

        data.append(row)

    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)

    return df

def analyze_clustering_by_embedding_model(labels, metadata, embedding_model_name):
    """Analyze cluster composition by response count and distribution."""
    print("\n" + "="*80)
    print(f"CLUSTER SIZE ANALYSIS ({embedding_model_name.upper()})")
    print("="*80)

    k = len(set(labels))

    data = []
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]
        cluster_size = len(cluster_responses)

        # Count categories
        categories = {}
        for r in cluster_responses:
            cat = r['category']
            categories[cat] = categories.get(cat, 0) + 1

        # Find dominant category
        dominant_cat = max(categories, key=categories.get) if categories else "N/A"
        dominant_count = categories.get(dominant_cat, 0)

        row = {
            'Cluster': cluster_id,
            'Size': cluster_size,
            'Pct': f"{(cluster_size/80)*100:.1f}%",
            'Dominant Category': dominant_cat,
            'Count': dominant_count
        }
        data.append(row)

    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)

    return df

def analyze_cluster_purity_by_llm(labels, metadata, k):
    """Calculate cluster purity by LLM model."""
    print("\n" + "="*80)
    print("CLUSTER PURITY ANALYSIS BY LLM MODEL")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))

    print(f"\nK = {k} clusters, 80 total responses")
    print(f"LLM Models: {', '.join(llm_models)}")
    print("\nPurity Score = (max LLM count in cluster) / (cluster size) × 100")
    print("-" * 70)

    data = []
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]
        cluster_size = len(cluster_responses)

        llm_counts = {}
        for llm in llm_models:
            count = sum(1 for r in cluster_responses if r['model'] == llm)
            llm_counts[llm] = count

        max_llm = max(llm_counts, key=llm_counts.get)
        max_count = llm_counts[max_llm]
        purity = (max_count / cluster_size * 100) if cluster_size > 0 else 0

        row = {
            'Cluster': cluster_id,
            'Size': cluster_size,
            'Dominant LLM': max_llm,
            'Dominant Count': max_count,
            'Purity %': f"{purity:.1f}%"
        }
        data.append(row)

    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))

    # Overall purity
    overall_purity = np.mean([float(row['Purity %'].rstrip('%')) for row in data])
    print(f"\nOverall Cluster Purity: {overall_purity:.2f}%")
    print("="*80)

    return df

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "new/clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("DETAILED K-MEANS CLUSTERING ANALYSIS")
    print("="*80)

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    # Analyze each embedding model
    for model_name in embedding_models:
        print(f"\n\n{'#'*80}")
        print(f"EMBEDDING MODEL: {model_name.upper()}")
        print(f"{'#'*80}")

        for norm_type in ['unnormalized', 'normalized']:
            print(f"\n{'='*80}")
            print(f"{norm_type.upper()} EMBEDDINGS")
            print(f"{'='*80}")

            embeddings = embeddings_dict[norm_type][model_name]

            # Perform clustering with K=5
            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Analysis by LLM model
            df_llm = analyze_clustering_by_llm_model(labels, metadata, k_value)

            # Analysis by size
            df_size = analyze_clustering_by_embedding_model(labels, metadata, model_name)

            # Purity analysis
            df_purity = analyze_cluster_purity_by_llm(labels, metadata, k_value)

            # Save to CSV
            csv_file = output_dir / f"cluster_analysis_{model_name}_{norm_type}_k{k_value}.csv"
            df_llm.to_csv(csv_file, index=False)
            print(f"\n✓ Saved analysis to: {csv_file}")

    print("\n" + "="*80)
    print("✓ DETAILED CLUSTERING ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
