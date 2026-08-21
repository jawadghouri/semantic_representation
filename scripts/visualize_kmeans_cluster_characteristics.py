import sys
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns

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

def get_cluster_characteristics(embeddings, labels, metadata, k):
    """Calculate cluster characteristics for radar plot."""
    llm_models = sorted(set(m['model'] for m in metadata))
    categories = llm_models + ['Size', 'Density']

    characteristics = []

    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_responses = [m for i, m in enumerate(metadata) if cluster_mask[i]]
        cluster_emb = embeddings[cluster_mask]

        values = []

        # LLM model distribution (normalized to 0-100)
        for llm in llm_models:
            count = sum(1 for r in cluster_responses if r['model'] == llm)
            pct = (count / len(cluster_responses)) * 100 if len(cluster_responses) > 0 else 0
            values.append(pct)

        # Cluster size (normalize to 0-100 based on 80 total responses)
        size = len(cluster_responses)
        size_norm = (size / 80) * 100
        values.append(size_norm)

        # Cluster density (variance in first 2 PCs normalized)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(cluster_emb)
        density = np.sqrt(np.var(proj[:, 0]) + np.var(proj[:, 1]))
        density_norm = min(100, (density / 10) * 100)  # Normalize assuming max density ~10
        values.append(density_norm)

        characteristics.append(values)

    return np.array(characteristics), categories

def plot_radar_chart(ax, values, categories, cluster_id, color, title):
    """Plot a single radar chart."""
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    values = list(values) + [values[0]]  # Complete the circle
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=f'Cluster {cluster_id}')
    ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=20)
    ax.grid(True)
    ax.set_yticks([20, 40, 60, 80, 100])

def visualize_cluster_characteristics_radar(embeddings_dict, metadata, project_root):
    """Create radar charts showing cluster characteristics."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING CLUSTER CHARACTERISTICS RADAR PLOTS")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    for model_name in embedding_models:
        print(f"\nProcessing {model_name.upper()}...")

        # UNNORMALIZED
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        kmeans_unnorm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_unnorm = kmeans_unnorm.fit_predict(emb_unnorm)

        chars_unnorm, categories = get_cluster_characteristics(emb_unnorm, labels_unnorm, metadata, k_value)

        # Create figure for unnormalized
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'{model_name.upper()} - Cluster Characteristics (UNNORMALIZED, K={k_value})',
                    fontsize=14, fontweight='bold')

        colors_unnorm = plt.cm.tab10(np.linspace(0, 1, k_value))
        for cluster_id in range(k_value):
            ax = fig.add_subplot(2, 3, cluster_id + 1, projection='polar')
            plot_radar_chart(ax, chars_unnorm[cluster_id], categories, cluster_id,
                           colors_unnorm[cluster_id], f'Cluster {cluster_id} (n={np.sum(labels_unnorm == cluster_id)})')

        plt.tight_layout()
        viz_file = output_dir / f"kmeans_radar_{model_name}_unnormalized_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name} (unnormalized)")
        plt.close()

        # NORMALIZED
        emb_norm = embeddings_dict['normalized'][model_name]
        kmeans_norm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_norm = kmeans_norm.fit_predict(emb_norm)

        chars_norm, _ = get_cluster_characteristics(emb_norm, labels_norm, metadata, k_value)

        # Create figure for normalized
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'{model_name.upper()} - Cluster Characteristics (NORMALIZED, K={k_value})',
                    fontsize=14, fontweight='bold')

        colors_norm = plt.cm.tab10(np.linspace(0, 1, k_value))
        for cluster_id in range(k_value):
            ax = fig.add_subplot(2, 3, cluster_id + 1, projection='polar')
            plot_radar_chart(ax, chars_norm[cluster_id], categories, cluster_id,
                           colors_norm[cluster_id], f'Cluster {cluster_id} (n={np.sum(labels_norm == cluster_id)})')

        plt.tight_layout()
        viz_file = output_dir / f"kmeans_radar_{model_name}_normalized_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name} (normalized)")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    visualize_cluster_characteristics_radar(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ CLUSTER CHARACTERISTICS RADAR PLOTS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
