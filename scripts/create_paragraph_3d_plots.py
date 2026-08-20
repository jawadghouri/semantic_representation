import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_all_data():
    """Load embeddings, metadata, and cluster labels."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    bge_emb = np.load(embedding_dir / "new_dataset_bge.npy")
    e5_emb = np.load(embedding_dir / "new_dataset_e5.npy")
    minilm_emb = np.load(embedding_dir / "new_dataset_minilm.npy")

    bge_labels = np.load(embedding_dir / "new_dataset_bge_kmeans_labels.npy")
    e5_labels = np.load(embedding_dir / "new_dataset_e5_kmeans_labels.npy")
    minilm_labels = np.load(embedding_dir / "new_dataset_minilm_kmeans_labels.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'bge': bge_emb,
        'e5': e5_emb,
        'minilm': minilm_emb
    }, {
        'bge': bge_labels,
        'e5': e5_labels,
        'minilm': minilm_labels
    }, metadata

def create_paragraph_centric_plots(embeddings_dict, cluster_labels, metadata, project_root):
    """Create plots for each prompt showing how 3 models embed the responses."""
    output_dir = project_root / "new/visualizations/prompt_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique prompts
    prompts = sorted(set(m['prompt'] for m in metadata))

    print("\n" + "="*80)
    print("CREATING PROMPT-CENTRIC COMPARISON PLOTS")
    print("="*80)

    for prompt_idx, prompt_text in enumerate(prompts, 1):
        # Get indices for this prompt
        prompt_indices = [i for i, m in enumerate(metadata) if m['prompt'] == prompt_text]
        prompt_id = metadata[prompt_indices[0]]['id']
        category = metadata[prompt_indices[0]]['category']

        print(f"\n[{prompt_idx:2d}] {prompt_id} - {category}")
        print(f"     Prompt: {prompt_text[:60]}...")

        # Get LLM responses for this prompt
        llm_responses = {}
        for idx in prompt_indices:
            llm_name = metadata[idx]['model']
            llm_responses[llm_name] = metadata[idx]

        # Apply PCA to each embedding model (using all data for consistency)
        pca_projections = {}
        for model_name, emb in embeddings_dict.items():
            pca = PCA(n_components=3)
            proj = pca.fit_transform(emb)
            pca_projections[model_name] = {
                'pca': pca,
                'projection': proj,
                'indices': prompt_indices
            }

        # Create 3D comparison plot
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f'3D Response Embedding Comparison — {prompt_id} ({category})\n"{prompt_text}"',
                    fontsize=14, fontweight='bold', y=0.98)

        # Add overall variance info
        var_info = f"Total Variance Explained: BGE={pca_projections['bge']['pca'].explained_variance_ratio_.sum()*100:.1f}% | " \
                  f"E5={pca_projections['e5']['pca'].explained_variance_ratio_.sum()*100:.1f}% | " \
                  f"MiniLM={pca_projections['minilm']['pca'].explained_variance_ratio_.sum()*100:.1f}%"
        fig.text(0.5, 0.94, var_info, ha='center', fontsize=11, style='italic')

        # 3D plots for each model
        colors_map = {
            'chatgpt': '#FF6B6B',
            'claude': '#4ECDC4',
            'deepseek': '#45B7D1',
            'gemini': '#FFA07A',
            'grok': '#98D8C8'
        }

        markers_map = {
            'chatgpt': 'o',
            'claude': 's',
            'deepseek': '^',
            'gemini': 'D',
            'grok': 'v'
        }

        for plot_idx, (model_name, pca_data) in enumerate(pca_projections.items(), 1):
            ax = fig.add_subplot(2, 3, plot_idx, projection='3d')

            proj = pca_data['projection']
            indices = pca_data['indices']
            pca_model = pca_data['pca']

            # Plot points for this prompt only, colored by LLM
            for idx in indices:
                llm_name = metadata[idx]['model']
                color = colors_map.get(llm_name, '#888888')
                marker = markers_map.get(llm_name, 'o')

                ax.scatter(proj[idx, 0], proj[idx, 1], proj[idx, 2],
                          c=color, marker=marker, s=200, alpha=0.8,
                          edgecolors='black', linewidth=1.5, label=llm_name)

            # Formatting
            ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
            ax.set_title(f'{model_name.upper()}', fontsize=12, fontweight='bold')
            ax.view_init(elev=20, azim=45)

            # Add grid
            ax.grid(True, alpha=0.3)

        # Add legend in the 6th subplot
        ax_legend = fig.add_subplot(2, 3, 6)
        ax_legend.axis('off')

        legend_elements = [Patch(facecolor=colors_map[llm], edgecolor='black', label=llm)
                          for llm in sorted(colors_map.keys())]
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=12, title='LLM Models')

        # Add cluster information
        cluster_info = f"\nCluster Distribution:\n"
        for model_name in ['bge', 'e5', 'minilm']:
            labels = cluster_labels[model_name]
            prompt_clusters = [labels[i] for i in indices]
            cluster_info += f"{model_name.upper()}: {prompt_clusters}\n"

        ax_legend.text(0.5, 0.3, cluster_info, transform=ax_legend.transAxes,
                      fontsize=10, verticalalignment='top', horizontalalignment='center',
                      family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Save plot
        plot_file = output_dir / f"{prompt_id}_3d_response_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {plot_file.name}")
        plt.close()

def create_all_responses_comparison(embeddings_dict, cluster_labels, metadata, project_root):
    """Create a comprehensive comparison of all 80 responses colored by LLM."""
    output_dir = project_root / "new/visualizations/prompt_comparison"

    print("\n" + "="*80)
    print("CREATING ALL RESPONSES COMPARISON PLOT")
    print("="*80)

    # Apply PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=3)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # Create figure
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('3D PCA Comparison — All 80 LLM Responses (Colored by LLM Model)',
                fontsize=14, fontweight='bold')

    colors_map = {
        'chatgpt': '#FF6B6B',
        'claude': '#4ECDC4',
        'deepseek': '#45B7D1',
        'gemini': '#FFA07A',
        'grok': '#98D8C8'
    }

    for plot_idx, (model_name, pca_data) in enumerate(pca_results.items(), 1):
        ax = fig.add_subplot(1, 3, plot_idx, projection='3d')

        proj = pca_data['projection']
        pca_model = pca_data['pca']

        # Plot each LLM with different color
        for llm_name in sorted(set(m['model'] for m in metadata)):
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            color = colors_map[llm_name]

            ax.scatter(proj[llm_mask, 0], proj[llm_mask, 1], proj[llm_mask, 2],
                      c=color, label=llm_name, s=100, alpha=0.7,
                      edgecolors='black', linewidth=0.5)

        # Formatting
        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
        ax.set_title(f'{model_name.upper()} — Total Variance: {pca_model.explained_variance_ratio_.sum()*100:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()

    plot_file = output_dir / "all_responses_3d_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_file.name}")
    plt.close()

def create_cluster_overlay_plots(embeddings_dict, cluster_labels, metadata, project_root):
    """Create plots showing K-Means clusters with prompt IDs labeled."""
    output_dir = project_root / "new/visualizations/prompt_comparison"

    print("\n" + "="*80)
    print("CREATING CLUSTER OVERLAY PLOTS WITH PROMPT IDS")
    print("="*80)

    # Apply PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=3)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # Create figure
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('3D PCA with K-Means Clusters and Prompt Labels',
                fontsize=14, fontweight='bold')

    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    for plot_idx, (model_name, pca_data) in enumerate(pca_results.items(), 1):
        ax = fig.add_subplot(1, 3, plot_idx, projection='3d')

        proj = pca_data['projection']
        pca_model = pca_data['pca']
        labels = cluster_labels[model_name]

        # Plot clusters
        for cluster_id in range(3):
            mask = labels == cluster_id
            ax.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2],
                      c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}',
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add prompt ID labels
        for i, m in enumerate(metadata):
            prompt_id = m['id']
            ax.text(proj[i, 0], proj[i, 1], proj[i, 2], prompt_id,
                   fontsize=7, ha='center', va='center', weight='bold')

        # Formatting
        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
        ax.set_title(f'{model_name.upper()}', fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()

    plot_file = output_dir / "clusters_with_prompt_labels.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_file.name}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, cluster_labels, metadata = load_all_data()

    # Create prompt-centric plots
    create_paragraph_centric_plots(embeddings_dict, cluster_labels, metadata, project_root)

    # Create all responses comparison
    create_all_responses_comparison(embeddings_dict, cluster_labels, metadata, project_root)

    # Create cluster overlay plots
    create_cluster_overlay_plots(embeddings_dict, cluster_labels, metadata, project_root)

    print("\n" + "="*80)
    print("✓ ALL COMPARISON PLOTS CREATED")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
