import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_embeddings():
    """Analyze the generated embeddings."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    # Load embeddings
    bge_emb = np.load(embedding_dir / "new_dataset_bge.npy")
    e5_emb = np.load(embedding_dir / "new_dataset_e5.npy")
    minilm_emb = np.load(embedding_dir / "new_dataset_minilm.npy")

    # Load metadata
    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    print("\n" + "="*80)
    print("DETAILED EMBEDDING ANALYSIS - New Dataset")
    print("="*80)

    # 1. Size Analysis
    print("\n1. EMBEDDING DIMENSIONS")
    print("-" * 80)
    print(f"BGE Embeddings:    {bge_emb.shape}")
    print(f"E5 Embeddings:     {e5_emb.shape}")
    print(f"MiniLM Embeddings: {minilm_emb.shape}")
    print(f"\nTotal responses: {bge_emb.shape[0]}")
    print(f"Total embeddings generated: {bge_emb.shape[0] * 3}")

    # 2. Value Statistics
    print("\n2. VALUE STATISTICS")
    print("-" * 80)

    def print_stats(name, arr):
        print(f"\n{name}:")
        print(f"  Range:       [{arr.min():.6f}, {arr.max():.6f}]")
        print(f"  Mean:        {arr.mean():.6f}")
        print(f"  Std Dev:     {arr.std():.6f}")
        print(f"  Median:      {np.median(arr):.6f}")
        print(f"  L2 norms:    min={np.linalg.norm(arr[0]):.4f}, max={np.linalg.norm(arr).max():.4f}, mean={np.linalg.norm(arr, axis=1).mean():.4f}")

    print_stats("BGE", bge_emb)
    print_stats("E5", e5_emb)
    print_stats("MiniLM", minilm_emb)

    # 3. Similarity analysis
    print("\n3. COSINE SIMILARITY ANALYSIS")
    print("-" * 80)

    def compute_similarities(emb):
        """Compute pairwise cosine similarities."""
        # Normalize for cosine similarity
        normalized = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(normalized, normalized.T)

        # Get upper triangle (avoid duplicates and diagonal)
        upper_indices = np.triu_indices(len(similarities), k=1)
        values = similarities[upper_indices]

        return values

    bge_sim = compute_similarities(bge_emb)
    e5_sim = compute_similarities(e5_emb)
    minilm_sim = compute_similarities(minilm_emb)

    print(f"BGE Cosine Similarities:")
    print(f"  Mean: {bge_sim.mean():.6f}, Std: {bge_sim.std():.6f}")
    print(f"  Range: [{bge_sim.min():.6f}, {bge_sim.max():.6f}]")

    print(f"\nE5 Cosine Similarities:")
    print(f"  Mean: {e5_sim.mean():.6f}, Std: {e5_sim.std():.6f}")
    print(f"  Range: [{e5_sim.min():.6f}, {e5_sim.max():.6f}]")

    print(f"\nMiniLM Cosine Similarities:")
    print(f"  Mean: {minilm_sim.mean():.6f}, Std: {minilm_sim.std():.6f}")
    print(f"  Range: [{minilm_sim.min():.6f}, {minilm_sim.max():.6f}]")

    # 4. Category Analysis
    print("\n4. CATEGORY BREAKDOWN")
    print("-" * 80)
    categories = {}
    for i, m in enumerate(metadata):
        cat = m['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    print(f"Total categories: {len(categories)}")
    for cat in sorted(categories.keys()):
        indices = categories[cat]
        print(f"  {cat:20} - {len(indices)} responses")

    # 5. Model Breakdown
    print("\n5. LLM MODEL BREAKDOWN")
    print("-" * 80)
    models = {}
    for i, m in enumerate(metadata):
        mod = m['model']
        if mod not in models:
            models[mod] = []
        models[mod].append(i)

    print(f"Total LLM models: {len(models)}")
    for mod in sorted(models.keys()):
        indices = models[mod]
        print(f"  {mod:15} - {len(indices)} responses")

    # 6. Memory Usage
    print("\n6. FILE SIZE ANALYSIS")
    print("-" * 80)
    bge_size = (bge_emb.nbytes / 1024)
    e5_size = (e5_emb.nbytes / 1024)
    minilm_size = (minilm_emb.nbytes / 1024)
    total_size = (bge_size + e5_size + minilm_size)

    print(f"BGE:    {bge_size:8.2f} KB")
    print(f"E5:     {e5_size:8.2f} KB")
    print(f"MiniLM: {minilm_size:8.2f} KB")
    print(f"Total:  {total_size:8.2f} KB (~{total_size/1024:.2f} MB)")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    analyze_embeddings()
