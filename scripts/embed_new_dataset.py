import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.bge_embedder import BGEEmbedder
from embeddings.e5_embedder import E5Embedder
from embeddings.minilm_embedder import MiniLMEmbedder

def load_new_dataset(json_path):
    """Load new_dataset.json and extract all responses."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_responses = []
    metadata = []

    for item in data:
        item_id = item['id']
        prompt = item['prompt']
        category = item['category']

        for model_name, response_text in item['responses'].items():
            all_responses.append(response_text)
            metadata.append({
                'id': item_id,
                'prompt': prompt,
                'category': category,
                'model': model_name
            })

    return all_responses, metadata

def embed_with_model(embedder, texts, model_name):
    """Generate embeddings using specified embedder."""
    print(f"\n{'='*60}")
    print(f"Embedding with {model_name}")
    print(f"{'='*60}")
    print(f"Total texts to embed: {len(texts)}")

    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Min value: {embeddings.min():.6f}")
    print(f"Max value: {embeddings.max():.6f}")
    print(f"Mean value: {embeddings.mean():.6f}")
    print(f"Std value: {embeddings.std():.6f}")

    return embeddings

def main():
    # Paths - relative to project root
    project_root = Path(__file__).parent.parent
    json_path = project_root / "progress_meeting/constraint_prompts/new_dataset.json"
    output_dir = project_root / "new/embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n" + "="*60)
    print("LOADING NEW DATASET")
    print("="*60)
    texts, metadata = load_new_dataset(json_path)
    print(f"Total responses extracted: {len(texts)}")
    print(f"Unique categories: {len(set(m['category'] for m in metadata))}")
    print(f"Unique models: {set(m['model'] for m in metadata)}")
    print(f"Unique prompts: {len(set(m['prompt'] for m in metadata))}")

    # Initialize embedders
    print("\n" + "="*60)
    print("INITIALIZING EMBEDDERS")
    print("="*60)
    embedders = {
        'BGE': BGEEmbedder(),
        'E5': E5Embedder(),
        'MiniLM': MiniLMEmbedder()
    }

    # Generate embeddings with each model
    results = {}
    for model_name, embedder in embedders.items():
        embeddings = embed_with_model(embedder, texts, model_name)
        results[model_name] = embeddings

        # Save to numpy file
        output_file = output_dir / f"new_dataset_{model_name.lower()}.npy"
        np.save(output_file, embeddings)
        print(f"✓ Saved to: {output_file}")

    # Save metadata
    metadata_file = output_dir / "new_dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_file}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal responses embedded: {len(texts)}")
    print(f"Breakdown by model:")
    for model_name, embeddings in results.items():
        print(f"  {model_name:8} - Shape: {embeddings.shape}, Dim: {embeddings.shape[1]}")

    print(f"\nBreakdown by category:")
    category_counts = {}
    for m in metadata:
        cat = m['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:20} - {count} responses")

    print(f"\nBreakdown by LLM model:")
    model_counts = {}
    for m in metadata:
        mod = m['model']
        model_counts[mod] = model_counts.get(mod, 0) + 1
    for mod, count in sorted(model_counts.items()):
        print(f"  {mod:15} - {count} responses")

    print("\n" + "="*60)
    print("✓ EMBEDDING GENERATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
