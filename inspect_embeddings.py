import numpy as np
import os

# List all embedding files
embeddings_dir = "data/processed/embeddings/"

print("=" * 60)
print("RESPONSE EMBEDDINGS SUMMARY")
print("=" * 60)

for filename in sorted(os.listdir(embeddings_dir)):
    if filename.endswith('.npy') and not filename.startswith('prompt'):
        filepath = os.path.join(embeddings_dir, filename)
        emb = np.load(filepath)

        print(f"\n{filename}")
        print(f"  Shape: {emb.shape}")
        print(f"  Data type: {emb.dtype}")
        print(f"  Min: {emb.min():.4f}, Max: {emb.max():.4f}")
        print(f"  Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")

print("\n" + "=" * 60)
print("PROMPT EMBEDDINGS SUMMARY")
print("=" * 60)

for filename in sorted(os.listdir(embeddings_dir)):
    if filename.startswith('prompt') and filename.endswith('.npy'):
        filepath = os.path.join(embeddings_dir, filename)
        emb = np.load(filepath)

        print(f"\n{filename}")
        print(f"  Shape: {emb.shape}")
        print(f"  Data type: {emb.dtype}")
        print(f"  Min: {emb.min():.4f}, Max: {emb.max():.4f}")
        print(f"  Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
