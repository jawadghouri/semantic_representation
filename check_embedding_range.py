import numpy as np

# Load a sample embedding file
embedding_file = "data/processed/embeddings/llama_bge.npy"
emb = np.load(embedding_file)

print("=" * 70)
print(f"FILE: {embedding_file}")
print("=" * 70)

print(f"\nShape: {emb.shape}")
print(f"Data type: {emb.dtype}")

# Get the first (and only) embedding vector
vector = emb[0]

print(f"\n--- RANGE ANALYSIS ---")
print(f"Min value: {vector.min():.6f}")
print(f"Max value: {vector.max():.6f}")
print(f"Mean value: {vector.mean():.6f}")
print(f"Std deviation: {vector.std():.6f}")

# Check L2 norm (for normalized embeddings, should be ~1.0)
l2_norm = np.linalg.norm(vector)
print(f"\nL2 Norm (length): {l2_norm:.6f}")
print(f"Is L2 normalized? {np.isclose(l2_norm, 1.0, atol=0.01)}")

# Display first 10 values
print(f"\n--- FIRST 10 EMBEDDING VALUES ---")
for i, val in enumerate(vector[:10]):
    print(f"  [{i}]: {val:8.6f}")

print(f"\n--- ANALYSIS ---")
if vector.min() >= 0 and vector.max() <= 1:
    print("✓ Embeddings are in range [0, 1] (likely unnormalized probabilities)")
elif vector.min() >= -1 and vector.max() <= 1:
    print("✓ Embeddings are in range [-1, 1]")
elif np.isclose(l2_norm, 1.0, atol=0.01):
    print("✓ Embeddings are L2 NORMALIZED (unit vectors)")
    print("  → Values are typically between -0.3 and 0.3")
else:
    print("? Embeddings are in a custom range")

print("=" * 70)

# Also show all files for comparison
print("\nCOMPARISON OF ALL EMBEDDINGS:\n")
import os
for filename in sorted(os.listdir("data/processed/embeddings/")):
    if filename.endswith('.npy') and not filename.startswith('prompt'):
        filepath = os.path.join("data/processed/embeddings/", filename)
        e = np.load(filepath)
        v = e[0]
        norm = np.linalg.norm(v)
        print(f"{filename:25} Min: {v.min():7.4f} | Max: {v.max():7.4f} | L2 Norm: {norm:.4f}")
