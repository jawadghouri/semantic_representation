from embedding import get_embeddings
from analysis import compute_similarity

# Step 1: Your sample texts (simulate LLM outputs)
texts = [
    "A distributed system is a group of computers working together.",
    "Multiple machines collaborate to perform tasks.",
    "Pizza is a popular food."
]

# Step 2: Generate embeddings
embeddings = get_embeddings(texts)
# print("Embeddings:")
# print(embeddings)

# Step 3: Compute similarity
similarity_matrix = compute_similarity(embeddings)

print("Similarity Matrix:")
print(similarity_matrix)