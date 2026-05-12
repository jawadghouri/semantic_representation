from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(embeddings):
    return cosine_similarity(embeddings)