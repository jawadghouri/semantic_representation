import numpy as np

from embeddings.bge_embedder import BGEEmbedder

from vectorstore.retrieval import Retriever


embedder = BGEEmbedder()

retriever = Retriever(
    "data/processed/faiss/llama_bge.index"
)

# Unrelated query to test retrieval
# query = "How deep-sea coral reefs survive in extreme underwater pressure?"

# Related query to test retrieval
query = "The inner workings of an internal combustion engine and its pistons."

query_vector = embedder.encode(
    [query]
)


scores, indices = retriever.retrieve(
    query_vector[0],
    top_k=1
)


print(scores)
print(indices)