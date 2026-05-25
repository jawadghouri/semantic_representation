import faiss
import numpy as np


class FAISSManager:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(
            dimension
        )

    def add_embeddings(self, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        self.index.add(
            embeddings
        )

    def save(self, path):

        faiss.write_index(
            self.index,
            path
        )

    def load(self, path):

        self.index = faiss.read_index(
            path
        )

    def search(
        self,
        query_vector,
        top_k=5
    ):

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_vector,
            top_k
        )

        return distances, indices