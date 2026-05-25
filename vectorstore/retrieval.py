import numpy as np

from vectorstore.faiss_manager import (
    FAISSManager
)


class Retriever:

    def __init__(
        self,
        index_path
    ):

        self.manager = FAISSManager(
            dimension=1
        )

        self.manager.load(
            index_path
        )

    def retrieve(
        self,
        query_embedding,
        top_k=5
    ):

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        distances, indices = (
            self.manager.search(
                query_embedding,
                top_k
            )
        )

        return (
            distances[0],
            indices[0]
        )