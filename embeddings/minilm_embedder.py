from sentence_transformers import SentenceTransformer

from embeddings.base_embedder import BaseEmbedder


class MiniLMEmbedder(BaseEmbedder):

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, texts):

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        return embeddings