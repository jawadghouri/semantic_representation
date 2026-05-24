from sentence_transformers import SentenceTransformer

from embeddings.base_embedder import BaseEmbedder


class BGEEmbedder(BaseEmbedder):

    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    def encode(self, texts):

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False
        )

        return embeddings