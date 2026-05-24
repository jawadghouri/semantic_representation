from sentence_transformers import SentenceTransformer

from embeddings.base_embedder import BaseEmbedder


class E5Embedder(BaseEmbedder):

    def __init__(self):

        self.model = SentenceTransformer(
            "intfloat/e5-small-v2"
        )

    def encode(self, texts):

        texts = [
            "passage: " + text
            for text in texts
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        return embeddings