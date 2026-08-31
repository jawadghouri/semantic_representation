from embeddings.base_embedder import BaseEmbedder

class MiniLMEmbedder(BaseEmbedder):
    def __init__(self):
        self._load_model("sentence-transformers/all-MiniLM-L6-v2")

    def _pool(self, outputs, inputs):
        # MiniLM models use mean pooling
        return self.mean_pooling(outputs, inputs)