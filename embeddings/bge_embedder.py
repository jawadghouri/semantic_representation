from embeddings.base_embedder import BaseEmbedder

class BGEEmbedder(BaseEmbedder):
    def __init__(self):
        self._load_model(
            "BAAI/bge-base-en-v1.5",
            use_safetensors=True
        )

    def _pool(self, outputs, inputs):
        # BGE models use the [CLS] token
        return self.cls_pooling(outputs)