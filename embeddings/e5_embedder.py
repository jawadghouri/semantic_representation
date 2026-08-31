from embeddings.base_embedder import BaseEmbedder

class E5Embedder(BaseEmbedder):
    def __init__(self):
        self._load_model("intfloat/e5-large-v2")

    def _pool(self, outputs, inputs):
        # E5 models use mean pooling
        return self.mean_pooling(outputs, inputs)

    def encode(self, texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False):
        # Prepend the required prefix for E5
        if isinstance(texts, str):
            texts = [texts]
            
        prefixed_texts = ["passage: " + text for text in texts]
        
        # Hand off to the BaseEmbedder's encode logic
        return super().encode(
            prefixed_texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            normalize_embeddings=normalize_embeddings
        )