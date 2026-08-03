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


# from embeddings.base_embedder import BaseEmbedder


# class BGEEmbedder(BaseEmbedder):

#     def __init__(self):
#         self.model = self._load_model(
#             "BAAI/bge-base-en-v1.5",
#             model_kwargs={"use_safetensors": True}
#         )

#     def encode(self, texts):
#         embeddings = self.model.encode(
#             texts,
#             batch_size=32,
#             show_progress_bar=True,
#             normalize_embeddings=False
#         )

#         return embeddings

# # from sentence_transformers import SentenceTransformer

# # from embeddings.base_embedder import BaseEmbedder


# # class BGEEmbedder(BaseEmbedder):

# #     def __init__(self):

# #         # self.model = SentenceTransformer(
# #         #     "BAAI/bge-base-en-v1.5"
# #         # )
# #         self.model = SentenceTransformer(
# #             "BAAI/bge-m3",
# #             model_kwargs={"use_safetensors": True}
# #         )

# #     def encode(self, texts):

# #         embeddings = self.model.encode(
# #             texts,
# #             batch_size=32,
# #             show_progress_bar=True,
# #             normalize_embeddings=False
# #         )

# #         return embeddings