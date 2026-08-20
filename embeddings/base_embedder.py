from abc import ABC, abstractmethod
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class BaseEmbedder(ABC):
    def _load_model(self, model_name: str, **kwargs):
        # 1. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)

        # 2. Strip Normalize layers to get raw (unnormalized) embeddings
        keys_to_remove = [
            k for k, v in self.model._modules.items()
            if type(v).__name__ == "Normalize"
        ]
        for k in keys_to_remove:
            self.model._modules.pop(k)

        # 3. Set Device & Eval Mode
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False):
        """Unified raw inference loop for all subclasses."""
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        
        if show_progress_bar:
            iterator = tqdm(iterator, desc=f"Encoding with {self.__class__.__name__}")
            
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Apply subclass-specific pooling (CLS or Mean)
                embeddings = self._pool(outputs, inputs)
                
                # Enforce unnormalized embeddings
                # If you ever want to allow normalization, add standard L2 norm here.
                if normalize_embeddings:
                     embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0).numpy()

    @abstractmethod
    def _pool(self, outputs, inputs) -> torch.Tensor:
        """Subclasses must define how to extract the final embeddings from model outputs."""
        pass

    # --- Pooling Helpers ---

    def cls_pooling(self, outputs) -> torch.Tensor:
        """Extracts the [CLS] token representation (used by BGE)."""
        return outputs.last_hidden_state[:, 0]

    def mean_pooling(self, outputs, inputs) -> torch.Tensor:
        """Averages the hidden states, masking out padding tokens (used by E5, MiniLM)."""
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask


# from abc import ABC, abstractmethod

# from transformers import AutoTokenizer, AutoModel
# import torch


# class BaseEmbedder(ABC):

#     def _load_model(self, model_name: str, **kwargs):
#         """
#         Load raw Hugging Face transformer model and tokenizer.

#         Unlike SentenceTransformer, this does NOT perform:
#             - pooling
#             - normalization
#             - any post-processing

#         Returns:
#             tokenizer, model
#         """
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             **kwargs
#         )

#         model = AutoModel.from_pretrained(
#             model_name,
#             **kwargs
#         )

#         model.eval()

#         print(
#             f"Successfully loaded raw transformer: {model_name}"
#         )

#         return tokenizer, model

#     @abstractmethod
#     def encode(self, texts):
#         pass

# # from abc import ABC, abstractmethod

# # from sentence_transformers import SentenceTransformer


# # class BaseEmbedder(ABC):

# #     def _load_model(self, model_name: str, **kwargs) -> SentenceTransformer:
# #         model = SentenceTransformer(model_name, **kwargs)

# #         # Strip Normalize modules baked into the pipeline.
# #         # Uses class name string instead of isinstance — immune to import path
# #         # changes across sentence_transformers versions.
# #         # Uses pop() to mutate _modules in-place instead of reassigning,
# #         # which is required because nn.Sequential holds a reference to the
# #         # original OrderedDict object.
# #         keys_to_remove = [
# #             k for k, v in model._modules.items()
# #             if type(v).__name__ == "Normalize"
# #         ]
# #         for k in keys_to_remove:
# #             model._modules.pop(k)

# #         # Sanity check — warn if something still looks wrong
# #         remaining = [type(v).__name__ for v in model._modules.values()]
# #         assert "Normalize" not in remaining, (
# #             f"Failed to strip Normalize from pipeline. Remaining modules: {remaining}"
# #         )
# #         print("Successfully stripped Normalize from pipeline.")
# #         return model

# #     @abstractmethod
# #     def encode(self, texts):
# #         pass

# # # from abc import ABC, abstractmethod

# # # class BaseEmbedder(ABC):

# # #     @abstractmethod
# # #     def encode(self, text: str):
# # #         pass