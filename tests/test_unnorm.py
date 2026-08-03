# from sentence_transformers import SentenceTransformer
# # from utils.io_utils import load_json, save_numpy


# # 1. Load the model
# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# # filepath = "data/raw_outputs/llama_outputs.json"

# # data = load_json(filepath)

# # text = data[0]["response"] if "response" in data[0] else data[0]["responses"][0]

# # print("Sample text:", text)

# # 2. Remove the Normalize layer
# # The mxbai-embed-large model typically stores its layers in: 
# # [Transformer, Pooling, Normalize]
# model._modules.pop('2')  # Remove the Normalize module

# # 3. Generate embeddings
# texts = ["Your first sentence", "Your second sentence"]
# embeddings = model.encode(texts, convert_to_numpy=True)

# # save_numpy(embeddings, f"tests/embeddings/test_unnorm.npy")

# print("Unnormalized embeddings shape:", embeddings.shape)

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize
from utils.io_utils import  save_numpy




# 1. Load the model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 2. Safely find and remove the Normalize layer
# Iterate through the model's modules to find the key for the Normalize layer
normalize_key = None
for key, module in model._modules.items():
    if isinstance(module, Normalize):
        normalize_key = key
        break

if normalize_key:
    # Remove the Normalize module
    model._modules.pop(normalize_key)
    print(f"Successfully removed the Normalize layer (key: '{normalize_key}').")
else:
    print("No Normalize layer found in this model.")

# 3. Generate embeddings
texts = ["Your first sentence", "Your second sentence"]
# We set normalize_embeddings=False just to be absolutely sure the encode method
# doesn't try to apply normalization independently of the model layers.
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
save_numpy(embeddings, f"tests/embeddings/test_unnorm.npy")

print("Unnormalized embeddings shape:", embeddings.shape)