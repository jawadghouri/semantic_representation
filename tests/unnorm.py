from transformers import AutoTokenizer, AutoModel
import torch

from utils.io_utils import save_numpy

# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example text
text = "This is an example sentence."

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Raw token embeddings
token_embeddings = outputs.last_hidden_state
save_numpy(token_embeddings, f"tests/embeddings/unnorm.npy")


print("Shape:", token_embeddings.shape)
print()
print("First token embedding:")
print(token_embeddings[0, 0])