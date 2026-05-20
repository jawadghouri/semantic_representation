from transformers import AutoTokenizer

import huggingface_hub

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Access successful.")
print("CURRENT CODE TOKEN USERS:", huggingface_hub.whoami())