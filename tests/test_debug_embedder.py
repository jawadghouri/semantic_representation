"""
Run this script standalone to diagnose why normalization persists.
Usage: python debug_embedder.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


def print_pipeline(label, model):
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"  _modules id     : {id(model._modules)}")
    print(f"  _modules keys   : {list(model._modules.keys())}")
    for k, v in model._modules.items():
        print(f"    [{k}] {type(v).__name__} (id={id(v)})")
    print(f"  named_children  : {[(k, type(v).__name__) for k, v in model.named_children()]}")


def check_norm(embeddings, label):
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  [{label}]")
    print(f"    norms          : {norms}")
    print(f"    mean norm      : {norms.mean():.6f}")
    print(f"    is normalized  : {np.allclose(norms, 1.0, atol=1e-3)}")


texts = ["This is a test sentence.", "Another sentence for embedding."]

# ── Step 1: Load and inspect raw pipeline ─────────────────────────────────────
model = SentenceTransformer(MODEL_NAME, model_kwargs={"use_safetensors": True})
print_pipeline("AFTER LOAD (before any strip)", model)

emb_raw = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
check_norm(emb_raw, "encode before strip")

# ── Step 2: Strip by class name, in-place ─────────────────────────────────────
print("\n--- Stripping Normalize modules ---")
keys_to_remove = [k for k, v in model._modules.items() if type(v).__name__ == "Normalize"]
print(f"  Keys found to remove : {keys_to_remove}")

for k in keys_to_remove:
    popped = model._modules.pop(k)
    print(f"  Popped [{k}] : {type(popped).__name__}")

print_pipeline("AFTER STRIP", model)

emb_stripped = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
check_norm(emb_stripped, "encode after strip")

# ── Step 3: Check if Normalize is hiding inside a child module ─────────────────
print("\n--- Deep scan: checking inside each child module ---")
for k, v in model._modules.items():
    for ck, cv in v._modules.items():
        print(f"  [{k}] -> [{ck}] : {type(cv).__name__}")
        if type(cv).__name__ == "Normalize":
            print(f"    *** FOUND NESTED Normalize at [{k}][{ck}] — popping it ***")
            v._modules.pop(ck)
            break

print_pipeline("AFTER DEEP STRIP", model)

emb_deep = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
check_norm(emb_deep, "encode after deep strip")

# ── Step 4: Nuclear option — monkey-patch normalize out entirely ───────────────
print("\n--- Nuclear option: patching F.normalize to be a no-op ---")
import torch.nn.functional as F

_original_normalize = F.normalize

def _noop_normalize(input, *args, **kwargs):
    return input  # return as-is

F.normalize = _noop_normalize

emb_patched = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
check_norm(emb_patched, "encode with F.normalize patched to no-op")

F.normalize = _original_normalize  # restore

# ── Step 5: Print sentence_transformers version ────────────────────────────────
import sentence_transformers
print(f"\n  sentence_transformers version : {sentence_transformers.__version__}")
import torch
print(f"  torch version                 : {torch.__version__}")