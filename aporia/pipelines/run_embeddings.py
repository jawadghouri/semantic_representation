import gc
import numpy as np
from tqdm import tqdm

from embeddings.minilm_embedder import MiniLMEmbedder
from embeddings.bge_embedder import BGEEmbedder
from embeddings.e5_embedder import E5Embedder
from utils.io_utils import load_json, save_numpy

RAW_DIR = "data/aporia/raw_outputs"
EMB_DIR = "data/aporia/embeddings"
LLM_NAMES = ["llama", "mistral", "phi"]

embedder_blueprints = {
    "minilm": MiniLMEmbedder,
    "bge": BGEEmbedder,
    "e5": E5Embedder,
}


def run_pipeline():
    for llm in LLM_NAMES:
        data = load_json(f"{RAW_DIR}/{llm}_responses.json")

        for emb_name, embedder_class in embedder_blueprints.items():
            print(f"\nEmbedding: {llm} x {emb_name}")
            embedder = embedder_class()

            for item in tqdm(data, desc=f"{llm}/{emb_name}"):
                prompt_id = item["prompt_id"]
                responses = item["responses"]

                # Encode all N responses at once → shape (N, dim)
                embeddings = embedder.encode(
                    responses,
                    batch_size=15,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )

                out_path = f"{EMB_DIR}/{llm}_{emb_name}_{prompt_id}.npy"
                save_numpy(embeddings, out_path)

            del embedder
            gc.collect()

        print(f"Done: {llm}")

    print("\nEmbedding complete.")


if __name__ == "__main__":
    run_pipeline()
