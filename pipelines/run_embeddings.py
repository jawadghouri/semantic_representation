from embeddings.minilm_embedder import MiniLMEmbedder
from embeddings.bge_embedder import BGEEmbedder
from embeddings.e5_embedder import E5Embedder

from utils.io_utils import load_json, save_numpy

from tqdm import tqdm


# ---------------------------------------------------
# LOAD RAW OUTPUTS
# ---------------------------------------------------

model_files = {
    "llama": "data/raw_outputs/llama_outputs.json",
    "mistral": "data/raw_outputs/mistral_outputs.json",
    "phi": "data/raw_outputs/phi_outputs.json"
}


# ---------------------------------------------------
# LOAD EMBEDDERS
# ---------------------------------------------------

embedders = {
    "minilm": MiniLMEmbedder(),
    "bge": BGEEmbedder(),
    "e5": E5Embedder()
}


# ---------------------------------------------------
# PROCESS EACH LLM OUTPUT
# ---------------------------------------------------

for llm_name, filepath in model_files.items():

    print(f"\nProcessing LLM outputs: {llm_name}")

    data = load_json(filepath)

    texts = []

    for item in data:

        # single response version
        if "response" in item:
            texts.append(item["response"])

        # multi-response version
        elif "responses" in item:

            for response in item["responses"]:
                texts.append(response)

    print(f"Total texts: {len(texts)}")


    # ---------------------------------------------------
    # RUN EACH EMBEDDING MODEL
    # ---------------------------------------------------

    for embed_name, embedder in embedders.items():

        print(f"\nEmbedding with: {embed_name}")

        embeddings = embedder.encode(texts)

        output_path = (
            f"data/processed/embeddings/"
            f"{llm_name}_{embed_name}.npy"
        )

        save_numpy(embeddings, output_path)

        print(f"Saved: {output_path}")

print("\nEmbedding pipeline complete.")