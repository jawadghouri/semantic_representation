##
# This generates embedding files for each response in the constraint prompts.
# It handles both normalized and unnormalized embeddings,
# saving the resulting embeddings to specified directories.
##

from http.client import responses

from embeddings.minilm_embedder import MiniLMEmbedder
from embeddings.bge_embedder import BGEEmbedder
from embeddings.e5_embedder import E5Embedder

from utils.io_utils import load_json, save_numpy

from utils.norm_utils import check_embeddings_normalization

# ---------------------------------------------------
# LOAD RAW OUTPUTS
# ---------------------------------------------------

model_files = {
    "C1": "progress_meeting/constraint_prompts/C1.json",
    "C2": "progress_meeting/constraint_prompts/C2.json",
    "C3": "progress_meeting/constraint_prompts/C3.json"
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

for constraint, filepath in model_files.items():

    print(f"\nProcessing Constraint: {constraint}")

    data = load_json(filepath)

    texts = []
    ids = []  # We will track the IDs parallel to the texts

    for item in data:
        item_id = item.get("id", "unknown_id")

        # single response version
        if "response" in item:
            texts.append(item["response"])
            ids.append(item_id)

        # multi-response version
        elif "responses" in item:
            for idx, response in enumerate(item["responses"]):
                texts.append(response)
                # Append an index to the ID so files don't overwrite each other
                ids.append(f"{item_id}_{idx}")

    print(f"Total texts: {len(texts)}")


    # ---------------------------------------------------
    # RUN EACH EMBEDDING MODEL
    # ---------------------------------------------------

    for embed_name, embedder in embedders.items():

        print(f"\nEmbedding with: {embed_name}")

        # Batch encode all texts for this JSON at once (much faster)
        # Returns a numpy matrix of shape (num_texts, embedding_dim)
        embeddings = embedder.encode(texts)

        # Iterate through the embedded matrix and save them individually
        for item_id, single_embedding in zip(ids, embeddings):
            
            output_path = f"progress_meeting/embeddings/{item_id}_{embed_name}.npy"
            
            # Save the individual 1D array (shape: embedding_dim,)
            save_numpy(single_embedding, output_path)

        print(f"Saved {len(ids)} individual files for {embed_name} (Constraint: {constraint})")

    # ---------------------------------------------------
    # RUN EACH EMBEDDING MODEL (NORMALIZED)
    # ---------------------------------------------------

    for embed_name, embedder in embedders.items():
        print(f"\nEmbedding with: {embed_name} (NORMALIZED)")

        embeddings = embedder.encode(
            texts, 
            normalize_embeddings=True 
        )

        for item_id, single_embedding in zip(ids, embeddings):
            # Save to a new folder to avoid overwriting your unnormalized data
            output_path = f"progress_meeting/embeddings_norm/{item_id}_{embed_name}_norm.npy"
            
            save_numpy(single_embedding, output_path)

        print(f"Saved {len(ids)} normalized files for {embed_name}")


check_embeddings_normalization("progress_meeting/embeddings/")
check_embeddings_normalization("progress_meeting/embeddings_norm/")

print("\nEmbedding pipeline complete.")