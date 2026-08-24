##
# This generates embedding files for each response in the constraint prompts.
# It handles both normalized and unnormalized embeddings,
# saving the resulting embeddings to specified directories.
#
# Supports two JSON shapes per item:
#   1) {"id": ..., "response": "..."}                     -> single response
#   2) {"id": ..., "responses": ["...", "..."]}            -> list of responses
#   3) {"id": ..., "prompt": ..., "responses": {           -> dict of llm -> response
#          "chatgpt": "...", "gemini": "...", ...
#      }}
##

from http.client import responses

from embeddings.bge_embedder import BGEEmbedder
from embeddings.e5_embedder import E5Embedder
from embeddings.minilm_embedder import MiniLMEmbedder

from utils.io_utils import load_json, save_numpy

from utils.norm_utils import check_embeddings_normalization

# ---------------------------------------------------
# LOAD RAW OUTPUTS
# ---------------------------------------------------

model_files = {
    "C1": "KMeanCluster/data/raw_inputs/constraint_prompts/C1.json",
    "C2": "KMeanCluster/data/raw_inputs/constraint_prompts/C2.json",
    "C3": "KMeanCluster/data/raw_inputs/constraint_prompts/C3.json",
    "A": "KMeanCluster/data/raw_inputs/online_prompts.json"
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

        # single response version: {"response": "..."}
        if "response" in item:
            texts.append(item["response"])
            ids.append(item_id)

        elif "responses" in item:
            responses_field = item["responses"]

            # multi-response version (list): {"responses": ["...", "..."]}
            if isinstance(responses_field, list):
                for idx, response in enumerate(responses_field):
                    texts.append(response)
                    # Append an index to the ID so files don't overwrite each other
                    ids.append(f"{item_id}_{idx}")

            # multi-model version (dict): {"responses": {"chatgpt": "...", "gemini": "..."}}
            elif isinstance(responses_field, dict):
                for llm_name, response in responses_field.items():
                    texts.append(response)
                    # Append the LLM name to the ID so files don't overwrite each other
                    ids.append(f"{item_id}_{llm_name}")

            else:
                print(f"Warning: unrecognized 'responses' type for id={item_id}, skipping")

        else:
            print(f"Warning: item id={item_id} has neither 'response' nor 'responses', skipping")

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
            
            output_path = f"KMeanCluster/data/processed/embeddings_unnorm/{item_id}_{embed_name}.npy"
            
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
            output_path = f"KMeanCluster/data/processed/embeddings_norm/{item_id}_{embed_name}_norm.npy"
            
            save_numpy(single_embedding, output_path)

        print(f"Saved {len(ids)} normalized files for {embed_name}")


check_embeddings_normalization("KMeanCluster/data/processed/embeddings_unnorm/")
check_embeddings_normalization("KMeanCluster/data/processed/embeddings_norm/")

print("\nEmbedding pipeline 2.0 complete.")