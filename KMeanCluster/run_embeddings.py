import os
from embeddings.minilm_embedder import MiniLMEmbedder
from utils.io_utils import load_json, save_numpy
from utils.norm_utils import check_embeddings_normalization

# ---------------------------------------------------
# LOAD RAW OUTPUTS
# ---------------------------------------------------

model_files = {
    # 1. Constraint Prompt files (R*)
    "C1": "data/raw_inputs/constraint_prompts/C1.json",
    "C2": "data/raw_inputs/constraint_prompts/C2.json",
    "C3": "data/raw_inputs/constraint_prompts/C3.json",
    
    # 2. LLM Benchmark files (A*)
     "A_responses": "data/raw_inputs/online_prompts.json",
    
    # 3. Ground Truth files (V* linked to A*)
     "V_ground_truth": "data/raw_inputs/ground_truth.json",
}

# ---------------------------------------------------
# LOAD EMBEDDERS
# ---------------------------------------------------

embedders = {
    "minilm": MiniLMEmbedder(),
}

# ---------------------------------------------------
# PROCESS EACH FILE
# ---------------------------------------------------

for group_key, filepath in model_files.items():
    print(f"\nProcessing File: {group_key} ({filepath})")
    data = load_json(filepath)

    texts = []
    ids = []

    for item in data:
        # Case 1: Ground truth format -> {"id": "V1", "prompt_id": "A1", "correct_answer": "..."}
        if "correct_answer" in item:
            v_id = item.get("id", "V_unknown")
            prompt_id = item.get("prompt_id", "A_unknown")
            text = item["correct_answer"]
            
            if text:
                texts.append(text)
                ids.append(f"{v_id}_{prompt_id}_correctanswer")

        # Case 2: LLM responses format -> {"id": "A1", "responses": {"chatgpt": "...", ...}}
        elif isinstance(item.get("responses"), dict):
            item_id = item.get("id", "unknown_id")

            for llm_name, response_text in item["responses"].items():
                if response_text:
                    texts.append(response_text)
                    ids.append(f"{item_id}_{llm_name}")

        # Case 3: Original constraint format -> {"id": "R1", "responses": [...] or "response": "..."}
        else:
            item_id = item.get("id", "unknown_id")
            responses_field = item.get("responses")

            if isinstance(responses_field, list):
                for idx, response_text in enumerate(responses_field):
                    if response_text:
                        texts.append(response_text)
                        ids.append(f"{item_id}_{idx}")
            elif "response" in item and item["response"]:
                texts.append(item["response"])
                ids.append(f"{item_id}_response")

    print(f"Total texts to embed: {len(texts)}")

    # ---------------------------------------------------
    # RUN UNNORMALIZED EMBEDDINGS
    # ---------------------------------------------------
    for embed_name, embedder in embedders.items():
        print(f"Embedding with: {embed_name}")
        embeddings = embedder.encode(texts)

        for text_id, single_embedding in zip(ids, embeddings):
            output_path = f"data/processed/embeddings_unnorm/{text_id}_{embed_name}.npy"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_numpy(single_embedding, output_path)

check_embeddings_normalization("data/processed/embeddings_unnorm/")

print("\nEmbedding pipeline complete.")