from utils.io_utils import load_json, save_json

RAW_DIR = "data/aporia/raw_outputs"
LABELS_DIR = "data/aporia/labels"
LLM_NAMES = ["llama", "mistral", "phi"]


def label_responses(responses, answers):
    labels = []
    for response in responses:
        response_lower = response.lower()
        hit = any(ans.lower() in response_lower for ans in answers)
        labels.append("G" if hit else "H")
    return labels


def run_pipeline():
    for llm in LLM_NAMES:
        data = load_json(f"{RAW_DIR}/{llm}_responses.json")
        labelled = []

        for item in data:
            labels = label_responses(item["responses"], item["answers"])
            n_g = labels.count("G")
            n_h = labels.count("H")
            print(f"{llm} | {item['prompt_id']}: {n_g} Genuine, {n_h} Hallucinated")
            labelled.append({
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "labels": labels,
            })

        out_path = f"{LABELS_DIR}/{llm}_labels.json"
        save_json(labelled, out_path)
        print(f"Saved labels -> {out_path}\n")

    print("Labelling complete.")


if __name__ == "__main__":
    run_pipeline()
