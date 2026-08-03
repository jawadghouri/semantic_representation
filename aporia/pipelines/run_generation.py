import gc
import torch
from tqdm import tqdm

from llm.llama_generator import LlamaGenerator
from llm.mistral_generator import MistralGenerator
from llm.phi_generator import PhiGenerator
from utils.io_utils import load_json, save_json

N_RESPONSES = 15
PROMPTS_PATH = "data/aporia/aporia_prompts.json"
OUTPUT_DIR = "data/aporia/raw_outputs"

model_blueprints = {
    "llama": LlamaGenerator,
    "mistral": MistralGenerator,
    "phi": PhiGenerator,
}


def run_pipeline():
    prompts = load_json(PROMPTS_PATH)

    for model_name, generator_class in model_blueprints.items():
        print(f"\nActivating: {model_name}")
        generator = generator_class()

        results = []

        for item in tqdm(prompts, desc=f"{model_name} prompts"):
            prompt_id = item["id"]
            prompt_text = item["prompt"]
            answers = item["answers"]

            responses = []
            for i in range(N_RESPONSES):
                response = generator.generate(prompt_text)
                responses.append(response)

            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "answers": answers,
                "responses": responses
            })

        output_path = f"{OUTPUT_DIR}/{model_name}_responses.json"
        save_json(results, output_path)
        print(f"Saved {len(results)} prompts x {N_RESPONSES} responses -> {output_path}")

        del generator
        gc.collect()
        torch.cuda.empty_cache()
        print(f"VRAM cleared after {model_name}.")

    print("\nGeneration complete.")


if __name__ == "__main__":
    run_pipeline()
