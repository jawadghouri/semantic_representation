from llm.llama_generator import LlamaGenerator

from utils.io_utils import load_json, save_json

from tqdm import tqdm

prompts = load_json("data/prompts/prompts.json")

models = {
    "llama": LlamaGenerator()
}

for model_name, generator in models.items():

    results = {}

    print(f"\n Running Model: {model_name}\n")

    for prompt in tqdm(prompts):
        response = generator.generate(prompt)

        results.append({
            "prompt": prompt,
            "response": response
        })

    save_json(
        results,
        f"data/raw_outputs/{model_name}_outputs.json"
    )

    print("\nGeneration complete.")