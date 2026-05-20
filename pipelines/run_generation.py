import gc
import torch
from tqdm import tqdm

# Import the Class Blueprints (No objects are created here)
from llm.llama_generator import LlamaGenerator
from llm.phi_generator import PhiGenerator
from llm.mistral_generator import MistralGenerator
from utils.io_utils import load_json, save_json

def run_pipeline():
    # 1. Load your evaluation prompts from Stage 1
    prompts = load_json("data/prompts/prompts.json")

    # 2. Store Class Blueprints rather than instantiating objects.
    # CRITICAL: Notice there are no parentheses () after LlamaGenerator or PhiGenerator.
    # This keeps VRAM usage at 0 GB right now.
    model_blueprints = {
        "llama": LlamaGenerator,
        "phi": PhiGenerator,
        "mistral": MistralGenerator
    }

    # 3. Iterate through the blueprint dictionary
    for model_name, generator_class in model_blueprints.items():
        results = []
        print(f"\n🚀 Activating Model Space: {model_name}")
        
        # 4. Instantiate ONLY the active model.
        # This allocates memory dynamically right before processing the prompts.
        generator = generator_class()

        print(f"Running generation loop for {model_name}...")
        for prompt in tqdm(prompts):
            response = generator.generate(prompt)

            results.append({
                "prompt": prompt,
                "response": response
            })

        # 5. Save Stage 3 text outputs directly to your raw data directory
        save_json(
            results,
            f"data/raw_outputs/{model_name}_outputs.json"
        )
        print(f"💾 Saved outputs to data/raw_outputs/{model_name}_outputs.json")

        # 6. THE VRAM PURGE
        # Completely destroy the generator instance and force a hard memory flush
        del generator
        gc.collect()             # Clear dead Python memory references
        torch.cuda.empty_cache() # Force the NVIDIA card to release VRAM blocks
        
        print(f"🧹 Flushed VRAM clean. Ready for the next model space.")

    print("\n✅ All models processed successfully. Generation stage complete!")

if __name__ == "__main__":
    run_pipeline()