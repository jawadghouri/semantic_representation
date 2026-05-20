from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from llm.base_llm import BaseLLM

from config.settings import DEVICE, MAX_NEW_TOKENS, TEMPERATURE
from config.model_config import LLM_MODELS


class LlamaGenerator(BaseLLM):
    def __init__(self):

        model_name = LLM_MODELS['llama']

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate(self, prompt: str):
        
        messages = [
            {"role": "user", "content": prompt}
        ]

        templated_input = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            return_dict=True
        ).to(DEVICE)

        outputs= self.model.generate(
            input_ids=templated_input["input_ids"],
            attention_mask=templated_input["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 1. Decode EVERYTHING (the prompt and the model's new answer combined)
        full_decoded_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        # 2. Look for your original prompt string inside the massive text block
        if prompt in full_decoded_text:
            # Split the text at your question, and take everything that came AFTER it [-1]
            response = full_decoded_text.split(prompt)[-1].strip()
        else:
            # Emergency fallback if something completely weird happened
            response = full_decoded_text.strip()

        return response   