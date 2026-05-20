from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from llm.base_llm import BaseLLM
from config.settings import DEVICE, MAX_NEW_TOKENS, TEMPERATURE
from config.model_config import LLM_MODELS


class MistralGenerator(BaseLLM):

    def __init__(self):

        model_name = LLM_MODELS["mistral"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate(self, prompt: str):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response