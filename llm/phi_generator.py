from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from llm.base_llm import BaseLLM
from config.settings import DEVICE, MAX_NEW_TOKENS, TEMPERATURE
from config.model_config import LLM_MODELS


class PhiGenerator(BaseLLM):

    def __init__(self):

        model_name = LLM_MODELS["phi"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response