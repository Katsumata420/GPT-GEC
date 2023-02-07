from typing import Any, List

import openai

from .config import OpenAIConfig


class GPTModel:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        openai.api_key = config.api_key

    def generate(self, input_text: str) -> str:
        """Error Correction for input text with GPT"""
        gpt_input_text = f"{self.config.prompt}\n\n{input_text}\n\n"

        gpt_output = openai.Completion.create(
            model=self.config.model,
            prompt=gpt_input_text,
            max_tokens=self.config.max_token,
            temperature=self.config.temperature,
            top_p=1,
            n=1,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
        )
        
        correct_sentence = gpt_output["choices"][0]["text"].strip()

        return correct_sentence

    def check_available_model(self) -> List[Any]:
        available_models = openai.Model.list()
        return available_models
