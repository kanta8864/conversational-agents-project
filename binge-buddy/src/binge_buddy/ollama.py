# Create custom LLM class
from typing import Any, List, Mapping, Optional

import requests
from langchain.llms.base import LLM


# Define the custom OllamaLLM class
class OllamaLLM:
    # Using the distilled model with 8B parameters
    model: str = "deepseek-r1:8b"
    temperature: float = 0.0

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    @property
    def _llm_type(self) -> str:
        return "ollama"
