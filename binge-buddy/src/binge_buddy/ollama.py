import requests
from langchain.llms.base import LLM
from langchain.schema import PromptValue  # Import ChatPromptValue

class OllamaLLM(LLM):  # Inherit from the LLM base class
    model: str = "deepseek-r1:8b"  # Default model
    temperature: float = 0.0  # Default temperature

    def _call(self, prompt: str, **kwargs) -> str:
        """
        Call the Ollama API with the given prompt and return the response.
        """
        # Convert ChatPromptValue to a string if necessary
        if isinstance(prompt, PromptValue):
            prompt = str(prompt)  # Convert ChatPromptValue to a string

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,  # Use the stringified prompt
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