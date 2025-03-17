from abc import ABC, abstractmethod

from binge_buddy.agent_state.states import AgentState, AgentStateDict
from binge_buddy.ollama import OllamaLLM


class BaseAgent(ABC):
    def __init__(self, llm: OllamaLLM, system_prompt_initial: str):
        self.llm = llm
        self.system_prompt_initial = system_prompt_initial

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        pass
