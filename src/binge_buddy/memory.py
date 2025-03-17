from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict


class MemoryDict(TypedDict):
    memory_type: str
    information: str


class Memory(ABC):
    def __init__(self, memory_type: str, information: str):
        self.memory_type = memory_type
        self.information = information

    @abstractmethod
    def get_type(self) -> str:
        """Returns the type of memory."""

    def as_dict(self) -> MemoryDict:
        """Convert memory object to dictionary."""
        return {
            "memory_type": self.memory_type,
            "information": self.information,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(data["memory_type"], data["information"])


class SemanticMemory(Memory):
    def __init__(self, information: str):
        super().__init__("Semantic", information)

    def get_type(self) -> str:
        return self.memory_type


class EpisodicMemory(Memory):
    def __init__(self, information: str):
        super().__init__("Episodic", information)

    def get_type(self) -> str:
        return self.memory_type
