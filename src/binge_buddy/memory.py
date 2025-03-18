from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, TypedDict

# Avoid circular import issue
if TYPE_CHECKING:
    from binge_buddy.agent_state.states import AgentState  # Only used for type hints


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

    @staticmethod
    def create(information: str, state: AgentState):
        if state.state_type == "semantic":
            return SemanticMemory(information)
        else:
            current_user_message = state.current_user_message
            return EpisodicMemory(information, current_user_message.timestamp)

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(data["memory_type"], data["information"])

    def __repr__(self):
        raise NotImplementedError("Should be implemented in the inherited class")

    def __str__(self):
        raise NotImplementedError("Should be implemented in the inherited class")


class SemanticMemory(Memory):
    def __init__(self, information: str):
        super().__init__("Semantic", information)

    def get_type(self) -> str:
        return self.memory_type

    def __repr__(self):
        return f"SemanticMemory(information={self.information})"

    def __str__(self):
        return f"SemanticMemory(information={self.information})"


class EpisodicMemory(Memory):
    timestamp: datetime

    def __init__(self, information: str, timestamp: datetime):
        super().__init__("Episodic", information)
        self.timestamp = timestamp

    def get_type(self) -> str:
        return self.memory_type

    def __repr__(self):
        return f"EpisodicMemory(information={self.information}, timestamp={self.timestamp})"

    def __str__(self):
        return f"EpisodicMemory(information={self.information}, timestamp={self.timestamp})"
