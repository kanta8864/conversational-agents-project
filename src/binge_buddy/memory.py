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
    attribute: Optional[str]


class Memory(ABC):
    def __init__(self, memory_type: str, information: str, attribute: Optional[str]):
        self.memory_type = memory_type
        self.information = information
        self.attribute = attribute

    @abstractmethod
    def get_type(self) -> str:
        """Returns the type of memory."""

    def as_dict(self) -> MemoryDict:
        """Convert memory object to dictionary."""
        return {
            "memory_type": self.memory_type,
            "information": self.information,
            "attribute": self.attribute,
        }

    def has_attribute(self) -> bool:
        if self.attribute is not None:
            return True
        return False

    @staticmethod
    def create(information: str, state: AgentState, attribute: Optional[str] = None):
        if state.state_type == "semantic":
            return SemanticMemory(information, attribute)
        current_user_message = state.current_user_message
        return EpisodicMemory(information, current_user_message.timestamp, attribute)

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(data["memory_type"], data["information"], data["attribute"])

    def as_db_entry(self):
        raise NotImplementedError("Should be implemented in the inherited class")

    def __repr__(self):
        raise NotImplementedError("Should be implemented in the inherited class")

    def __str__(self):
        raise NotImplementedError("Should be implemented in the inherited class")


class SemanticMemory(Memory):
    def __init__(self, information: str, attribute: Optional[str] = None):
        super().__init__("Semantic", information, attribute)

    def get_type(self) -> str:
        return self.memory_type

    def as_db_entry(self):
        if self.attribute is not None:
            return {self.attribute: self.information}
        else:
            return {"UNKNOWN": self.information}

    def __repr__(self):
        return f"SemanticMemory(information={self.information}, attribute={self.attribute})"

    def __str__(self):
        return f"SemanticMemory(information={self.information}, attribute={self.attribute})"


class EpisodicMemory(Memory):
    timestamp: datetime

    def __init__(
        self, information: str, timestamp: datetime, attribute: Optional[str] = None
    ):
        super().__init__("Episodic", information, attribute)
        self.timestamp = timestamp

    def as_db_entry(self):
        if self.attribute is not None:
            return {
                self.attribute: self.information,
                "timestamp": self.timestamp.isoformat(),
            }
        else:
            return {
                "UNKNOWN": self.information,
                "timestamp": self.timestamp.isoformat(),
            }

    def get_type(self) -> str:
        return self.memory_type

    def __repr__(self):
        return f"EpisodicMemory(information={self.information}, attribute={self.attribute}, timestamp={self.timestamp})"

    def __str__(self):
        return f"EpisodicMemory(information={self.information}, attribute={self.attribute}, timestamp={self.timestamp})"
