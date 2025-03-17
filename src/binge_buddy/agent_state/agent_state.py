from abc import ABC
from typing import List, Optional

from binge_buddy.memory.memory import Memory
from binge_buddy.message.human_message import HumanMessage

class AgentState(ABC):
    def __init__(
        self,
        user_id: str,
        memories: List[Memory],
        current_user_message: HumanMessage,
        contains_information: bool,
        extracted_knowldge: List[Memory],
        needs_repair: bool,
        repair_message: Optional[str] = None
    ):
        self.user_id = user_id
        self.memories = memories
        self.current_user_message = current_user_message
        self.contains_information = contains_information
        self.extracted_knowldge = extracted_knowldge
        self.needs_repair = needs_repair
        self.repair_message = repair_message

