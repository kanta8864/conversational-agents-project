from typing import List, Optional

from binge_buddy.agent_state.agent_state import AgentState
from binge_buddy.memory.memory import Memory
from binge_buddy.message.human_message import HumanMessage

class EpisodicAgentState(AgentState):
    def __init__(
        self,
        user_id: str,
        memories: List[Memory],
        current_user_message: HumanMessage,
    ):
        super().__init__(
            user_id=user_id,
            memories=memories,
            current_user_message=current_user_message,
            contains_information=None,
            extracted_knowldge=None,
            needs_repair=None,
            repair_message=None
        )


