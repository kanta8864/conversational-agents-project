from binge_buddy.conversational_agent import ConversationalAgent
from binge_buddy.memory_db import MemoryDB
from binge_buddy.message_log import MessageLog


class ConversationalAgentManager:
    def __init__(self, llm, message_log, memory_handler):
        self.llm = llm  # Store shared LLM instance
        self.agents = {}  # Store agents per session
        self.memory_handler = memory_handler
        self.message_log = message_log

    def get_agent(self, user_id: str, session_id: str) -> ConversationalAgent:
        if session_id not in self.agents:
            self.agents[session_id] = ConversationalAgent(
                self.llm, self.message_log, self.memory_handler
            )  # Use shared LLM

        return self.agents[session_id]

    def remove_agent(self, session_id: str):
        """Cleanup when a session ends."""
        if session_id in self.agents:
            del self.agents[session_id]
