from binge_buddy.conversational_agent import ConversationalAgent
from binge_buddy.memory_db import MemoryDB
from binge_buddy.message_log import MessageLog


class ConversationalAgentManager:
    def __init__(self, llm):
        self.llm = llm  # Store shared LLM instance
        self.agents = {}  # Store agents per session

    def get_agent(self, user_id: str, session_id: str) -> ConversationalAgent:
        if session_id not in self.agents:
            message_log = MessageLog(user_id, session_id)
            memory_db = MemoryDB()
            self.agents[session_id] = ConversationalAgent(self.llm, message_log, memory_db)  # Use shared LLM
        
        return self.agents[session_id]

    def remove_agent(self, session_id: str):
        """Cleanup when a session ends."""
        if session_id in self.agents:
            del self.agents[session_id]
