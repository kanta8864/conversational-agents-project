from datetime import datetime
from typing import Optional
from binge_buddy.message.message import Message
from langchain.schema import SystemMessage as LangChainSystemMessage

class SystemMessage(Message):
    def __init__(
        self,
        content: str,
        user_id: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(content=content, role="system", user_id=user_id, session_id=session_id, timestamp=timestamp)

    def to_langchain_message(self) -> LangChainSystemMessage:
        return LangChainSystemMessage(content=self.content)