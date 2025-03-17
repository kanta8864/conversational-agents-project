from datetime import datetime
from typing import Optional, Union
from binge_buddy.message.message import Message
from langchain.schema import HumanMessage as LangChainHumanMessage

class HumanMessage(Message):
    def __init__(
        self,
        content: str,
        user_id: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(content=content, role="user", user_id=user_id, session_id=session_id, timestamp=timestamp)

    def to_langchain_message(self) -> LangChainHumanMessage:
        return LangChainHumanMessage(content=self.content)
    