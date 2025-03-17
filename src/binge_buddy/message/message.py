import uuid
from datetime import datetime
from typing import Optional, Union
from abc import ABC, abstractmethod
from langchain.schema import HumanMessage as LangChainHumanMessage, SystemMessage as LangChainSystemMessage

class Message(ABC):
    def __init__(
        self,
        content: str,
        role: str,
        user_id: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
    ):
        self.message_id = str(uuid.uuid4()) 
        self.content = content
        self.role = role  
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp if timestamp else datetime.now()

    @abstractmethod
    def to_langchain_message(self) -> Union[LangChainHumanMessage, LangChainSystemMessage]:
        pass

    def from_langchain_message(self, lc_message: Union[LangChainHumanMessage, LangChainSystemMessage], user_id: str, session_id: str):
        # Import inside the method to avoid circular import
        from binge_buddy.message.human_message import HumanMessage
        from binge_buddy.message.system_message import SystemMessage
        
        if isinstance(lc_message, LangChainHumanMessage):
            return HumanMessage(content=lc_message.content, user_id=user_id, session_id=session_id)
        elif isinstance(lc_message, LangChainSystemMessage):
            return SystemMessage(content=lc_message.content, user_id=user_id, session_id=session_id)
        else:
            raise ValueError("Unknown message type")

    def __repr__(self):
        return f"AbstractMessage(user_id={self.user_id}, session_id={self.session_id}, role={self.role}, timestamp={self.timestamp})"

    def __str__(self):
        return f"AbstractMessage(role:{self.role}, content:{self.content})"
