import uuid
from datetime import datetime
from typing import Optional, Union

from langchain.schema import HumanMessage, SystemMessage


class Message:
    def __init__(
        self,
        content: str,
        role: str,
        user_id: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize a message.

        :param content: The content of the message (text or speech).
        :param role: The role of the sender (user/agent).
        :param user_id: The unique ID of the user (for accessing long-term memory in MongoDB).
        :param session_id: The unique ID of the current session (to track this conversation).
        :param timestamp: Timestamp when the message was created (defaults to now).
        """
        self.message_id = str(uuid.uuid4())  # Unique message identifier
        self.content = content
        self.role = role  # Can be 'user' or 'agent'
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = (
            timestamp if timestamp else datetime.now()
        )  # Default to current time

    def __repr__(self):
        return f"Message(user_id={self.user_id}, session_id={self.session_id}, role={self.role}, timestamp={self.timestamp})"

    def to_langchain_message(self) -> Union[HumanMessage, SystemMessage]:
        """
        Converts the current message object to a LangChain-compatible message
        (either HumanMessage or SystemMessage).
        """
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else:
            raise ValueError(f"Unsupported message role: {self.role}")
