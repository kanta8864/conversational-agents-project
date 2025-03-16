from typing import Iterator, List, Optional

from .message import Message


class MessageLog:
    def __init__(self, session_id: str, user_id: str):
        """
        Initialize the message log for the current session.

        :param session_id: The unique session identifier.
        :param user_id: The unique user identifier.
        """
        self.session_id = session_id
        self.user_id = user_id
        self.messages: List[Message] = []  # List of Message objects for this session

    def add_message(self, message: Message) -> None:
        """
        Adds a Message object to the session log.

        :param message: The Message object to be added to the log.
        """
        self.messages.append(message)

    def get_history(self) -> List[str]:
        """
        Returns the conversation history as a list of message contents, including roles.

        :return: A list of message contents with role information.
        """
        return [f"{message.role}: {message.content}" for message in self.messages]

    def get_last_message(self) -> Optional[Message]:
        """
        Returns the last message content in the session log, including its role.

        :return: The last message content along with its role or None if no messages exist.
        """
        if self.messages:
            return self.messages[-1]
        return None

    def clear_log(self) -> None:
        """
        Clears the message log (resets the log for a new session or scenario).
        """
        self.messages.clear()

    def get_message_count(self) -> int:
        """
        Get the number of messages in the current session.

        :return: The number of messages in the log.
        """
        return len(self.messages)

    def __len__(self) -> int:
        """
        Allows using len() on the MessageLog to get the number of messages.

        :return: The number of messages in the log.
        """
        return len(self.messages)

    def __str__(self) -> str:
        """
        Return a string representation of the message log, useful for debugging.

        :return: A string representation of the log with roles and contents.
        """
        return "\n".join(
            f"{message.role}: {message.content}" for message in self.messages
        )

    def __iter__(self) -> Iterator[Message]:
        """
        Returns a new iterator for the messages each time __iter__ is called.
        """
        return MessageLogIterator(self.messages)


class MessageLogIterator:
    def __init__(self, messages: List[Message]):
        """
        Initialize the iterator with a list of messages.
        """
        self.messages = messages
        self._index = 0  # Initialize index to start from the first message

    def __iter__(self):
        """
        Return the iterator itself.
        """
        return self

    def __next__(self) -> Message:
        """
        Return the next message in the list, or raise StopIteration if done.
        """
        if self._index < len(self.messages):
            message = self.messages[self._index]
            self._index += 1
            return message
        else:
            raise StopIteration
