from typing import List

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

from binge_buddy.message import Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM


class MemorySentinel:
    def __init__(self, long_term_memory_db, message_log: "MessageLog", llm: OllamaLLM):
        """
        Initializes the MemorySentinel agent.

        :param long_term_memory_db: The long-term memory database (MongoDB or similar).
        :param message_log: The message log for the current session.
        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
        self.long_term_memory_db = long_term_memory_db
        self.message_log = message_log
        self.llm = llm

        # System prompt for the memory sentinel to decide whether to store information
        self.system_prompt_initial = """
        Your job is to assess a brief chat history in order to determine if the conversation contains any details about a user's watching habits regarding streaming content. 
        You are part of a team building a knowledge base regarding a user's watching habits to assist in highly customized streaming content recommendations.
        You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.
        
        You are only interested in the following categories of information:
        1. The user's likes (e.g., "I love action movies!")
        2. The user's dislikes (e.g., "I hate horror movies!")
        3. The user's favorite shows or movies (e.g., "My favorite show is Game of Thrones!")
        4. The user's desire to watch a show or movie (e.g., "I really want to watch the new Marvel movie!")
        5. The user's preferred streaming platform (e.g., "I watch everything on Netflix!")
        6. The user's preferred genre (e.g., "I love romantic comedies!")
        7. The user's disinterest in a show or movie (e.g., "I'm not interested in watching that new series.")
        8. The user's personality traits that may influence their watching habits (e.g., "I'm a huge fan of sci-fi because I love imagining the future.")
        9. The user's watching habits (e.g., "I watch a movie every night before bed.")
        10. The frequency of the user's watching habits (e.g., "I watch TV shows every weekend.")
        11. The user's avoidance of certain types of content (e.g., "I avoid watching horror movies because they scare me.")
        12. The user's preferred tone of content (e.g., "I prefer light-hearted comedies over dark dramas.")
        13. The user's preference for certain types of characters (e.g., "I love shows with strong female leads.")
        14. The user's preferred length of shows or movies (e.g., "I prefer short episodes that I can watch during my lunch break.")
        15. The user's tendency to rewatch shows or movies (e.g., "I rewatch my favorite movie every year.")
        
        When you receive a message, you perform a sequence of steps consisting of:
        1. Analyze the message for information.
        2. If it has any information worth recording, return TRUE. If not, return FALSE.
        
        You should ONLY RESPOND WITH TRUE OR FALSE. Absolutely no other information should be provided.
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Remember, only respond with TRUE or FALSE. Do not provide any other information.",
                ),
            ]
        )

    def format_messages(self, messages: List["Message"]) -> str:
        """
        Converts messages into LangChain message objects and formats them for the LLM.
        """
        langchain_messages = [msg.to_langchain_message() for msg in messages]
        chat_prompt_value = self.prompt.format_messages(messages=langchain_messages)
        formatted_prompt = "\n".join([message.content for message in chat_prompt_value])
        return formatted_prompt

    def run_analysis_pipeline(self, formatted_prompt: str) -> str:
        """
        Sends the formatted prompt to the LLM and returns the response.
        """
        return self.llm._call(formatted_prompt)

    def analyze_message(self, current_message: str) -> str:
        """
        Analyzes the current message to check if it contains useful information for long-term memory.

        :param current_message: The message to be analyzed.
        :return: True if the message contains useful information, otherwise False.
        """

        # Get the full message history
        message_history = self.message_log.get_history()

        # Create Message objects for the history
        messages = [Message(role="user", content=msg) for msg in message_history]

        # Format the messages for the LLM
        formatted_prompt = self.format_messages(messages)

        # Run the pipeline and get the response
        response = self.run_analysis_pipeline(formatted_prompt)

        # Return True/False based on the response
        return response


if __name__ == "__main__":
    # Initialize the message log and LLM (for now, using a mock LLM)
    llm = OllamaLLM()  # or any other LLM model you are using
    message_log = MessageLog(session_id="12345", user_id="user123")
    memory_sentinel = MemorySentinel(
        long_term_memory_db=None, message_log=message_log, llm=llm
    )

    # Test message
    current_message = "I love watching sci-fi movies like The Matrix!"

    # Analyze the message
    result = memory_sentinel.analyze_message(current_message)
    print("Store this information?", result)

