from typing import Optional

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.message import Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM


class MemorySentinel:
    def __init__(self, llm: OllamaLLM, message_log: MessageLog):
        """
        Initializes the MemorySentinel agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        :param message_log: The message_log that it needs to be observing
        """
        self.llm = llm
        self.message_log = message_log

        # System prompt for the memory sentinel to decide whether to store information
        self.system_prompt_initial = """
        Your job is to assess a brief chat history in order to determine if the conversation contains any details about a user's watching habits regarding streaming content. 
        You are part of a team building a knowledge base regarding a user's watching habits to assist in highly customized streaming content recommendations.
        You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.
        
        You are only interested in the following categories of information:
        1. **Movies and Genres the user likes** 
            - This helps recommend movies the user is likely to enjoy.  
        2. **Movies and Genres the user dislikes**  
            - This helps avoid recommending content the user won't enjoy.  
        3. **Favorite movies**  
            - This helps refine recommendations by finding similar movies.  
        4. **Movies the user wants to watch** 
            - This ensures the system prioritizes unwatched recommendations.  
        5. **Preferred streaming platforms**  
            - This ensures recommendations are available on the user's preferred services.          
        6. **Preferred genres**
            - This helps suggest content that aligns with the user's tastes.
        7. **Personality of the user**  
            - Example: *Enjoys lighthearted comedies; Finds fast-paced movies engaging.*  
        8. **Watching habits** 
            - This helps suggest content that fits the user's lifestyle.  
        9. **Frequency**  
            - This helps suggest content that fits the user's frequency of watching movies/shows.  
        10. **Avoid categories**  
            - Ensures the system respects the user’s hard limits.  
        11. **Character preferences**  
            - Example: *Prefers witty and sarcastic characters; Enjoys dark and mysterious protagonists.*  
        12. **Show length preferences**   
            - This ensures recommendations align with preferred pacing and format.  
        13. **Rewatching tendencies**   
            - This helps tailor recommendations for fresh or nostalgic content.  
        14. **Popularity preferences**  
            - This helps suggest content based on mainstream vs. niche tastes
        
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
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_sentinel_runnable = self.prompt | self.llm_runnable

    def run(self) -> Optional[bool]:
        """
        Analyzes the current message to check if it contains useful information for long-term memory.

        :return: True if the message contains useful information, otherwise False.
        """

        # Get the latest message from the log
        message = self.message_log.get_last_message()

        if not message:
            return None

        print(message.to_langchain_message())

        # Run the pipeline and get the response
        response = self.memory_sentinel_runnable.invoke(
            [message.to_langchain_message()]
        )

        # Return True/False based on the response
        return utils.remove_think_tags(response) == "TRUE"


if __name__ == "__main__":
    # Initialize the message log and LLM (for now, using a mock LLM)
    llm = OllamaLLM()
    message_log = MessageLog("user", "session")
    memory_sentinel = MemorySentinel(llm=llm, message_log=message_log)

    # Test message
    current_message = Message(
        content="I love watching sci-fi movies like The Matrix!",
        role="user",
        session_id="session",
        user_id="user",
    )

    message_log.add_message(current_message)

    response = memory_sentinel.run()
    print("Store this information?", response)
