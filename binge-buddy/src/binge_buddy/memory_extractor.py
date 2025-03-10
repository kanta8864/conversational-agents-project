from typing import Dict, List, Optional

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.message import Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM


class MemoryExtractor:
    def __init__(self, llm: OllamaLLM, message_log: MessageLog):
        """
        Initializes the MemorySentinel agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
        self.llm = llm
        self.message_log = message_log

        # System prompt for the memory sentinel to decide whether to store information
        self.system_prompt_initial = """
        You are a supervisor managing a team of movie recommendation experts.

        Your team's job is to build the perfect knowledge base about a user's movie preferences in order to provide highly personalized recommendations.

        The knowledge base should ultimately consist of many discrete pieces of information that add up to a rich persona (e.g. Likes action movies; Dislikes horror; Favorite movie is Inception; Watches mostly on Netflix; Enjoys long TV series; Prefers witty and sarcastic characters etc).

        Every time you receive a message, you will evaluate whether it contains any information worth recording in the knowledge base.

        A message may contain multiple pieces of information that should be saved separately.

        You are only interested in the following categories of information:

        1. **Movies and Genres the user likes** (e.g. Likes sci-fi; Enjoyed Inception)  
        - This helps recommend movies the user is likely to enjoy.  

        2. **Movies and Genres the user dislikes** (e.g. Dislikes horror; Didn't enjoy The Conjuring)  
        - This helps avoid recommending content the user won't enjoy.  

        3. **Favorite movies** (e.g. Favorite movie is The Matrix)  
        - This helps refine recommendations by finding similar movies.  

        4. **Movies the user wants to watch** (e.g. Wants to watch Oppenheimer)  
        - This ensures the system prioritizes unwatched recommendations.  

        5. **Preferred streaming platforms** (e.g. Watches mostly on Netflix and Hulu)  
        - This ensures recommendations are available on the user's preferred services.  

        6. **Personality of the user**  
        - Example: *Enjoys lighthearted comedies; Finds fast-paced movies engaging.*  

        7. **Watching habits** (e.g. Prefers binge-watching TV shows over time)  
        - This helps suggest content that fits the user's lifestyle.  

        8. **Frequency** (e.g. Watches movies/shows two days a week for about 2 hours each)  
        - This helps suggest content that fits the user's frequency of watching movies/shows.  

        9. **Avoid categories** (e.g. Avoids anything with excessive gore; Doesn't like political dramas)  
        - Ensures the system respects the user’s hard limits.  

        10. **Tone**  
        - tone of the user's messages to understand if the user is frustrated, happy etc.  

        11. **Character preferences**  
        - Example: *Prefers witty and sarcastic characters; Enjoys dark and mysterious protagonists.*  

        12. **Show length preferences** (e.g. Prefers miniseries; Enjoys long TV series with deep character development)  
        - This ensures recommendations align with preferred pacing and format.  

        13. **Rewatching tendencies** (e.g. Frequently rewatches Friends; Rarely rewatch movies)  
        - This helps tailor recommendations for fresh or nostalgic content.  

        14. **Popularity preferences** (e.g. Prefers cult classics over mainstream hits)  
        - This helps suggest content based on mainstream vs. niche tastes.  

        When you receive a message, you perform a sequence of steps consisting of:

        1. Analyze the most recent Human message for new information. You will see multiple messages for context, but we are only looking for new information in the most recent message.  
        2. Extract new information and return new memories from this information. We can have multiple pieces of information so we can also get multiple memories. 

        I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information.

        Take a deep breath, think step by step and in the end simply return a list of new information extracted under the title "Memory Extractor Result:" Be concrete with your final result.
        List out all the memories as a json array in the format 
            [
                { "memory" : ...},
                { "memory" : ...},
                ...
            ]
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_extractor_runnable = self.prompt | self.llm_runnable

    def format_memories(self, response: str) -> List[Dict]: ...

    def run(self) -> Optional[List[Dict]]:
        """
        Analyzes the current message to check if it contains useful information for long-term memory.

        :return: List of dictionary objects with memories.
        """

        if len(self.message_log) == 0:
            return None

        messages = []
        for message in self.message_log:
            if message.role == "user":
                messages.append(message.to_langchain_message())

        print(messages)
        # Run the pipeline and get the response
        response = utils.remove_think_tags(
            self.memory_extractor_runnable.invoke({"messages": messages})
        )

        return response


if __name__ == "__main__":
    llm = OllamaLLM()
    message_log = MessageLog(user_id="user", session_id="session")
    memory_extractor = MemoryExtractor(llm=llm, message_log=message_log)

    messages = [
        Message(
            role="user",
            content="I really enjoyed Inception and I want to watch Oppenheimer next.",
            user_id="user",
            session_id="session",
        ),
        Message(
            role="user",
            content="Sci-fi movies are really great but crime thrillers are even better.",
            user_id="user",
            session_id="session",
        ),
    ]

    for message in messages:
        message_log.add_message(message)

    response = memory_extractor.run()

    print(response)

