from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


class MemoryExtractor: 
    def __init__(self, llm: OllamaLLM):
        """
        Initializes the MemorySentinel agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
        self.llm = llm

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
        2. Compare this to the knowledge you already have.  
        3. Determine if this is new knowledge, an update to existing knowledge, or if previously stored information needs to be corrected. In case it is an updat to existing knowledge, 
        aggregate the old and new information, instead of overwriting and retain old knowledge too. 
        - Example: A movie the user previously disliked might now be a favorite, or they might have switched to a new streaming platform.  

        Here are the existing bits of information we have about the user:

        ```
        {memories}
        ```

        Call the right tools to save the information, then respond with DONE. If you identiy multiple pieces of information, call everything at once. You only have one chance to call tools.

        I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or change any incorrect information.

        Take a deep breath, think step by step, and then analyze the following message:
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_extractor_runnable = self.prompt | self.llm_runnable

if __name__ == "__main__":
    llm = OllamaLLM()
    memory_extractor = MemoryExtractor(llm=llm)

    messages = [
        {"role": "user", "content": "I really enjoyed Inception and I want to watch Oppenheimer next."},
    ]

    memories = """
    - Likes sci-fi movies
    - Favorite movie is The Matrix
    - Prefers Netflix for streaming
    """

    response = memory_extractor.memory_extractor_runnable.invoke({
        "messages": messages,  
        "memories": memories   
    })
    print(response)