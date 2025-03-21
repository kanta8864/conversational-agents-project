import json

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.ollama import OllamaLLM


class MemoryAggregator:
    def __init__(self, llm: OllamaLLM):
        """
        Initializes the MemoryAggregator agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
        self.llm = llm

        # System prompt for the memory aggregator to know what needs to be done
        self.system_prompt_initial = """
        You are a supervisor managing a team of movie recommendation experts.

        Your team's job is to build and maintain a comprehensive knowledge base about a user's movie preferences to provide highly personalized recommendations.

        The knowledge base consists of multiple discrete pieces of information that collectively form a rich user profile. These pieces of information should always be categorized properly to maintain structure and accuracy.

        Whenever you receive a new message, your primary responsibility is to extract and integrate relevant information into the knowledge base without losing any details. 

        Here is the information that you are given with: 

        ## New memories
        {extracted_knowledge} 

        ## Existing memories
        {existing_memories} 

        ### Task Breakdown:

        1. **Extract and Categorize Information:**
        - Assign each extracted piece of information in {extracted_knowledge} to its appropriate attribute from the predefined list below.
        - A single message may contain multiple relevant pieces of information that should be categorized separately.
        - It could be that an attribute is empty, because the user didn't provide any information for that attribute. In that case, you don't need to add anything to that specific attribute in memory.

        2. **Compare with Existing Knowledge:**
        - Retrieve the current knowledge base stored in `{existing_memories}`.
        - Identify whether the new knowledge is:
            - **Completely new information** that needs to be added.
            - **An update to existing knowledge** that should be aggregated.
            - **A correction to previously stored knowledge** (e.g., if a user has changed their opinion about a movie or switched streaming platforms).

        3. **Aggregate Without Losing Information:**
        - If a piece of information already exists in `{existing_memories}`, **do not overwrite it**. Instead, merge the old and new information logically while ensuring clarity.
        - Maintain **historical context** where relevant. For example:
            - If a user initially disliked a movie but now enjoys it, update the record to reflect this change.
            - If a user watches movies on multiple platforms, ensure all platforms are recorded rather than replacing the old preference.
        - The goal is to **build upon** existing knowledge rather than replace it unless explicitly necessary.

        4. **Attributes of Interest:**
        - Every extracted piece of information must belong to one of the following attributes:
            - LIKES: **Movies and Genres the user likes** 
                - This helps recommend movies the user is likely to enjoy.  
            - DISLIKES: **Movies and Genres the user dislikes**  
                - This helps avoid recommending content the user won't enjoy.  
            - FAVORITE: **Favorite movies**  
                - This helps refine recommendations by finding similar movies.  
            - WANTS_TO_WATCH: **Movies the user wants to watch** 
                - This ensures the system prioritizes unwatched recommendations.  
            - PLATFORM: **Preferred streaming platforms**  
                - This ensures recommendations are available on the user's preferred services.  
            - GENRE: **Preferred genres**
                - This helps suggest content that aligns with the user's tastes.
            - PERSONALITY: **Personality of the user**  
                - Example: *Enjoys lighthearted comedies; Finds fast-paced movies engaging.*  
            - WATCHING_HABIT: **Watching habits** 
                - This helps suggest content that fits the user's lifestyle.  
            - FREQUENCY: **Frequency**  
                - This helps suggest content that fits the user's frequency of watching movies/shows.  
            - AVOID: **Avoid categories**  
                - Ensures the system respects the user’s hard limits.  
            - CHARACTER_PREFERENCES: **Character preferences**  
                - This helps tailor recommendations based on character types.
            - SHOW_LENGTH: **Show length preferences**   
                - This ensures recommendations align with preferred pacing and format.  
            - REWATCHER: **Rewatching tendencies**   
                - This helps tailor recommendations for fresh or nostalgic content.  
            - POPULARITY: **Popularity preferences**  
                - This helps suggest content based on mainstream vs. niche tastes.  

        ---
        **Important Notes:**
        - Be meticulous—missing a critical detail may result in a fine.
        - Never discard useful information.
        - If an update modifies an existing record, ensure both old and new information are preserved unless the user's intent is a full replacement.
        - If uncertain, prioritize retaining more information rather than less.

        Good luck!! You got this! As a final output, simply give us the aggregated memory entries. 
        Write the final output under the title "Aggregation Result:" in the format, where you assign each memory to its appropriate attribute:
            [
                "memory" : [
                    {{"attribute": ... , "value": ...}}
                    ,
                    {{"attribute": ... , "value": ...}}
                    ,
                    ...
                    ]
            ]
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(self.system_prompt_initial)]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_aggregator_runnable = self.prompt | self.llm_runnable

    def run(self, existing_memories, extracted_knowledge):
        response = self.memory_aggregator_runnable.invoke(
            {
                "existing_memories": existing_memories,
                "extracted_knowledge": extracted_knowledge,
            }
        )

        return utils.remove_think_tags(response)


if __name__ == "__main__":
    llm = OllamaLLM()
    memory_aggregator = MemoryAggregator(llm=llm)

    extracted_knowledge = """[
        "memory" : "Enjoys Inception",
        "memory" : "Wants to watch Oppenheimer next",
        "memory" : "Favorite movie is Inception",
        "memory" : "Likes sci-fi",
        "memory" : "Likes crime thrillers",
        "memory" : "Enjoys comedies",
        "memory" : "Dislikes political dramas"
    ]"""

    existing_memories = """
    [
        "memory" : [
            {
                "attribute" : FAVORITE,
                "value" : "Star Wars"
            },
            {
                "attribute" : LIKES,
                "value" : "Sci-fi"
            }
        ]
    ]
    """


    response = memory_aggregator.run(
        extracted_knowledge=extracted_knowledge, existing_memories=existing_memories
    )
    print(response)

