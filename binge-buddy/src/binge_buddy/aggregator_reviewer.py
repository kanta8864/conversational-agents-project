from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM


class AggregatorReviewer:
    def __init__(self, llm: OllamaLLM):
        """
        Initializes the MemorySentinel agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
        self.llm = llm

        # System prompt for the memory reviewer
        self.system_prompt_initial = """
        You are an expert memory reviewer tasked with ensuring the integrity of a user's long-term memory. Your role is to critically evaluate the **aggregated memory** produced by the system, ensuring it is:
        1. **Accurate** - It must correctly reflect the existing memories and new memory without distortion.
        2. **Complete** - It should capture all relevant details without omitting important information.
        3. **Consistent** - It must align with the user's existing long-term memory and should not contradict known facts.
        4. **Non-hallucinatory** - It should not introduce details that were never mentioned in either the existing or new memory.

        ---

        ### **Review Process**
        To verify the correctness of the **aggregated memory**, follow these steps:

        #### **1. Check for Hallucinations**  
        - Ensure that no **new, fabricated, or assumed information** has been added.  
        - Every detail in the aggregated memory must be **traceable** to either the existing memories or the new memory.  

        #### **2. Check for Completeness**  
        - Verify that all **key details** from the **existing memories** and the **new memory** are present.  
        - If any critical piece of information is missing, flag it.  

        #### **3. Check for Incorrect Aggregation**  
        - Ensure that previously stored knowledge has **not been misrepresented** or **incorrectly altered**.  
        - If an existing memory has been **overwritten incorrectly** or the meaning has been subtly changed, flag it.  

        ---

        ### **Existing Memories**
        {existing_memories}

        ### **Newly Extracted Memory**
        {extracted_knowledge}

        ### **Aggregated Memory (Final Output)**
        {aggregated_memory}

        ---

        ### **Output Format**
        If the **aggregated memory** is **correct**, respond with:
        APPROVED

        If the **aggregated memory** is incorrect, respond with:
        REJECTED
        **Reason:** Clearly explain why the aggregated memory is incorrect, specifying whether it introduces hallucinations, omits crucial details, or alters existing knowledge incorrectly.
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(self.system_prompt_initial)]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.aggregator_reviewer_runnable = self.prompt | self.llm_runnable

    def run(self, existing_memories, extracted_knowledge, aggregated_memory):

        response = self.aggregator_reviewer_runnable.invoke(
            {
                "existing_memories": existing_memories,
                "extracted_knowledge": extracted_knowledge,
                "aggregated_memory": aggregated_memory,
            }
        )

        return utils.remove_think_tags(response)


if __name__ == "__main__":
    llm = OllamaLLM()
    message_log = MessageLog(session_id="12345", user_id="user123")
    memory_reviewer = AggregatorReviewer(llm=llm)

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

    extracted_knowledge = """[
        "memory" : "Enjoys Inception",
        "memory" : "Wants to watch Oppenheimer next",
        "memory" : "Favorite movie is Inception",
        "memory" : "Likes sci-fi",
        "memory" : "Likes crime thrillers",
        "memory" : "Enjoys comedies",
        "memory" : "Dislikes political dramas"
    ]"""

    aggregated_memory = """
    [
            "memory": [
                {
                    "attribute": "FAVORITE",
                    "value": "The Dark Knight"
                },
                {
                    "attribute": "FAVORITE",
                    "value": "Inception"
                },
                {
                    "attribute": "LIKES",
                    "value": "Dramas"
                },
                {
                    "attribute": "LIKES",
                    "value": "Thrillers"
                },
                {
                    "attribute": "LIKES",
                    "value": "Sci-fi"
                },
                {
                    "attribute": "LIKES",
                    "value": "Crime Thrillers"
                },
                {
                    "attribute": "LIKES",
                    "value": "Comedies"
                },
                {
                    "attribute": "WANTS_TO_WATCH",
                    "value": "Oppenheimer"
                },
                {
                    "attribute": "DISLIKES",
                    "value": "Political Dramas"
                }
            ]
    ]"""

    response = memory_reviewer.run(
        existing_memories=existing_memories,
        extracted_knowledge=extracted_knowledge,
        aggregated_memory=aggregated_memory,
    )
    print(response)
