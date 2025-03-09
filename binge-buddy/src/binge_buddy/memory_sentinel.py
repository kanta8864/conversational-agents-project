from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage
from binge_buddy.ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda



class MemorySentinel:
    def __init__(self, llm: OllamaLLM):
        """
        Initializes the MemorySentinel agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
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
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_sentinel_runnable = self.prompt | self.llm_runnable


if __name__ == "__main__":
    # Initialize the message log and LLM (for now, using a mock LLM)
    llm = OllamaLLM() 
    memory_sentinel = MemorySentinel(llm=llm)

    # Test message
    current_message = "I love watching sci-fi movies like The Matrix!"

    response = memory_sentinel.memory_sentinel_runnable.invoke({
        "messages": [HumanMessage(content=current_message)],   
    })

    print("Store this information?", response)

