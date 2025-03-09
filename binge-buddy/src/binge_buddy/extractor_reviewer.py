from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.message import Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM


class ExtractorReviewer:
    def __init__(self, llm: OllamaLLM, message_log: MessageLog):
        """
        Initializes the MemorySentinel agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        """
        self.llm = llm

        self.message_log = message_log

        # System prompt for the memory reviewer
        self.system_prompt_initial = """
        You are an expert memory reviewer.

        Your job is to ensure that the newly generated memory is:
        1. **Accurate** - It should correctly reflect the user's message.
        2. **Complete** - It must include all relevant information from the user's message.
        3. **Non-hallucinatory** - It should not introduce information that was never mentioned.

        ---

        ### **Review Process**
        1. **Check for Hallucinations**  
        - If the new memory contains any information that was not explicitly mentioned in the user's message or long-term memory, flag it.
        
        2. **Check for Completeness**  
        - Ensure all relevant details from the user's message are captured.

        ---

        ### **User's Message**
        {user_message}

        ### **Proposed New Memory**
        {extracted_knowledge}

        ---

        ### **Output Format**
        If the new memory is **correct**, respond with:
        ✅ APPROVED  

        If the new memory **needs fixing**, respond with:
        ❌ REJECTED  
        **Reason** Give reasons for flagging new aggreagated memory. 
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="user_message"),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_reviewer_runnable = self.prompt | self.llm_runnable

    def run(self, new_memory):

        current_message = self.message_log.get_last_message()
        if not current_message:
            return None

        return utils.remove_think_tags(
            self.memory_reviewer_runnable.invoke(
                {
                    "user_message": [current_message.to_langchain_message()],
                    "extracted_knowledge": new_memory,
                }
            )
        )


if __name__ == "__main__":
    llm = OllamaLLM()
    message_log = MessageLog(session_id="12345", user_id="user123")
    memory_reviewer = ExtractorReviewer(llm=llm, message_log=message_log)

    user_message = "I love action movies, but I don’t like slow-paced dramas."

    message_log.add_message(
        Message(
            content=user_message, role="user", session_id="12345", user_id="user123"
        )
    )

    new_memory = "Loves action movies and dislikes slow-paced dramas."

    response = memory_reviewer.run(new_memory)

    response = print(response)
