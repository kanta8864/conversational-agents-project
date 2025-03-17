from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.agent_state.states import AgentState, AgentStateDict
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.memory import SemanticMemory
from binge_buddy.ollama import OllamaLLM


class ExtractorReviewer(BaseAgent):
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
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
            
            3. **Check if format is correct**
            - Ensure the format of the memory is correct. So in the form of a List of all the memories as a json array in the format 
                [
                    "memory" : ...,
                    "memory" : ...,
                    ...
                ]

            ---

            ### **User's Message**
            {user_message}

            ### **Proposed New Memory**
            {extracted_knowledge}

            ---

            ### **Output Format**
            If the new memory is **correct**, respond with:
            APPROVED  

            If the new memory **needs fixing**, respond with:
            REJECTED  
            **Reason** Give reasons for flagging new aggreagated memory. 
            """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="user_message"),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_reviewer_runnable = self.prompt | self.llm_runnable

    def process(self, state: AgentState) -> AgentState:

        response = utils.remove_think_tags(
            self.memory_reviewer_runnable.invoke(
                {
                    "user_message": [state.current_user_message.to_langchain_message()],
                    "extracted_knowledge": state.extracted_memories,
                }
            )
        )

        # For now I am using a SemanticMemory but I think we should let the
        # state handle the creation of the Memory object.
        # A method like create_memory so SemanticAgentState returns an object of SemanticMemory
        # EpisodicAgentState returns an object of EpisodicMemory
        state.extracted_memories = [SemanticMemory(response)]
        return state
