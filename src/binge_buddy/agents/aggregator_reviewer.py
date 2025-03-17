from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.agent_state.states import AgentState, SemanticAgentState
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.ollama import OllamaLLM


class AggregatorReviewer(BaseAgent):
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
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
            """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(self.system_prompt_initial)]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.aggregator_reviewer_runnable = self.prompt | self.llm_runnable

    def process(self, state: AgentState) -> AgentState:
        if not isinstance(state, SemanticAgentState):
            raise TypeError(f"Expected SemanticAgentState, got {type(state).__name__}")

        response = self.aggregator_reviewer_runnable.invoke(
            {
                "existing_memories": state.existing_memories,
                "extracted_knowledge": state.extracted_memories,
                "aggregated_memories": state.aggregated_memories,
            }
        )
        response = utils.remove_think_tags(response)

        state.needs_repair = response == "APPROVED"
        return state
