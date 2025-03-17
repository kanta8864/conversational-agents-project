from abc import ABC, abstractmethod
import json
from typing import Optional

from binge_buddy.agent_state.agent_state import AgentState
from binge_buddy.enums import Action, Attribute
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph



class MultiAgentWorkflow(ABC):
    def __init__(self):
        self.state_graph: Optional[StateGraph] = None
        self.agent_tools = [self._initialize_modify_knowledge_tool()]

    @abstractmethod
    def run(self, initial_state: AgentState):
        pass

    def call_tool(self, state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        
        if "tool_calls" in last_message.additional_kwargs:
            for tool_call in last_message.additional_kwargs["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_input = json.loads(tool_call["function"]["arguments"])
                
                tool = next((t for t in self.agent_tools if t.name == tool_name), None)
                if tool:
                    response = tool.invoke(tool_input)
                    function_message = ToolMessage(
                        content=str(response),
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    )
                    messages.append(function_message)
        
        return {"messages": messages}
    
    def modify_knowledge(
        self, knowledge: str, attribute: str, action: str, knowledge_old: str = ""
    ) -> dict:
        print("Modifying Knowledge:", knowledge, knowledge_old, attribute, action)
        
        memory = {}  # Placeholder for database retrieval
        if attribute in memory and action == "update":
            memory[attribute] = memory[attribute].replace(
                knowledge_old, f"{knowledge_old}; {knowledge}"
            )
        
        return {"memories": memory, "response": "Memory updated successfully"}
    
    def _initialize_modify_knowledge_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.modify_knowledge,
            name="Knowledge_Modifier",
            description="Add or update memory",
            args_schema=AddKnowledge,
        )

# defines argument type
class AddKnowledge(BaseModel):
    knowledge: str = Field(
        ...,
        description="Condensed bit of knowledge to be saved for future reference in the format: [person(s) this is relevant to] [fact to store] (e.g. Husband doesn't like sci-fi; I love horror movies; etc)",
    )
    knowledge_old: Optional[str] = Field(
        None,
        description="If updating, the complete, exact phrase of the existing knowledge to modify",
    )
    attribute: Attribute = Field(
        ...,
        description="Attribute that this knowledge belongs to"
    )
    action: Action = Field(
        ...,
        description="Whether this knowledge is adding a new record, or updating an existing record with aggregated information",
    )