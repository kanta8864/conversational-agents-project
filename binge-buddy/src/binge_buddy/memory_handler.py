import json
from typing import TypedDict, Sequence
from binge_buddy.aggregator_reviewer import AggregatorReviewer
from binge_buddy.enums import Action, Category
from binge_buddy.extractor_reviewer import ExtractorReviewer
from binge_buddy.memory_aggregator import MemoryAggregator
from binge_buddy.memory_extractor import MemoryExtractor
from binge_buddy.memory_sentinel import MemorySentinel
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from .ollama import OllamaLLM
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


llm = OllamaLLM()

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
    category: Category = Field(
        ..., description="Category that this knowledge belongs to"
    )
    action: Action = Field(
        ...,
        description="Whether this knowledge is adding a new record, or updating an existing record with aggregated information",
    )


def modify_knowledge(
    knowledge: str,
    category: str,
    action: str,
    knowledge_old: str = "",
) -> dict:
    print("Modifying Knowledge: ", knowledge, knowledge_old, category, action)
    # retrieve current knowledge base
    # todo: replace with database retrieval 
    memory = {}
    if category in memory and action == "update":
        # aggregate old and new knowledge and update memory 
        # todo: change this temporary aggreation
        memory[category] = memory[category].replace(knowledge_old, f"{knowledge_old}; {knowledge}")
    
    return "Modified Knowledge"


tool_modify_knowledge = StructuredTool.from_function(
    func=modify_knowledge,
    name="Knowledge_Modifier",
    description="Add or update memory",
    args_schema=AddKnowledge,
)

# Set up the agent's tools
agent_tools = [tool_modify_knowledge]

# tool_executor = lp.ToolExecutor(agent_tools)


###### SET UP THE GRAPH ######
class AgentState(TypedDict):
    # The list of previous messages in the conversation
    messages: Sequence[BaseMessage]
    # The long-term memories to remember
    memories: Sequence[str]
    # Whether the information is relevant
    contains_information: str

    new_memories: Sequence[str]

# Define the function that determines whether to continue or not
def should_continue(state):
    last_message = state["messages"][-1]
    # If there are no tool calls, then we finish
    if "tool_calls" not in last_message.additional_kwargs:
        return "end"
    # Otherwise, we continue
    else:
        return "continue"


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]

    if "tool_calls" in last_message.additional_kwargs:
        for tool_call in last_message.additional_kwargs["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_input = json.loads(tool_call["function"]["arguments"])

            tool = next(t for t in agent_tools if t.name == tool_name)
            response = tool.invoke(tool_input)

            function_message = ToolMessage(
                content=str(response), name=tool_name, tool_call_id=tool_call["id"]
            )

            messages.append(function_message)
    return {"messages": messages}


def call_memory_sentinel(state):
    messages = state["messages"]
    last_message = messages[-1]
    memory_sentinel = MemorySentinel(llm=llm)
    response = memory_sentinel.memory_sentinel_runnable.invoke(last_message)
    return {"contains_information": "TRUE" in response.content and "yes" or "no"}

def call_memory_extractor(state):
    messages = state["messages"]
    last_message = messages[-1]
    memories = state["memories"]
    memory_extractor = MemoryExtractor(llm=llm)
    response = memory_extractor.memory_extractor_runnable(
        {"messages": last_message, "memories": memories}
    )
    print(f"call_memory_extractor output: {response.content}")
    return {"extracted_knowledge": f"{response.content}"}

def call_extractor_reviewer(state):
    messages = state["messages"]
    last_message = messages[-1]
    new_memories = state["new_memories"]
    memory_reviewer = ExtractorReviewer(llm=llm)
    response = memory_reviewer.memory_reviewer_runnable.invoke({
        "user_message": last_message,  
        "new_memory": new_memories   
    })
    return {"extractor_valid": "valid" if "APPROVED" in response.content else f"invalid: {response.content}"}

def call_memory_aggregator(state):
    memories = state["memories"]
    new_memories = state["new_memories"]
    memory_aggregator = MemoryAggregator(llm=llm)
    response = memory_aggregator.memory_aggregator_runnable.invoke({
        "existing_memories": memories,  
        "new_memories": new_memories   
    })
    return {"aggregated_memory": f"{response.content}"}

def call_aggregator_reviewer(state):
    messages = state["messages"]
    last_message = messages[-1]
    new_memories = state["new_memories"]
    aggregator_reviewer = AggregatorReviewer(llm=llm)
    response = aggregator_reviewer.aggregator_reviewer_runnable.invoke({
        "user_message": last_message,  
        "new_memory": new_memories   
    })
    return {"aggregator_valid": "valid" if "APPROVED" in response.content else f"invalid: {response.content}"}


# Initialize a new graph
graph = StateGraph(AgentState)

# Define the two "Nodes"" we will cycle between
graph.add_node("sentinel", call_memory_sentinel)
graph.add_node("memory_extractor", call_memory_extractor)
graph.add_node("memory_reviewer", call_extractor_reviewer)
graph.add_node("memory_aggregator", call_memory_aggregator)
graph.add_node("aggregator_reviewer", call_aggregator_reviewer)
graph.add_node("action", call_tool)

# Define all our Edges

# Set the Starting Edge
graph.set_entry_point("sentinel")

# We now add Conditional Edges
graph.add_conditional_edges(
    "sentinel",
    lambda x: x["contains_information"],
    {
        "yes": "memory_extractor",
        "no": END,
    },
)
graph.add_conditional_edges(
    "memory_extractor",
    should_continue,
    {
        "continue": "memory_reviewer",
        "end": END,
    },
)
graph.add_conditional_edges(
    "memory_reviewer",
    lambda x: x["extractor_valid"],
    {
        "valid": "memory_aggregator",
    }.get, 
    "memory_extractor",  
)
graph.add_conditional_edges(
    "memory_aggregator",
    should_continue,
    {
        "continue": "aggregator_reviewer",
        "end": END,
    },
)
graph.add_conditional_edges(
    "aggregator_reviewer",
    lambda x: x["extractor_valid"],
    {
        "valid": "action",
    }.get, 
    "memory_aggregator",  
)

# We now add Normal Edges that should always be called after another
graph.add_edge("action", END)

# We compile the entire workflow as a runnable
app = graph.compile()


