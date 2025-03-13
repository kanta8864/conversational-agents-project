import json
from typing import Dict, Optional, Sequence, TypedDict

from langchain.tools import StructuredTool
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from binge_buddy import utils
from binge_buddy.aggregator_reviewer import AggregatorReviewer
from binge_buddy.enums import Action, Attribute
from binge_buddy.extractor_reviewer import ExtractorReviewer
from binge_buddy.memory_aggregator import MemoryAggregator
from binge_buddy.memory_db import MemoryDB
from binge_buddy.memory_extractor import MemoryExtractor
from binge_buddy.memory_sentinel import MemorySentinel
from binge_buddy.message_log import MessageLog

from .ollama import OllamaLLM

llm = OllamaLLM()
# db = MemoryDB()


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


def modify_knowledge(
    knowledge: str,
    attribute: str,
    action: str,
    knowledge_old: str = "",
) -> dict:
    print("Modifying Knowledge: ", knowledge, knowledge_old, attribute, action)
    # retrieve current knowledge base
    # todo: replace with database retrieval
    memory = {}
    if attribute in memory and action == "update":
        # aggregate old and new knowledge and update memory
        # todo: change this temporary aggreation
        memory[attribute] = memory[attribute].replace(
            knowledge_old, f"{knowledge_old}; {knowledge}"
        )

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
    memories: Dict[str, str]
    # Whether the information is relevant
    contains_information: str

    extracted_knowledge: str

    aggregated_memory: str


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
    message_log = MessageLog(user_id="user", session_id="session")
    memory_sentinel = MemorySentinel(llm=llm, message_log=message_log)
    response = utils.remove_think_tags(
        memory_sentinel.memory_sentinel_runnable.invoke([last_message])
    )
    return {"contains_information": "TRUE" in response and "yes" or "no"}


def call_memory_extractor(state):
    # memories = db.get_collection("memories").find()
    # for document in memories:
    #     print(document)
    messages = state["messages"]
    last_message = messages[-1]
    message_log = MessageLog(user_id="user", session_id="session")
    memory_extractor = MemoryExtractor(llm=llm, message_log=message_log)
    response = utils.remove_think_tags(memory_extractor.memory_extractor_runnable.invoke(
        {"messages": [last_message]}
    ))
    response = response.split("Memory Extractor Result:", 1)[-1].strip()
    return {"extracted_knowledge": f"{response}"}


def call_extractor_reviewer(state):
    messages = state["messages"]
    last_message = messages[-1]
    extracted_knowledge = state["extracted_knowledge"]
    message_log = MessageLog(user_id="user", session_id="session")
    memory_reviewer = ExtractorReviewer(llm=llm, message_log=message_log)
    response =utils.remove_think_tags( memory_reviewer.memory_reviewer_runnable.invoke(
        {"user_message": [last_message], "extracted_knowledge": extracted_knowledge}
    ))
    return {
        "extractor_valid": "valid" if "APPROVED" in response else f"invalid: {response}"
    }


def call_memory_aggregator(state):
    memories = state.get("memories", [])
    extracted_knowledge = state["extracted_knowledge"]
    memory_aggregator = MemoryAggregator(llm=llm)
    response = utils.remove_think_tags(memory_aggregator.run(
        existing_memories=memories, extracted_knowledge=extracted_knowledge
    ))
    response = response.split("Aggregation Result:", 1)[-1].strip()
    return {"aggregated_memory": f"{response}"}


def call_aggregator_reviewer(state):
    memories = state.get("memories", [])
    extracted_knowledge = state["extracted_knowledge"]
    aggregated_memory = state["aggregated_memory"]
    aggregator_reviewer = AggregatorReviewer(llm=llm)
    response = utils.remove_think_tags(aggregator_reviewer.run(
        existing_memories=memories,
        extracted_knowledge=extracted_knowledge,
        aggregated_memory=aggregated_memory,
    ))
    response = response.split("Aggregation Result:", 1)[-1].strip()
    return {
        "aggregator_valid": (
            "valid" if "APPROVED" in response else f"invalid: {response}"
        )
    }


# Initialize a new graph
graph = StateGraph(AgentState)

# Define the "Nodes"" we will cycle between
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
    lambda state: (
        "continue" if bool(state["extracted_knowledge"]) else "end"
    ),  # Ensure to return a string key
    {
        "continue": "memory_reviewer",
        "end": END,
    },
)

graph.add_conditional_edges(
    "memory_reviewer",
    lambda state: (
        "memory_aggregator"
        if "APPROVED" in state["extractor_valid"]
        else "memory_extractor"
    ),
)

graph.add_conditional_edges(
    "memory_aggregator",
    lambda state: (
        "continue" if bool(state["aggregated_memory"]) else "end"
    ),  # Ensure to return a string key
    {
        "continue": "aggregator_reviewer",
        "end": END,
    },
)

graph.add_conditional_edges(
    "aggregator_reviewer",
    lambda state: (
        "action"
        if "APPROVED" in state["aggregator_valid"]
        else "memory_aggregator"
    ),
)

# We now add Normal Edges that should always be called after another
graph.add_edge("action", END)

# We compile the entire workflow as a runnable
app = graph.compile()
