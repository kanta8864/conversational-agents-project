from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
from .memory_sentinel import call_memory_sentinel
from .memory_extractor import call_memory_extractor

###### SET UP THE GRAPH ######
class AgentState(TypedDict):
    # The list of previous messages in the conversation
    messages: Sequence[BaseMessage]
    # The long-term memories to remember
    memories: Sequence[str]
    # Whether the information is relevant
    contains_information: str

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
    # We know the last message involves at least one tool call
    last_message = messages[-1]

    # We loop through all tool calls and append the message to our message log
    for tool_call in last_message.additional_kwargs["tool_calls"]:
        action = ToolInvocation(
            tool=tool_call["function"]["name"],
            tool_input=json.loads(tool_call["function"]["arguments"]),
            id=tool_call["id"],
        )

        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )

        # Add the function message to the list
        messages.append(function_message)
    return {"messages": messages}

# Initialize a new graph
graph = StateGraph(AgentState)

# Define the two "Nodes"" we will cycle between
graph.add_node("sentinel", call_memory_sentinel)
graph.add_node("memory_extractor", call_memory_extractor)
graph.add_node("memory_reviewer", call_memory_reviewer)
graph.add_node("memory_attributor", call_memory_attributor)
graph.add_node("attribute_reviwer", call_attribute_reviewer)
graph.add_node("action_assigner", call_action_assigner)
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
    should_continue,
    {
        "continue": "memory_attributor",
        "end": "memory_extractor",
    },
)
graph.add_conditional_edges(
    "memory_attributor",
    should_continue,
    {
        "continue": "attribute_reviwer",
        "end": END,
    },
)
graph.add_conditional_edges(
    "attribute_reviwer",
    should_continue,
    {
        "continue": "action_assigner",
        "end": "memory_attributor",
    },
)

graph.add_conditional_edges(
    "action_assigner",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# We now add Normal Edges that should always be called after another
graph.add_edge("action", END)

# We compile the entire workflow as a runnable
app = graph.compile()

