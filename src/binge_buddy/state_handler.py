# import json
# from typing import Dict, Optional, Sequence, TypedDict, List

# from langchain.tools import StructuredTool
# from langchain_core.messages import BaseMessage, ToolMessage
# from langgraph.graph import END, StateGraph
# from pydantic import BaseModel, Field

# from binge_buddy import utils
# from binge_buddy.agents.aggregator_reviewer import AggregatorReviewer
# from binge_buddy.enums import Action, Attribute
# from binge_buddy.agents.extractor_reviewer import ExtractorReviewer
# from binge_buddy.agents.memory_aggregator import MemoryAggregator
# from binge_buddy.memory_db import MemoryDB
# from binge_buddy.agents.memory_extractor import MemoryExtractor
# from binge_buddy.agents.memory_sentinel import MemorySentinel
# from binge_buddy.message_log import MessageLog
# from binge_buddy.ollama import OllamaLLM

# llm = OllamaLLM()
# # db = MemoryDB()


# # defines argument type
# class AddKnowledge(BaseModel):
#     knowledge: str = Field(
#         ...,
#         description="Condensed bit of knowledge to be saved for future reference in the format: [person(s) this is relevant to] [fact to store] (e.g. Husband doesn't like sci-fi; I love horror movies; etc)",
#     )
#     knowledge_old: Optional[str] = Field(
#         None,
#         description="If updating, the complete, exact phrase of the existing knowledge to modify",
#     )
#     attribute: Attribute = Field(
#         ...,
#         description="Attribute that this knowledge belongs to"
#     )
#     action: Action = Field(
#         ...,
#         description="Whether this knowledge is adding a new record, or updating an existing record with aggregated information",
#     )

# # Define the state of the agent
# class AgentState(TypedDict):
#     # The list of previous messages in the conversation
#     messages: Sequence[BaseMessage]
#     # The long-term memories to remember
#     memories: Dict[str, str]
#     # Whether the information is relevant
#     contains_information: str
#     # The extracted knowledge from the user's message
#     extracted_knowledge: str
#     # The aggregations of the extracted knowledge assigned to an attribute
#     aggregated_memory: str

# def modify_knowledge(
#     knowledge: str,
#     attribute: str,
#     action: str,
#     knowledge_old: str = "",
# ) -> dict:
#     print("Modifying Knowledge: ", knowledge, knowledge_old, attribute, action)
#     # retrieve current knowledge base
#     # todo: replace with database retrieval
#     memory = {}
#     if attribute in memory and action == "update":
#         # aggregate old and new knowledge and update memory
#         # todo: change this temporary aggreation
#         memory[attribute] = memory[attribute].replace(
#             knowledge_old, f"{knowledge_old}; {knowledge}"
#         )

#     return {"memories": memory, "response": "Memory updated successfully"}


# tool_modify_knowledge = StructuredTool.from_function(
#     func=modify_knowledge,
#     name="Knowledge_Modifier",
#     description="Add or update memory",
#     args_schema=AddKnowledge,
# )

# # Set up the agent's tools
# agent_tools = [tool_modify_knowledge]

# #region Graph Node Functions
# # Define the function that determines whether to continue or not
# def should_continue(state):
#     last_message = state["messages"][-1]
#     # If there are no tool calls, then we finish
#     if "tool_calls" not in last_message.additional_kwargs:
#         return "end"
#     # Otherwise, we continue
#     else:
#         return "continue"



# def call_memory_sentinel(state):
#     messages = state["messages"]
#     last_message = messages[-1]
#     message_log = MessageLog(user_id="user", session_id="session")
#     memory_sentinel = MemorySentinel(llm=llm, message_log=message_log)
#     response = utils.remove_think_tags(
#         memory_sentinel.memory_sentinel_runnable.invoke([last_message])
#     )
#     return {"contains_information": "TRUE" in response and "yes" or "no"}


# def call_memory_extractor(state):
#     # memories = db.get_collection("memories").find()
#     # for document in memories:
#     #     print(document)
#     messages = state["messages"]
#     last_message = messages[-1]
#     message_log = MessageLog(user_id="user", session_id="session")
#     memory_extractor = MemoryExtractor(llm=llm, message_log=message_log)
#     response = utils.remove_think_tags(memory_extractor.memory_extractor_runnable.invoke(
#         {"messages": [last_message]}
#     ))
#     response = response.split("Memory Extractor Result:", 1)[-1].strip()
#     return {"extracted_knowledge": f"{response}"}


# def call_extractor_reviewer(state):
#     messages = state["messages"]
#     last_message = messages[-1]
#     extracted_knowledge = state["extracted_knowledge"]
#     message_log = MessageLog(user_id="user", session_id="session")
#     memory_reviewer = ExtractorReviewer(llm=llm, message_log=message_log)
#     response =utils.remove_think_tags( memory_reviewer.memory_reviewer_runnable.invoke(
#         {"user_message": [last_message], "extracted_knowledge": extracted_knowledge}
#     ))
#     return {
#         "extractor_valid": "valid" if "APPROVED" in response else f"invalid: {response}"
#     }


# def call_memory_aggregator(state):
#     memories = state.get("memories", [])
#     extracted_knowledge = state["extracted_knowledge"]
#     memory_aggregator = MemoryAggregator(llm=llm)
#     response = utils.remove_think_tags(memory_aggregator.run(
#         existing_memories=memories, extracted_knowledge=extracted_knowledge
#     ))
#     response = response.split("Aggregation Result:", 1)[-1].strip()
#     return {"aggregated_memory": f"{response}"}


# def call_aggregator_reviewer(state):
#     memories = state.get("memories", [])
#     extracted_knowledge = state["extracted_knowledge"]
#     aggregated_memory = state["aggregated_memory"]
#     aggregator_reviewer = AggregatorReviewer(llm=llm)
#     response = utils.remove_think_tags(aggregator_reviewer.run(
#         existing_memories=memories,
#         extracted_knowledge=extracted_knowledge,
#         aggregated_memory=aggregated_memory,
#     ))
#     response = response.split("Aggregation Result:", 1)[-1].strip()
#     return {
#         "aggregator_valid": (
#             "valid" if "APPROVED" in response else f"invalid: {response}"
#         )
#     }
# #endregion

# class GraphHandler():
#     '''
#     This class is responsible for setting up the graph assigning nodes.
#     '''
#     def __init__(self):
#         self.state = AgentState
#         self.graph = None  # Initialize graph as None

#     def run(self):
#         # Initialize a new graph
#         self.graph = StateGraph(self.state)

#         # Define the "Nodes"" we will cycle between
#         self.graph.add_node("sentinel", call_memory_sentinel)
#         self.graph.add_node("memory_extractor", call_memory_extractor)
#         self.graph.add_node("memory_reviewer", call_extractor_reviewer)
#         self.graph.add_node("memory_aggregator", call_memory_aggregator)
#         self.graph.add_node("aggregator_reviewer", call_aggregator_reviewer)
#         self.graph.add_node("action", call_tool)

#         # Define all our Edges

#         # Set the Starting Edge
#         self.graph.set_entry_point("sentinel")

#         # We now add Conditional Edges
#         self.graph.add_conditional_edges(
#             "sentinel",
#             lambda x: x["contains_information"],
#             {
#                 "yes": "memory_extractor",
#                 "no": END,
#             },
#         )

#         self.graph.add_conditional_edges(
#             "memory_extractor",
#             lambda state: (
#                 "continue" if bool(state["extracted_knowledge"]) else "end"
#             ),  # Ensure to return a string key
#             {
#                 "continue": "memory_reviewer",
#                 "end": END,
#             },
#         )

#         self.graph.add_conditional_edges(
#             "memory_reviewer",
#             lambda state: (
#                 "memory_aggregator"
#                 if "APPROVED" in state["extractor_valid"]
#                 else "memory_extractor"
#             ),
#         )

#         self.graph.add_conditional_edges(
#             "memory_aggregator",
#             lambda state: (
#                 "continue" if bool(state["aggregated_memory"]) else "end"
#             ),  # Ensure to return a string key
#             {
#                 "continue": "aggregator_reviewer",
#                 "end": END,
#             },
#         )

#         self.graph.add_conditional_edges(
#             "aggregator_reviewer",
#             lambda state: (
#                 "action"
#                 if "APPROVED" in state["aggregator_valid"]
#                 else "memory_aggregator"
#             ),
#         )

#         # We now add Normal Edges that should always be called after another
#         self.graph.add_edge("action", END)

#         # We compile the entire workflow as a runnable
#         app = self.graph.compile()
#         return app

#     def print_nodes(self):
#         if self.graph is None:
#             print("Graph has not been initialized yet.")
#             return

#         nodes = self.graph.nodes
#         print("Nodes in the graph:")
#         for node in nodes:
#             print(f"- {node}")

# if __name__ == "__main__":
#     handler = GraphHandler()
#     app = handler.run()

#     # Print all the nodes in the graph
#     handler.print_nodes()


