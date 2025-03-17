from langgraph.graph import END, StateGraph

from binge_buddy.agent_state.states import AgentState, SemanticAgentState
from binge_buddy.agents.aggregator_reviewer import AggregatorReviewer
from binge_buddy.agents.extractor_reviewer import ExtractorReviewer
from binge_buddy.agents.memory_extractor import MemoryExtractor
from binge_buddy.agents.memory_sentinel import MemorySentinel
from binge_buddy.memory_workflow.multi_agent_workflow import MultiAgentWorkflow
from binge_buddy.message import UserMessage
from binge_buddy.ollama import OllamaLLM


class SemanticWorkflow(MultiAgentWorkflow):
    def __init__(self):
        super().__init__()

        # todo: change this to use global llm
        llm = OllamaLLM()
        memory_sentinel = MemorySentinel(llm)
        memory_extractor = MemoryExtractor(llm)
        extractor_reviewer = ExtractorReviewer(llm)
        memory_aggregator = MemoryExtractor(llm)
        aggregator_reviewer = AggregatorReviewer(llm)

        # Initialize a new graph
        self.state_graph = StateGraph(SemanticAgentState)

        self.state_graph.add_node("sentinel", memory_sentinel.process)
        self.state_graph.add_node("memory_extractor", memory_extractor.process)
        self.state_graph.add_node("memory_reviewer", extractor_reviewer.process)
        self.state_graph.add_node("memory_aggregator", memory_aggregator.process)
        self.state_graph.add_node("aggregator_reviewer", aggregator_reviewer.process)

        # What is this doing, we don't use any tools directly, right?
        # Call super().call_tool() directly if it's callable
        self.state_graph.add_node("action", print)

        # Define all our Edges

        # Set the Starting Edge
        self.state_graph.set_entry_point("sentinel")

        # We now add Conditional Edges
        self.state_graph.add_conditional_edges(
            "sentinel",
            lambda x: x["contains_information"],
            {
                "yes": "memory_extractor",
                "no": END,
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_extractor",
            lambda state: (
                "continue" if bool(state["extracted_knowledge"]) else "end"
            ),  # Ensure to return a string key, (Shreyas: Why do you mean by a string key?)
            {
                "continue": "memory_reviewer",
                "end": END,
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_reviewer",
            lambda state: (
                "memory_aggregator"
                if "APPROVED" in state["extractor_valid"]
                else "memory_extractor"
            ),
        )

        self.state_graph.add_conditional_edges(
            "memory_aggregator",
            lambda state: (
                "continue" if bool(state["aggregated_memory"]) else "end"
            ),  # Ensure to return a string key
            {
                "continue": "aggregator_reviewer",
                "end": END,
            },
        )

        self.state_graph.add_conditional_edges(
            "aggregator_reviewer",
            lambda state: (
                "action"
                if "APPROVED" in state["aggregator_valid"]
                else "memory_aggregator"
            ),
        )

        # We now add Normal Edges that should always be called after another
        self.state_graph.add_edge("action", END)

        # We compile the entire workflow as a runnable
        self.state_graph_runnable = self.state_graph.compile()

    def run(self, initial_state: AgentState):

        self.state_graph_runnable.invoke(initial_state.as_dict())
        print("hey")

        # for output in app.with_config({"run_name": "Memory"}).stream(initial_state):
        #     print(output)
        #     # Output from the graph nodes (app stream processing)
        #     for key, value in output.items():
        #         print(f"Output from node '{key}':")
        #         print("---")
        #         print(value)
        #     print("\n---\n")

    def print_nodes(self):
        if self.state_graph is None:
            print("Graph has not been initialized yet.")
            return

        nodes = self.state_graph.nodes
        print("Nodes in the graph:")
        for node in nodes:
            print(f"- {node}")


if __name__ == "__main__":
    semantic_workflow = SemanticWorkflow()
    message = UserMessage(content="hi", user_id="12", session_id="123")

    state = SemanticAgentState(
        user_id="userID",
        existing_memories=[],
        current_user_message=message,
    )

    # Print all the nodes in the graph
    semantic_workflow.run(state)

