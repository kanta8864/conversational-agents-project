from binge_buddy.agent_state.agent_state import AgentState
from binge_buddy.agent_state.semantic_agent_state import SemanticAgentState
from binge_buddy.agents.aggregator_reviewer import AggregatorReviewer
from binge_buddy.agents.extractor_reviewer import ExtractorReviewer
from binge_buddy.agents.memory_extractor import MemoryExtractor
from binge_buddy.agents.memory_sentinel import MemorySentinel
from binge_buddy.memory_workflow.multi_agent_workflow import MultiAgentWorkflow
from binge_buddy.ollama import OllamaLLM
from langgraph.graph import END, StateGraph

class SemanticWorkflow(MultiAgentWorkflow):
    def __init__(self):
        super().__init__()
    
    def run(self, initial_state: AgentState):
        # Initialize a new graph
        self.state_graph = StateGraph(initial_state)

        # todo: change this to use global llm
        llm = OllamaLLM()
        memory_sentinel = MemorySentinel(llm)
        memory_extractor = MemoryExtractor(llm)
        extractor_reviewer = ExtractorReviewer(llm)
        memory_aggregator = MemoryExtractor(llm)
        aggregator_reviewer = AggregatorReviewer(llm)

        self.state_graph.add_node("sentinel", lambda: memory_sentinel.process(initial_state))
        self.state_graph.add_node("memory_extractor", lambda:memory_extractor.process(initial_state))
        self.state_graph.add_node("memory_reviewer", lambda:extractor_reviewer.process(initial_state))
        self.state_graph.add_node("memory_aggregator",lambda: memory_aggregator.process(initial_state))
        self.state_graph.add_node("aggregator_reviewer", lambda:aggregator_reviewer.process(initial_state))

        # Call super().call_tool() directly if it's callable
        self.state_graph.add_node("action", lambda:super().call_tool)


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
            ),  # Ensure to return a string key
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
        app = self.state_graph.compile()

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
    handler = SemanticWorkflow()
    state = SemanticAgentState(user_id="userID", memories=[], current_user_message="hi")
    app = handler.run(state)

    # Print all the nodes in the graph
    handler.print_nodes()
  