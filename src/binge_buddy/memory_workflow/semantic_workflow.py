# from langgraph.graph import END, StateGraph
from binge_buddy.agent_state.states import AgentState, SemanticAgentState
from binge_buddy.agents.aggregator_reviewer import AggregatorReviewer
from binge_buddy.agents.extractor_reviewer import ExtractorReviewer
from binge_buddy.agents.memory_extractor import MemoryExtractor
from binge_buddy.agents.memory_sentinel import MemorySentinel
from binge_buddy.memory_workflow.multi_agent_workflow import MultiAgentWorkflow
from binge_buddy.message import UserMessage
from binge_buddy.ollama import OllamaLLM
from binge_buddy.state_graph import CustomStateGraph


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
        self.state_graph: CustomStateGraph = CustomStateGraph(SemanticAgentState)

        # Add Nodes
        self.state_graph.add_node("sentinel", memory_sentinel.process)
        self.state_graph.add_node("memory_extractor", memory_extractor.process)
        self.state_graph.add_node("memory_reviewer", extractor_reviewer.process)
        self.state_graph.add_node("memory_aggregator", memory_aggregator.process)
        self.state_graph.add_node("aggregator_reviewer", aggregator_reviewer.process)
        self.state_graph.add_node("action", print)  # Final action node

        # Set the Starting Edge
        self.state_graph.set_entry_point("sentinel")

        # Add Conditional Edges (Updated to use attributes instead of dictionary keys)
        self.state_graph.add_conditional_edges(
            "sentinel",
            lambda state: "yes" if state.contains_information else "no",
            {
                "yes": "memory_extractor",
                "no": None,  # END is typically None in a custom graph
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_extractor",
            lambda state: "continue" if state.extracted_memories else "end",
            {
                "continue": "memory_reviewer",
                "end": None,
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_reviewer",
            lambda state: ("continue" if not state.needs_repair else "repair"),
            {
                "continue": "memory_aggregator",
                "repair": "memory_extractor",
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_aggregator",
            lambda state: "continue" if state.aggregated_memories else "end",
            {
                "continue": "aggregator_reviewer",
                "end": None,
            },
        )

        self.state_graph.add_conditional_edges(
            "aggregator_reviewer",
            lambda state: ("continue" if not state.needs_repair else "repair"),
            {
                "continue": "action",
                "repair": "memory_aggregator",
            },
        )

        # Add Normal Edges
        self.state_graph.add_edge("action", None)

    def run(self, initial_state: AgentState):
        self.state_graph.run(initial_state)

    def run_with_logging(self, initial_state: AgentState):
        self.state_graph.run_with_logging(initial_state)

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

    content = """
    I am Kanta, I love furries and a lot of kitty cats. I am interested in watching many cat related
    movies but I ocassionally enjoy some doggy content too.
    """
    message = UserMessage(content=content, user_id="12", session_id="123")

    state = SemanticAgentState(
        user_id="userID",
        existing_memories=[],
        current_user_message=message,
    )

    # semantic_workflow.run(state)
    # Run with logging
    semantic_workflow.run_with_logging(state)
