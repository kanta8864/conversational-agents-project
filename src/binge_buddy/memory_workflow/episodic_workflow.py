from binge_buddy.agent_state.agent_state import AgentState
from typing import Any
from binge_buddy.memory_workflow.multi_agent_workflow import MultiAgentWorkflow

class EpisodicWorkflow(MultiAgentWorkflow):
    def __init__(self):
        super().__init__()
    
    def run(self, initial_state: AgentState):
       return