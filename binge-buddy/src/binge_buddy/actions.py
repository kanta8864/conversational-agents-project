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

tool_executor = ToolExecutor(agent_tools)