from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from ollama import OllamaLLM

# Define the schema for memory review
class MemoryReviewRequest(BaseModel):
    user_message: str = Field(..., description="The latest message from the user.")
    new_memory: str = Field(..., description="The newly generated memory after aggregation.")
    existing_memories: List[str] = Field(..., description="The list of existing stored memories.")

# System prompt for the memory reviewer
system_prompt_review = """
You are an expert memory reviewer.

Your job is to ensure that the newly generated memory is:
1. **Accurate** - It should correctly reflect the user's message.
2. **Complete** - It must include all relevant information from the user's message.
3. **Consistent** - It should not contradict or incorrectly overwrite existing long-term memory.
4. **Non-hallucinatory** - It should not introduce information that was never mentioned.

---

### **Review Process**
1. **Check for Hallucinations**  
   - If the new memory contains any information that was not explicitly mentioned in the user's message or long-term memory, flag it.
  
2. **Check for Completeness**  
   - Ensure all relevant details from the user's message are captured.

3. **Check for Incorrect Aggregation**  
   - If an existing memory has been wrongly modified or overwritten, flag it.

---

### **Existing Memories**
{existing_memories}

### **User's Message**
{user_message}

### **Proposed New Memory**
{new_memory}


---

### **Output Format**
If the new memory is **correct**, respond with:
✅ APPROVED  

If the new memory **needs fixing**, respond with:
❌ REJECTED  
**Reason** Give reasons for flagging new aggreagated memory. 
"""

# Initialize LLM
llm = OllamaLLM()

# Define a function to review the memory
def review_memory(user_message: str, new_memory: str, existing_memories: List[str]) -> str:
    # Format the prompt
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_review)
        ]
    )

    # Create formatted prompt
    formatted_prompt = prompt_template.format_messages(
        user_message=user_message, new_memory=new_memory, existing_memories="\n".join(existing_memories)
    )

    # Get LLM response
    response = llm._call("\n".join([msg.content for msg in formatted_prompt]))

    return response


# Example Usage
user_message = "I love action movies, but I don’t like slow-paced dramas."
existing_memories = [
    "Likes sci-fi movies",
    "Enjoys fast-paced thrillers",
    "Does not enjoy horror movies"
]
new_memory = "Loves action movies and dislikes slow-paced dramas."

# Review the new memory
review_result = review_memory(user_message, new_memory, existing_memories)
print(review_result)

