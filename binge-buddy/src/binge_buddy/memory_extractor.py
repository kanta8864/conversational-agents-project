from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage, HumanMessage
from ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda


system_prompt_initial = """
You are a supervisor managing a team of movie recommendation experts.

Your team's job is to build the perfect knowledge base about a user's movie preferences in order to provide highly personalized recommendations.

The knowledge base should ultimately consist of many discrete pieces of information that add up to a rich persona (e.g. Likes action movies; Dislikes horror; Favorite movie is Inception; Watches mostly on Netflix; Enjoys long TV series; Prefers witty and sarcastic characters etc).

Every time you receive a message, you will evaluate whether it contains any information worth recording in the knowledge base.

A message may contain multiple pieces of information that should be saved separately.

You are only interested in the following categories of information:

1. **Movies and Genres the user likes** (e.g. Likes sci-fi; Enjoyed Inception)  
   - This helps recommend movies the user is likely to enjoy.  

2. **Movies and Genres the user dislikes** (e.g. Dislikes horror; Didn't enjoy The Conjuring)  
   - This helps avoid recommending content the user won't enjoy.  

3. **Favorite movies** (e.g. Favorite movie is The Matrix)  
   - This helps refine recommendations by finding similar movies.  

4. **Movies the user wants to watch** (e.g. Wants to watch Oppenheimer)  
   - This ensures the system prioritizes unwatched recommendations.  

5. **Preferred streaming platforms** (e.g. Watches mostly on Netflix and Hulu)  
   - This ensures recommendations are available on the user's preferred services.  

6. **Personality of the user**  
   - Example: *Enjoys lighthearted comedies; Finds fast-paced movies engaging.*  

7. **Watching habits** (e.g. Prefers binge-watching TV shows over time)  
   - This helps suggest content that fits the user's lifestyle.  

8. **Frequency** (e.g. Watches movies/shows two days a week for about 2 hours each)  
   - This helps suggest content that fits the user's frequency of watching movies/shows.  

9. **Avoid categories** (e.g. Avoids anything with excessive gore; Doesn't like political dramas)  
   - Ensures the system respects the user’s hard limits.  

10. **Tone**  
   - tone of the user's messages to understand if the user is frustrated, happy etc.  

11. **Character preferences**  
   - Example: *Prefers witty and sarcastic characters; Enjoys dark and mysterious protagonists.*  

12. **Show length preferences** (e.g. Prefers miniseries; Enjoys long TV series with deep character development)  
   - This ensures recommendations align with preferred pacing and format.  

13. **Rewatching tendencies** (e.g. Frequently rewatches Friends; Rarely rewatch movies)  
   - This helps tailor recommendations for fresh or nostalgic content.  

14. **Popularity preferences** (e.g. Prefers cult classics over mainstream hits)  
   - This helps suggest content based on mainstream vs. niche tastes.  

When you receive a message, you perform a sequence of steps consisting of:

1. Analyze the most recent Human message for new information. You will see multiple messages for context, but we are only looking for new information in the most recent message.  
2. Compare this to the knowledge you already have.  
3. Determine if this is new knowledge, an update to existing knowledge, or if previously stored information needs to be corrected. In case it is an updat to existing knowledge, 
aggregate the old and new information, instead of overwriting and retain old knowledge too. 
   - Example: A movie the user previously disliked might now be a favorite, or they might have switched to a new streaming platform.  

Here are the existing bits of information we have about the user:

```
{memories}
```

Call the right tools to save the information, then respond with DONE. If you identiy multiple pieces of information, call everything at once. You only have one chance to call tools.

I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or change any incorrect information.

Take a deep breath, think step by step, and then analyze the following message:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = OllamaLLM()

# Wrap OllamaLLM with RunnableLambda to make it compatible
llm_runnable = RunnableLambda(lambda x: llm._call(x))

# Create the runnable sequence
memory_extractor_runnable = prompt | llm_runnable


# # Define a function to convert dictionaries to LangChain message objects
# def convert_to_messages(messages):
#     langchain_messages = []
#     for message in messages:
#         if message["role"] == "user":
#             langchain_messages.append(HumanMessage(content=message["content"]))
#         elif message["role"] == "system":
#             langchain_messages.append(SystemMessage(content=message["content"]))
#     return langchain_messages

# # Define a function to format the prompt
# def format_prompt(messages, memories):
#     # Convert the input messages to LangChain message objects
#     langchain_messages = convert_to_messages(messages)

#     # Format the system prompt with the memories
#     system_message = SystemMessage(content=system_prompt_initial.format(memories=memories))

#     # Format the messages into a ChatPromptValue object
#     chat_prompt_value = prompt.format_messages(
#         system_message=system_message,
#         messages=langchain_messages
#     )

#     # Convert the ChatPromptValue object into a single string
#     formatted_prompt = "\n".join([message.content for message in chat_prompt_value])
#     return formatted_prompt


# # Define a function to run the pipeline
# def run_memory_extractor(messages, memories):
#     # Format the prompt
#     formatted_prompt = format_prompt(messages, memories)

#     # Call the OllamaLLM
#     response = llm._call(formatted_prompt)
#     return response


# Example usage
if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "I really enjoyed Inception and I want to watch Oppenheimer next."},
    ]

    memories = """
    - Likes sci-fi movies
    - Favorite movie is The Matrix
    - Prefers Netflix for streaming
    """

    response = memory_extractor_runnable.invoke({
        "messages": messages,  # Your list of messages
        "memories": memories   # Your list of memories
    })
    print(response)