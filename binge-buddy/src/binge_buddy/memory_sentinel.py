# from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage, HumanMessage
from ollama import OllamaLLM

system_prompt_initial = """
Your job is to assess a brief chat history in order to determine if the conversation contains any details about a user's watching habits regarding streaming content. 

You are part of a team building a knowledge base regarding a users' watching habits to assist in highly customized streamming content recommendation.

You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.

You are only interested in the following categories of information:

1. The user's likes (e.g., "I love action movies!")
2. The user's dislikes (e.g., "I hate horror movies!")
3. The user's favorite shows or movies (e.g., "My favorite show is Game of Thrones!")
4. The user's desire to watch a show or movie (e.g., "I really want to watch the new Marvel movie!")
5. The user's preferred streaming platform (e.g., "I watch everything on Netflix!")
6. The user's preferred genre (e.g., "I love romantic comedies!")
7. The user's disinterest in a show or movie (e.g., "I'm not interested in watching that new series.")
8. The user's personality traits that may influence their watching habits (e.g., "I'm a huge fan of sci-fi because I love imagining the future.")
9. The user's watching habits (e.g., "I watch a movie every night before bed.")
10. The frequency of the user's watching habits (e.g., "I watch TV shows every weekend.")
11. The user's avoidance of certain types of content (e.g., "I avoid watching horror movies because they scare me.")
12. The user's preferred tone of content (e.g., "I prefer light-hearted comedies over dark dramas.")
13. The user's preference for certain types of characters (e.g., "I love shows with strong female leads.")
14. The user's preferred length of shows or movies (e.g., "I prefer short episodes that I can watch during my lunch break.")
15. The user's tendency to rewatch shows or movies (e.g., "I rewatch my favorite movie every year.")

When you receive a message, you perform a sequence of steps consisting of:

1. Analyze the message for information.
2. If it has any information worth recording, return TRUE. If not, return FALSE.

You should ONLY RESPOND WITH TRUE OR FALSE. Absolutely no other information should be provided.

Take a deep breath, think step by step, and then analyze the following message:
"""

# Not sure if we need this as a category:
# 16. The user's awareness of the popularity of a show or movie (e.g., "I know that show is really popular, but I haven't watched it yet.")

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Remember, only respond with TRUE or FALSE. Do not provide any other information.",
        )
    ]
)

llm = OllamaLLM()


# Define a function to convert dictionaries to LangChain message objects
def convert_to_messages(messages):
    langchain_messages = []
    for message in messages:
        if message["role"] == "user":
            langchain_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "system":
            langchain_messages.append(SystemMessage(content=message["content"]))
    return langchain_messages


# Define a function to format the prompt
def format_prompt(messages):
    # Convert the input messages to LangChain message objects
    langchain_messages = convert_to_messages(messages)

    # Format the messages into a ChatPromptValue object
    chat_prompt_value = prompt.format_messages(messages=langchain_messages)

    # Convert the ChatPromptValue object into a single string
    formatted_prompt = "\n".join([message.content for message in chat_prompt_value])
    return formatted_prompt


# Define a function to run the pipeline
def run_pipeline(messages):
    # Format the prompt
    formatted_prompt = format_prompt(messages)

    # Call the OllamaLLM
    response = llm._call(formatted_prompt)
    return response


messages = [
    {"role": "user", "content": "I despite eggplants please dont even mention the word eggplant"}
]

# Run the pipeline
response = run_pipeline(messages)
print(response)