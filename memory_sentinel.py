# from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage, HumanMessage
from ollamaLLM import OllamaLLM

system_prompt_initial = """
Your job is to assess a brief chat history in order to determine if the conversation contains any details about a user's watching habits regarding streaming content. 

You are part of a team building a knowledge base regarding a users' watching habits to assist in highly customized streamming content recommendation.

You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.

You are only interested in the following categories of information:

1. The family's food allergies (e.g. a dairy or soy allergy)
2. Foods the family likes (e.g. likes pasta)
3. Foods the family dislikes (e.g. doesn't eat mussels)
4. Attributes about the family that may impact weekly meal planning (e.g. lives in Austin; has a husband and 2 children; has a garden; likes big lunches; etc.)

When you receive a message, you perform a sequence of steps consisting of:

1. Analyze the message for information.
2. If it has any information worth recording, return TRUE. If not, return FALSE.

You should ONLY RESPOND WITH TRUE OR FALSE. Absolutely no other information should be provided.

Take a deep breath, think step by step, and then analyze the following message:
"""

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