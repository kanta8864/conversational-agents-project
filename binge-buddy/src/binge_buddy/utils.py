import re


def remove_think_tags(response: str) -> str:
    """
    Removes all <think>...</think> tags and their content from the response string.

    :param response: The raw string containing the response with <think> tags.
    :return: The cleaned response with the <think> tags removed.
    """
    # Regular expression to match <think>...</think> tags and their content
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    return clean_response.strip()
