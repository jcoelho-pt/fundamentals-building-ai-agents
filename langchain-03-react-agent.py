import os
import langchain
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chat_models import init_chat_model # Initialize chat model function
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
import re
from typing import Dict, Union

# Variables and Constants
load_dotenv()  # Load environment variables from .env file
llm_model = "gemini-3-flash-preview" # options: gemini-3-flash-preview, gpt-4o-mini
llm_provider = "google_genai" # alternatives: google_genai, openai
llm_temperature = 0.8 # Temperature setting for the LLM, the higher the value, the more creative the output. > 1 is very creative but may lose coherence.

print(f"-- Using LangChain version: {langchain.__version__}")


# Initialize a language model (LLM) ###################################
print("Initializing LLM...")
llm = init_chat_model(model=llm_model, model_provider=llm_provider, temperature=llm_temperature)

exit_if_not_configured = llm is None
if exit_if_not_configured:
    print("LLM not configured properly. Exiting.")
    exit(1)

print(f"-- Using model: {llm.name}")
print("...")


# Defining the tools #########################################
print("Defining Tools...")
@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """Extracts and sum all integers and decimals from the input.

    Args:
        inputs (str): A list of numbers to sum.
    Returns:
        dict: A dictionary containing the sum and a message.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", inputs)
    if not matches:
        return {"result": "No numbers found in the input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
        return {"result": total}
    except Exception as e:
        return {"error": f"Error during summation: {str(e)}"}

print("Defining 'add_two_integers' Tool...")
@tool
def add_two_integers(a: int, b: int) -> int:
    """Add a and b (integers) and return the result."""
    return a + b

print("Defining 'subtract_two_integers' Tool...")
@tool
def subtract_two_integers(a: int, b: int) -> int:
    """Subtract b from a (integers) and return the result."""
    return a - b

print("Defining 'multiply_two_integers' Tool...")
@tool
def multiply_two_integers(a: int, b: int) -> int:
    """Multiply a and b (integers) and return the result."""
    return a * b

print("Defining 'wikipedia_search' Tool...")
@tool
def wikipedia_search(query: str) -> dict:
    """Query Wikipedia and return both the text result and the page URL.

    Returns a dict with keys `text` and `url` so callers can access
    the article content and the canonical Wikipedia page link.

    Args:
        query (str): The search query string.
    Returns:
        dict: A dictionary with 'text' and 'url' keys.
    """
    # NOTE: You don’t need an API key for basic Wikipedia queries,
    # but be mindful of rate limits if making frequent requests.

    if not hasattr(wikipedia_search, "wikipedia_api_wrapper"):
        wikipedia_api_wrapper = WikipediaAPIWrapper(
            lang="en", top_k_results=1, doc_content_chars_max=1000
        )

    text = ""
    try:
        text = wikipedia_api_wrapper.run(query)
    except Exception:
        text = ""

    # Resolve the page title & URL using MediaWiki search API (best-effort)
    page_title = ""
    try:
        search_results = wikipedia_api_wrapper.load(query)
        search_result = search_results[0]
        page_title = search_result.metadata["title"]
        print(f"Search result page title: {page_title}")
    except Exception:
        print("Could not retrieve Wikipedia page title.")
        page_title = ""
    
    try:
        wiki_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    except Exception:
        wiki_url = ""

    return {"text": text, "url": wiki_url}

print("...")


# Conntecting the list of tools to the model  ###########################
print("Connecting Tools to the LLM...")

math_tools = [add_two_integers, subtract_two_integers, multiply_two_integers] # List of tool functions to connect

llm_with_tools = llm.bind_tools(math_tools) # Bind the tools to the chat model
print("...")


# Creating the agent
math_agent = create_agent (
    model = llm, # the model handles reasoning and decides whent to call a tool
    tools = math_tools, # list of tool functions the agent can use
    system_prompt = "You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning.", # system prompt defining the agent’s personality and instructions
)

# Giving the agent a conversation history, which simulates a chat interaction
result = math_agent.invoke({
    "messages": [
        { "role": "user", "content": "Add the numbers -10 and -20"}
    ]
})

print("Agent Result:", result)