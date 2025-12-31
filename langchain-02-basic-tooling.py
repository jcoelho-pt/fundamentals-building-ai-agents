import os
import langchain
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Variables and Constants
load_dotenv()  # Load environment variables from .env file

# Configure the Wikipedia wrapper to fetch 1 result with up to 1000 characters
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1000
)

# Initialize the Wikipedia query tool
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# NOTE:You don’t need an API key for basic Wikipedia queries, 
# but be mindful of rate limits if making frequent requests. 

# Displaying tool metadata
print("-- Tool Metadata ------------")
print(tool.name)         # wikipedia
print(tool.description)  # A wrapper around Wikipedia...
print(tool.args)         # {'query': {'title': 'Query', 'type': 'string'}}
print(".")

# Running the Wikipedia tool
print("Querying Wikipedia for 'Worldbuilding'...")
result = tool.run({"query": "What is Worldbuilding?"})
print(result)
