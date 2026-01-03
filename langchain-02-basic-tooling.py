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


# Use the `@tool` decorator to expose a function-based Wikipedia tool
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

# Running the Wikipedia tool
print("#### Querying Wikipedia for 'Worldbuilding'...")
try:
    # If decorator produced a Tool-like object with a .run API
    result = wikipedia_search.run({"query": "What is Worldbuilding?"})
except Exception:
    # Otherwise call the decorated callable directly with a string
    result = wikipedia_search("What is Worldbuilding?")

print("#### Result from Wikipedia Tool:")
print(result)
