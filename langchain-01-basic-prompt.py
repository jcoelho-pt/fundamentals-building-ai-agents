import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAI

# Variables and Constants
load_dotenv()  # Load environment variables from .env file

# Ollama LLM
# MODEL = "llama3.2"
# llm = ChatOllama(model=MODEL, temperature=0,# other params...)

# Initialize a language model (LLM) ###################################
print("Intializing LLM...")
# llm = ChatOllama(model=MODEL)
llm = GoogleGenerativeAI(model="gemini-3-flash-preview") # or gemini-3-pro-preview
print(f"-- Using model: {llm.model}")

# Invoking the LLM with a prompt #########################################
prompt = "What would be a good company name for a startup that makes CMMS software for Water Utilities? Please suggest three names."
print(f"Invoking LLM with prompt: {prompt}")

# Run the LLM with the prompt
response = llm.invoke(prompt)
print(response)