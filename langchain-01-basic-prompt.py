import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.messages import AIMessage,HumanMessage,SystemMessage
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAI

# Variables and Constants
load_dotenv()  # Load environment variables from .env file
llm_temperature = 0.8 # Temperature setting for the LLM, the higher the value, the more creative the output. > 1 is very creative but may lose coherence.


# Initialize a language model (LLM) ###################################
print("Intializing LLM...")
# llm = ChatOllama(model="llama3.2")
llm = GoogleGenerativeAI(model = "gemini-3-flash-preview", temperature = llm_temperature) # or gemini-3-pro-preview
print(f"-- Using model: {llm.model}")


# Message examples ######################################################
system_msg = SystemMessage(content="You are a helpful assistant specialist in dark fantasy literature and worldbuilding.") # To set the behavior and context of the AI model.
human_msg = HumanMessage(content="Suggest three names for a dark fantasy world.")   # To simulate or process a question, command, or text input made by a user.
ai_msg = AIMessage(content="1. Shadowmoor\n2. Nightfall Dominion\n3. Duskveil") # To represent the model's response to the user's input.
messages = [system_msg, human_msg]  # Combine messages into a list for processing.

# Invoking the LLM with a prompt #########################################
# prompt = "Suggest three names for a dark fantasy world.
print(f"Invoking LLM with prompt: {human_msg.content}")


# Run the LLM with the prompt
response = llm.invoke(messages)
print(response)