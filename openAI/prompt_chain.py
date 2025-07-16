from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM (uses OPENAI_API_KEY from environment)
llm = OpenAI(temperature=0.7)

def run_prompt_chain(prompt: str) -> str:
    """
    Run a prompt through the LLM and return the response.
    """
    # You can add more orchestration steps here if needed
    return llm.invoke(prompt) 