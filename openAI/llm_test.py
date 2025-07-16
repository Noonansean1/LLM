from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM (ensure your OPENAI_API_KEY is set in your environment)
llm = OpenAI(temperature=0.7)

# Run a simple prompt
prompt = "What are three interesting facts about the planet Mars?"
response = llm(prompt)

print("Response from LLM:")
print(response)
