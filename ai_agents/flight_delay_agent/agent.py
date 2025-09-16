from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv
from flight_tools import extract_flight_info, check_flight_delay

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def flight_tool_func(user_input: str):
    flight, date = extract_flight_info(user_input)
    if not flight or not date:
        return "Please provide a valid flight number and date."
    return check_flight_delay(flight, date)

def get_agent():
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    flight_tool = Tool(
        name="Flight Delay Checker",
        func=flight_tool_func,
        description="Use this to check if a flight is delayed. Input should include a flight number and date."
    )

    agent = initialize_agent(
        tools=[flight_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
