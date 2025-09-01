import re
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
FLIGHT_API_KEY = os.getenv("FLIGHT_API_KEY")
FLIGHT_API_URL = "http://api.aviationstack.com/v1/flights"

def extract_flight_info(user_input: str):
    match = re.search(r'\b([A-Z]{2,3}\d{2,4})\b', user_input)
    date_match = re.search(r'(\b\d{4}-\d{2}-\d{2}\b|\b\w+ \d{1,2}\b)', user_input)

    flight_number = match.group(1) if match else None
    date_str = date_match.group(1) if date_match else None

    try:
        if date_str and len(date_str.split()) == 2:
            date = datetime.strptime(date_str, "%B %d").replace(year=datetime.now().year)
        else:
            date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        date = None

    return flight_number, date

def check_flight_delay(flight_number: str, date: datetime) -> str:
    params = {
        "access_key": FLIGHT_API_KEY,
        "flight_iata": flight_number,
        "flight_date": date.strftime("%Y-%m-%d")
    }

    try:
        resp = requests.get(FLIGHT_API_URL, params=params)
        data = resp.json()

        if data.get("data"):
            flight = data["data"][0]
            status = flight.get("flight_status")
            delay = flight.get("departure", {}).get("delay")

            if status == "active" and delay:
                return f"Yes, flight {flight_number} is currently delayed by {delay} minutes."
            elif status == "landed" and delay:
                return f"Flight {flight_number} landed with a delay of {delay} minutes."
            else:
                return f"Flight {flight_number} on {date.strftime('%B %d')} is on time."
        return "I couldn't find information for that flight."

    except Exception as e:
        return f"Error checking flight status: {str(e)}"
