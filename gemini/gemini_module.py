from google import genai
from ics import Calendar, Event
import os
from datetime import datetime
import json
import time

def convert_text_to_ics(text, api_key, output_file="event.ics"):
    """
    Uses Gemini API to extract event details from text and create an .ics file.
    """
    client = genai.Client(api_key=api_key)

    model = 'models/gemma-4-26b-a4b-it'
    
    today = datetime.now().strftime("%Y-%m-%d")
    timezone = time.strftime('%Z %z')
    prompt = (
        f"Today's date is {today} and the user's timezone is {timezone}. "
        f"Extract event details from the following text: '{text}'. "
        "Return the result in a strict JSON format with the following keys: "
        "'summary', 'begin' (ISO 8601 format), 'end' (ISO 8601 format), 'description', 'location'. "
        "If a field is missing, use null. Only return the JSON."
    )
    
    start_time = time.time()
    response = client.models.generate_content(model=model, contents=prompt)
    end_time = time.time()
    print(f"API call took {end_time - start_time:.2f} seconds")

    # Basic cleaning of the response to ensure it's valid JSON
    if response.text is None:
        raise ValueError("Gemini API returned an empty response.")
    json_text: str = response.text.strip().replace('```json', '').replace('```', '')
    print(f"Raw Gemini response: {json_text}")  # Debugging output
    event_data = json.loads(json_text)
    
    c = Calendar() 
    e = Event()
    e.name = event_data.get('summary', 'Untitled Event')
    e.begin = event_data.get('begin')
    e.end = event_data.get('end')
    e.description = event_data.get('description', '')
    e.location = event_data.get('location', '')
    
    c.events.add(e)
    
    with open(output_file, 'w') as f:
        f.writelines(c.serialize_iter())
    
    return output_file

if __name__ == "__main__":
    # Example usage
    API_KEY = "AIzaSyC2SqGRRZ3qXP1kWSe-j5ZMr0_stjlu53c"
    test_text = "Meeting with John tomorrow at 10am for 1 hour at the coffee shop"
    if API_KEY != "YOUR_GEMINI_API_KEY":
        ics_file = convert_text_to_ics(test_text, API_KEY)
        print(f"ICS file created: {ics_file}")
    else:
        print("Please provide a valid Gemini API key.")
