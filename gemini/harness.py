import datetime
import os
import json
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# 1. Define the Expected Output Schema
# ==========================================
class EventModel(BaseModel):
    summary: str = Field(description="Brief title of the event")
    begin: Optional[str] = Field(description="ISO 8601 format date-time string. Default to an hour duration if missing.")
    end: Optional[str] = Field(description="ISO 8601 format date-time string. Default to an hour duration if missing.")
    description: Optional[str] = Field(default=None, description="Detailed description of the event, or null")
    location: Optional[str] = Field(default=None, description="Location/venue of the event, or null")

class EventResponseSchema(BaseModel):
    events: List[EventModel] = Field(description="List of events extracted from the text. Empty if no events found.")

# ==========================================
# 2. Evaluation Helper Functions
# ==========================================
def normalize_event(event: dict) -> dict:
    """Normalizes keys and lowercase values for uniform comparison."""
    return {
        "summary": str(event.get("summary", "")).strip().lower(),
        "begin": event.get("begin"),
        "end": event.get("end"),
        "description": event.get("description"),
        "location": str(event.get("location")).strip().lower() if event.get("location") else None
    }

def evaluate_prediction(expected: list, predicted: list) -> bool:
    """Checks if the predicted array matches the expected array regardless of order."""
    if len(expected) != len(predicted):
        return False
    
    norm_expected = sorted([normalize_event(e) for e in expected], key=lambda x: x.get('begin') or '')
    norm_predicted = sorted([normalize_event(p) for p in predicted], key=lambda x: x.get('begin') or '')
    
    for exp, pred in zip(norm_expected, norm_predicted):
        if exp["begin"] != pred["begin"] or exp["end"] != pred["end"] or exp["location"] != pred["location"]:
            return False
    return True

# ==========================================
# 3. Main Execution Engine
# ==========================================
def run_test_harness():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # Load the dataset dynamically from the local JSON file
    dataset_path = "dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Target dataset file '{dataset_path}' not found in current directory.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        try:
            test_dataset = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse '{dataset_path}'. Verify it is valid JSON. \nDetail: {e}")
            return

    # Initialize Client
    client = genai.Client()
    
    # System instructions to lock in the temporal anchor matching the dataset context
    system_instruction = (
        "You are an AI that extracts event timelines from unstructured text expressions. "
        "The current reference date is Tuesday, May 19, 2026. "
        "Use this anchor point to calculate relative dates like 'tomorrow', 'tonight', or 'this afternoon'. "
        "If an event's exact duration is not specified, default to exactly 1 hour. "
        "Set missing or unknown descriptions or locations to null values."
    )

    success_count = 0
    total_count = len(test_dataset)

    print(f"Loaded {total_count} test scenarios from '{dataset_path}'...")
    print(f"Starting evaluation...\n")
    print(f"{'-'*70}")

    for idx, item in enumerate(test_dataset, 1):
        text = item.get("input", "")
        # Fallback to look for either 'expected' or 'output' as the key name
        expected_output = item.get("expected") if "expected" in item else item.get("output", [])
        today = datetime.date(2026, 5, 19)  # datetime.now().strftime("%Y-%m-%d")
        timezone = time.strftime('%Z %z')
        prompt = (
            f"Today's date is {today} and the user's timezone is {timezone}. "
            f"Extract event details from the following text: '{text}'. "
            "Return the result in a strict JSON format with the following keys: "
            "'summary', 'begin' (ISO 8601 format), 'end' (ISO 8601 format), 'description', 'location'. "
            "If a field is missing, use null. Only return the JSON."
        )
        
        try:
            # Query Gemini using Structured Output rules
            response = client.models.generate_content(
                model='gemma-4-31b-it',
                contents=prompt,
                # config=types.GenerateContentConfig(
                #     system_instruction=system_instruction,
                #     response_mime_type="application/json",
                #     response_schema=EventResponseSchema,
                #     temperature=0.1 
                # ),
            )
            
            raw_json = json.loads(response.text)
            predicted_output = raw_json.get("events", [])
            
            is_correct = evaluate_prediction(expected_output, predicted_output)
            if is_correct:
                success_count += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(f"Test #{idx}: {status}")
            print(f"Input   : {text_input}")
            print(f"Expected: {json.dumps(expected_output)}")
            print(f"Got     : {json.dumps(predicted_output)}")
            print(f"{'-'*70}")

        except Exception as e:
            print(f"Test #{idx}: 💥 ERROR evaluating request: {e}")
            print(f"{'-'*70}")

    # Calculate Accuracy Score
    accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Cases Evaluated : {total_count}")
    print(f"Successful Identifications : {success_count}")
    print(f"Final Model Accuracy       : {accuracy:.2f}%")
    print(f"====================================================")

if __name__ == "__main__":
    run_test_harness()