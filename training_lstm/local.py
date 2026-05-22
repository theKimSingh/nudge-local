import asyncio
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# Configuration
# Point this to your local fine-tuned folder
MODEL_NAME = "./opt350m_event_extractor1/checkpoint-20"
# Fallback to base repository for tokenizer if local files are missing/corrupted
TOKENIZER_NAME = "facebook/opt-350m" 
OUTPUT_DIR = "./opt350m_event_extractor1"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Read and parse your synthetic test suite safely
try:
    with open("./synthetic_event_extraction_dataset.json", "r", encoding="utf-8") as f:
        test_suite = json.load(f)
except FileNotFoundError:
    print("❌ Error: 'synthetic_event_extraction_dataset.json' not found.")
    test_suite = []

print(f"📦 Loading local weights from: {MODEL_NAME}...")
# Initialize the Hugging Face generation pipeline safely
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1 # Uses GPU if available, else CPU
)
print("✅ Model loaded successfully.\n")


# 2. Core extraction logic using local Hugging Face model
def extract_event_data_sync(text: str, test_date: str = None) -> dict:
    today = test_date or datetime.utcnow().strftime("%Y-%m-%d")
    timezone = "America/New_York"

    # Because OPT-350M is a base completion model, we use a few-shot prompt 
    # to teach it by example how to extract the target data format.
    prompt = f"""Task: Extract event details as a single JSON object.
Today's date: {today}
Timezone: {timezone}

Input: "Let's meet tomorrow at 3pm for an hour to review the project."
Output: {{"id": null, "summary": "Review the project", "begin": "{today[:4]}-05-21T15:00:00", "duration": 60, "repeats": null}}

Input: "Team sync every Wednesday at 10 AM"
Output: {{"id": null, "summary": "Team sync", "begin": "{today[:4]}-05-20T10:00:00", "duration": 60, "repeats": "weekly"}}

Input: "{text}"
Output:"""

    # Generate text completion
    outputs = generator(
        prompt, 
        temperature=0.1, 
        do_sample=False,
        max_new_tokens=128,  # Forces the small model to stop generating and not hang
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]['generated_text']
    
    # Isolate just the generated output segment following our prompt
    response_content = generated_text[len(prompt):].strip()
    
    # Attempt to extract a JSON string block if the model added trailing text
    json_match = re.search(r'\{.*?\}', response_content, re.DOTALL)
    if json_match:
        response_content = json_match.group(0)

    try:
        raw_event = json.loads(response_content)
    except json.JSONDecodeError:
        # Fallback if the small model didn't construct perfect JSON
        return {"id": None, "summary": f"Parse Error (Raw: {response_content[:30]})", "begin": None, "duration": None, "repeats": None}

    # Safely format the 'begin' date
    begin_raw = raw_event.get("begin") if isinstance(raw_event, dict) else None
    begin_formatted = begin_raw.split(".")[0] if begin_raw else None

    # Handle duration rounding safely
    duration_raw = raw_event.get("duration") if isinstance(raw_event, dict) else None
    duration_formatted = round(duration_raw) if duration_raw is not None else None

    return {
        "id": raw_event.get("id"),
        "summary": raw_event.get("summary"),
        "begin": begin_formatted,
        "duration": duration_formatted,
        "repeats": raw_event.get("repeats"),
    }


# Wrapper to preserve the async structural needs of your pipeline
async def extract_event_data(text: str, test_date: str = None) -> dict:
    # Run the synchronous heavy transformer prediction inside a thread pool 
    # so it won't lock up your async event loop runner.
    return await asyncio.to_thread(extract_event_data_sync, text, test_date)


# 3. Automated Test Execution Loop
async def run_test_suite():
    if not test_suite:
        print("No tests to run.")
        return

    print(
        f"🚀 Loaded {len(test_suite)} cases from synthetic.json. Running accuracy tests against OPT-350M...\n"
    )
    passed_count = 0
    mock_today = "2026-05-20"

    for test in test_suite:
        print(f'Testing: "{test.get("name")}"')
        print(f'Input:   "{test.get("input")}"')

        try:
            actual = await extract_event_data(test.get("input"), mock_today)

            # Read expected properties
            expected_output = test.get("output", {})
            expected_summary = expected_output.get("summary", "")
            expected_begin = expected_output.get("begin", "")
            expected_duration = expected_output.get("duration")

            # Perform safe verification checks
            summary_pass = (
                expected_summary.lower() in actual["summary"].lower()
                if actual and actual.get("summary")
                else False
            )

            begin_pass = (
                actual["begin"].startswith(expected_begin)
                if actual and actual.get("begin") and expected_begin
                else False
            )

            duration_pass = actual and actual.get("duration") == expected_duration

            if summary_pass and begin_pass and duration_pass:
                print("✅ PASSED\n")
                passed_count += 1
            else:
                expected_display = {
                    "summary": expected_summary,
                    "begin": expected_begin,
                    "duration": expected_duration,
                }
                print("❌ FAILED")
                print(f"   Expected: {json.dumps(expected_display)}")
                print(f"   Actual:   {json.dumps(actual)}")
                print()

        except Exception as error:
            print(f"💥 ERROR: {str(error)}\n")

    print("--- Test Summary ---")
    print(f"Passed: {passed_count} / {len(test_suite)}")


if __name__ == "__main__":
    asyncio.run(run_test_suite())