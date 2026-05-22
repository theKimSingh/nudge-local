import os
from stt_module import transcribe_audio
from gemini_module import convert_text_to_ics
import whisper

def main():
    # Configuration
    model = whisper.load_model("base")
    AUDIO_FILE = "input.wav"  # Change this to your audio file path
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Set this in your environment variables
    
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # if not os.path.exists(AUDIO_FILE):
    #     print(f"Error: Audio file {AUDIO_FILE} not found.")
    #     return

    try:
        # Step 1: Speech to Text
        print("Step 1: Transcribing audio...")
        # transcribed_text = transcribe_audio(AUDIO_FILE)
        # print(f"Transcribed Text: {transcribed_text}")

        # Step 2: Text to ICS via Gemini
        print("\nStep 2: Converting text to .ics calendar event...")
        ics_filename = convert_text_to_ics("hello meet me my mom at 5:30pm on may 19", GEMINI_API_KEY)
        print(f"Success! Calendar event saved to: {ics_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
