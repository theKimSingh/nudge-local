import whisper
import os
import whisper

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribes audio to text using OpenAI's Whisper model.
    """
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing audio from {audio_path}...")
    result = model.transcribe(audio_path)
    
    return result['text']

if __name__ == "__main__":
    # Example usage
    test_audio = "test.wav"
    if os.path.exists(test_audio):
        text = transcribe_audio(test_audio)
        print(f"Transcription: {text}")
    else:
        print(f"Please provide a {test_audio} file for testing.")
