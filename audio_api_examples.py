from openai import OpenAI
import os
from pathlib import Path

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Text-to-Speech (TTS) ----
# Generate spoken audio from text
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="The quick brown fox jumped over the lazy dog."
)

# Stream to file
response.stream_to_file(speech_file_path)
print(f"TTS output saved to {speech_file_path}")

# ---- Speech-to-Text (Whisper) ----
# Transcribe the audio file we just created
with open(speech_file_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

print("Transcription:\n", transcription.text)
