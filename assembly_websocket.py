from typing import Optional
import assemblyai as aai
import pydub
import soundfile as sf
import sounddevice as sd
import os
import asyncio
import websockets
import json
from assembly_db import SoundAssigner, OPENAI_API_KEY

aai.settings.api_key = "09485e2cc7b741d4aa2922da67f84094" 

assigner = SoundAssigner(chroma_path="ESC-50")

all_sounds = [
    "dog", "rain", "crying_baby", "door_knock", "helicopter",
    "rooster", "sea_waves", "sneezing", "mouse_click", "chainsaw",
    "pig", "crackling_fire", "clapping", "keyboard_typing", "siren",
    "cow", "crickets", "breathing", "door_wood_creaks", "car_horn",
    "frog", "chirping_birds", "coughing", "can_opening", "engine",
    "cat", "water_drops", "footsteps", "washing_machine", "train",
    "hen", "wind", "laughing", "vacuum_cleaner", "church_bells",
    "insects", "pouring_water", "brushing_teeth", "clock_alarm", "airplane",
    "sheep", "toilet_flush", "snoring", "clock_tick", "fireworks",
    "crow", "thunderstorm", "drinking_sipping", "glass_breaking", "hand_saw"
]

# Function to convert audio to WAV and read binary data
def get_audio_bytes(audio_path):
    pydub.AudioSegment.from_file(audio_path).export("temp.wav", format="wav")
    with open("temp.wav", "rb") as audio_file:
        return audio_file.read()

async def process_transcript_async(transcript, websocket):
    """Async function to process and send the audio file if a match is found."""
    if not transcript.text:
        return
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end="\r\n")
        filename, category = assigner.retrieve_src_file(transcript.text)
        if filename is not None:
            print(f"Sending sound for category '{category}' to client.")
            audio_bytes = get_audio_bytes(os.path.join("ESC-50-master/audio", filename))
            try:
                await websocket.send(category)  # Debug confirmation message
                await websocket.send(audio_bytes)  # Send the actual audio bytes
                print("[DEBUG] Audio data sent to client.")
            except Exception as e:
                print(f"[ERROR] Failed to send audio data: {e}")
        else:
            print("No sound found for the given text.")

def on_data(transcript, websocket, loop):
    """Callback function that schedules async transcript processing in the main event loop."""
    asyncio.run_coroutine_threadsafe(process_transcript_async(transcript, websocket), loop)

def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    print("Closing Session")

# WebSocket server to receive audio from a client
async def audio_receiver(websocket, path):
    print("Client connected")
    
    # Get the main event loop
    loop = asyncio.get_running_loop()

    # Initialize the transcriber with an on_data handler that uses the main loop
    transcriber = aai.RealtimeTranscriber(
        on_data=lambda transcript: on_data(transcript, websocket, loop),
        on_error=on_error,
        sample_rate=44_100,
        on_open=lambda session: print("Session ID:", session.session_id),
        on_close=on_close,
        word_boost=all_sounds,
        end_utterance_silence_threshold=500
    )

    try:
        # Connect the transcriber
        transcriber.connect()

        # Stream audio data from the WebSocket client to AssemblyAI
        async for message in websocket:
            # Assume audio data is received as binary data
            transcriber.stream([message])  # Streaming to AssemblyAI
     
    except websockets.ConnectionClosed as e:
        print(f"Client disconnected: {e}")
    finally:
        transcriber.close()


# Start WebSocket server
async def start_server():
    server = await websockets.serve(audio_receiver, "0.0.0.0", 8765)
    print("WebSocket server started on ws://0.0.0.0:8765")
    await server.wait_closed()

# Run the server
if __name__ == "__main__":
    asyncio.run(start_server())