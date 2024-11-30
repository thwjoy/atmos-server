import argparse
import asyncio
import sys
import numpy as np
import websockets
import pyaudio
import wave
import io
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
import struct
import uuid

message_queue = asyncio.Queue()

# WebSocket server URL
SERVER_URI = "ws://localhost:5001"  # Replace <server_ip> with your server's IP
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJleHAiOjE3MzMwNjg5NzksImlhdCI6MTczMjIwNDk3OSwiaXNzIjoieW91ci1hcHAtbmFtZSJ9.irNjsFJSjdxWqfRZqHclf4Pb78-hNIYTr9PRuZJYtQ8"

# Audio stream settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1

playing_narration = asyncio.Event()

def is_uuid(message):
    try:
        # Attempt to create a UUID object from the message
        uuid_obj = uuid.UUID(message)
        return True
    except (ValueError, TypeError):
        # If parsing fails, it's not a valid UUID
        return False

def play_audio_sync(audio_bytes, sample_rate=SAMPLE_RATE, volume=1.0):
    """Plays the received audio data with adjustable volume."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, output=True)

    # Use a BytesIO stream to read the audio bytes
    with io.BytesIO(audio_bytes) as audio_stream:
        wave_file = wave.open(audio_stream, "rb")
        data = wave_file.readframes(CHUNK_SIZE)
        while data:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Apply volume adjustment
            audio_array = (audio_array * volume).astype(np.int16)
            
            # Convert back to bytes
            adjusted_data = audio_array.tobytes()
            
            # Play adjusted data
            stream.write(adjusted_data)
            
            # Read the next chunk of audio data
            data = wave_file.readframes(CHUNK_SIZE)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Audio playback complete.")

async def play_audio(audio_bytes, sample_rate, indicator):
    """Plays audio asynchronously and manages the `pause_recording` event based on indicator."""
    if indicator == "STORY":
        playing_narration.set()  # Pause recording only for STORY
    try:
        await asyncio.to_thread(play_audio_sync, audio_bytes, sample_rate)
    finally:
        if indicator == "STORY":
            playing_narration.clear()  # Resume recording after STORY playback

async def send_recording(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    print("Streaming audio to server... Press Ctrl+C to stop.")
    try:
        while True:
            # Read audio chunk from the microphone
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # Send audio chunk to the server
            if not playing_narration.is_set():
                await websocket.send(audio_chunk)
                # TODO check this
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        print("Stopped streaming.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def send_audio(websocket, wav_file_path):
    # Open the WAV file
    wav_file = wave.open(wav_file_path, 'rb')

    # Get audio parameters
    channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    chunk_size = 1024  # Adjust the chunk size as needed

    # Set up PyAudio for playback
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True)

    print("Streaming and playing audio... Press Ctrl+C to stop.")
    try:
        while True:
            # Read a chunk of audio data from the WAV file
            audio_chunk = wav_file.readframes(chunk_size)

            # Stop when we reach the end of the file
            if not audio_chunk:
                print("Finished streaming and playing the WAV file.")
                break

            # Send the audio chunk to the server
            await websocket.send(audio_chunk)

            # Play the audio chunk
            stream.write(audio_chunk)

            # Simulate real-time streaming
            await asyncio.sleep(chunk_size / frame_rate)
    except KeyboardInterrupt:
        print("Stopped streaming and playback.")
    finally:
        # Clean up
        wav_file.close()
        stream.stop_stream()
        stream.close()
        p.terminate()

async def receive_audio(websocket):
    print("Listening for audio data...")
    HEADER_FORMAT = ">5sI16sIII"  # Updated to match the sender's format
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    # Dictionary to store accumulated audio data by sequence ID
    accumulated_audio = {}

    while True:
        try:
            # Receive each message from the server
            message = await websocket.recv()
            # Check if the message is text or binary data
            if isinstance(message, str):
                # Handle text messages, like notifications or category labels
                print("Received text message:", message)
            else:
                # Unpack the header
                indicator, audio_size, sequence_id, packet_count, total_packets, sample_rate = struct.unpack(
                    HEADER_FORMAT, message[:HEADER_SIZE]
                )
                indicator = indicator.decode().strip()  # Convert bytes to string and remove any padding
                sequence_id = uuid.UUID(bytes=sequence_id)  # Convert binary UUID to string representation

                # Accumulate audio data for the specific sequence
                if sequence_id not in accumulated_audio:
                    accumulated_audio[sequence_id] = b""

                accumulated_audio[sequence_id] += message[HEADER_SIZE:]

                # Check if the sequence is complete
                if len(accumulated_audio[sequence_id]) >= audio_size:
                    print(f"Received complete sequence for {indicator} with ID {sequence_id}.")
                    asyncio.create_task(play_audio(accumulated_audio.pop(sequence_id), sample_rate=sample_rate, indicator=indicator))
        except websockets.ConnectionClosed:
            print("WebSocket connection closed.")
            break

async def poll_input():
    """Poll for input without blocking"""
    print("Enter a message: ", end="", flush=True)
    loop = asyncio.get_running_loop()
    while True:
        # Check if there is input available without blocking
        if await loop.run_in_executor(None, sys.stdin.readable):
            # Read the line in a non-blocking way
            message = await loop.run_in_executor(None, sys.stdin.readline)
            await message_queue.put(message.strip())
        await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
            
# send a message entered in command line to the server
async def send_text(websocket):
    while True:
        message = await message_queue.get()
        await websocket.send(message)
        await asyncio.sleep(0.1)  # Small delay to allow message processing

async def main():
    parser = argparse.ArgumentParser(description="Stream and play a WAV file while sending it to a WebSocket server.")
    parser.add_argument(
        "--wav_file",
        type=str,
        help="Path to the WAV file to stream and play."
    )
    args = parser.parse_args()

    co_auth = args.wav_file is None

    headers = {
        "Authorization": f"Bearer {TOKEN}",  # Common format for tokens
        "CO-AUTH": f"{co_auth}"
    }
    async with websockets.connect(SERVER_URI, extra_headers=headers) as websocket:
        # recieve the first message
        message = await websocket.recv()
        print(message)
        # check if message is a UUID type
        if is_uuid(message):
            print("Received UUID:", message)

            if co_auth:
                send_task = asyncio.create_task(send_recording(websocket))
            else:
                send_task = asyncio.create_task(send_audio(websocket, args.wav_file))
            receive_task = asyncio.create_task(receive_audio(websocket))
            # input_task = asyncio.create_task(poll_input())
            await asyncio.gather(receive_task, send_task)

asyncio.run(main())
