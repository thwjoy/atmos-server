import asyncio
import numpy as np
import websockets
import pyaudio
import wave
import io
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
import struct

message_queue = asyncio.Queue()

# WebSocket server URL
SERVER_URI = "ws://localhost:8765"  # Replace <server_ip> with your server's IP

# Audio stream settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1

playing_narration = asyncio.Event()

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

async def play_audio(audio_bytes, sample_rate):
    """Runs play_audio_sync in a background thread."""
    await asyncio.to_thread(play_audio_sync, audio_bytes, sample_rate)

async def send_audio(websocket):
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

async def receive_audio(websocket):
    print("Listening for audio data...")
    HEADER_FORMAT = '>5sII'  # Matches server's format: 5-byte indicator, 4-byte audio size
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

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
                indicator, audio_size, sample_rate = struct.unpack(HEADER_FORMAT, message[:HEADER_SIZE])
                indicator = indicator.decode().strip()  # Convert bytes to string and remove any padding
                print(indicator, audio_size, sample_rate)
                # Receive the audio data in chunks until complete
                accumulated_audio = message[HEADER_SIZE:]

                # Keep receiving data until the accumulated data matches the expected audio size
                while len(accumulated_audio) < audio_size:
                    chunk = await websocket.recv()
                    accumulated_audio += chunk
                
                if indicator == "STORY":
                    playing_narration.set()
                    play_audio_sync(accumulated_audio, sample_rate, volume=2.0)
                    playing_narration.clear()
                else:
                    asyncio.create_task(play_audio(accumulated_audio, sample_rate=sample_rate))  # Play in the background

        except websockets.ConnectionClosed:
            print("Server disconnected")
            break

# async def poll_input():
#     """Poll for input without blocking"""
#     print("Enter a message: ", end="", flush=True)
#     loop = asyncio.get_running_loop()
#     while True:
#         # Check if there is input available without blocking
#         if await loop.run_in_executor(None, sys.stdin.readable):
#             # Read the line in a non-blocking way
#             message = await loop.run_in_executor(None, sys.stdin.readline)
#             await message_queue.put(message.strip())
#         await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
            
# send a message entered in command line to the server
# async def send_text(websocket):
#     while True:
#         message = await message_queue.get()
#         await websocket.send(message)
#         await asyncio.sleep(0.1)  # Small delay to allow message processing

async def main():
    async with websockets.connect(SERVER_URI) as websocket:
        send_task = asyncio.create_task(send_audio(websocket))
        receive_task = asyncio.create_task(receive_audio(websocket))
        # input_task = asyncio.create_task(poll_input())
        await asyncio.gather(send_task, receive_task)

asyncio.run(main())
