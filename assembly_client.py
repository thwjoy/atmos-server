import asyncio
import websockets
import pyaudio
import wave
import io

# WebSocket server URL
SERVER_URI = "ws://localhost:8765"  # Replace <server_ip> with your server's IP

# Audio stream settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1

async def play_audio(audio_bytes):
    """Plays the received audio data."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True)

    # Use a BytesIO stream to read the audio bytes
    with io.BytesIO(audio_bytes) as audio_stream:
        wave_file = wave.open(audio_stream, "rb")
        data = wave_file.readframes(CHUNK_SIZE)
        while data:
            stream.write(data)
            data = wave_file.readframes(CHUNK_SIZE)

    stream.stop_stream()
    stream.close()
    audio.terminate()

async def send_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    print("Streaming audio to server... Press Ctrl+C to stop.")
    try:
        while True:
            # Read audio chunk from the microphone
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # Send audio chunk to the server
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
    while True:
        # Explicitly receive each message without buffering
        message = await websocket.recv()
        # Check if the message is text or binary data
        if isinstance(message, str):
            # Handle text message
            print(message)
        else:
            # Play the received audio data
            await play_audio(message)

async def main():
    async with websockets.connect(SERVER_URI) as websocket:
        send_task = asyncio.create_task(send_audio(websocket))
        receive_task = asyncio.create_task(receive_audio(websocket))
        await asyncio.gather(send_task, receive_task)

asyncio.run(main())
