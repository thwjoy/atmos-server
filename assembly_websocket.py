import base64
from typing import Optional
import assemblyai as aai
import pydub
import asyncio
import websockets
import struct
import os
from assembly_db import SoundAssigner, OPENAI_API_KEY
from openai import OpenAI

import librosa
import numpy as np


SAMPLE_RATE = 44100

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def resample_audio(audio, orig_sample_rate, target_sample_rate=44100):
    # Convert the audio to a numpy array if it's not already
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    
    # Resample the audio to the target sample rate
    resampled_audio = librosa.resample(audio, orig_sample_rate, target_sample_rate)
    return resampled_audio

class AudioServer:
    HEADER_FORMAT = '>5sII'  # 5-byte string (indicator) and 4-byte integer (audio size)

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.music_sent_event = asyncio.Event()  # Track whether a MUSIC track has been sent
        self.audio_lock = asyncio.Lock()  # Lock to prevent simultaneous audio sends
        aai.settings.api_key = "09485e2cc7b741d4aa2922da67f84094"
        self.assigner_SFX = SoundAssigner(chroma_path="ESC-50_db")
        self.assigner_Music = SoundAssigner(chroma_path="SA_db")
        self.transcript = ""
        self.story = ""
        self.client = OpenAI()
        self.all_sounds = [
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

    def get_audio_bytes(self, audio_path):
        pydub.AudioSegment.from_file(audio_path).export("temp.wav", format="wav")
        with open("temp.wav", "rb") as audio_file:
            return audio_file.read()
        
    def get_next_story_section(self, transcript):
        # use ChatGPT to generate the next section of the story
        completion = self.client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {"role": "system", "content": """You are a helpful assistent
                 who is going to make a story with me. I will start the story
                 and you will continue it. Once you are done, I will continue.
                 The story should be around 3 minutes long and is for children."""},
                {
                    "role": "user",
                    "content": transcript
                }
            ]
        )

        return completion.choices[0].message.audio.data, completion.choices[0].message.audio.transcript


    async def send_audio_in_chunks(self, websocket, audio_bytes, chunk_size=1024 * 1000):
        async with self.audio_lock:  # Ensure only one audio is sent at a time
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                await websocket.send(chunk)

    async def send_audio_with_header(self, websocket, audio_bytes, indicator, sample_rate=SAMPLE_RATE):
        # Pack the indicator and audio size into a structured header
        indicator = indicator[:5].ljust(5)
        header = struct.pack(self.HEADER_FORMAT, indicator.encode(), len(audio_bytes), sample_rate)
        message = header + audio_bytes  # Combine header and audio bytes
        await self.send_audio_in_chunks(websocket, message)

    async def send_audio_from_transcript(self, transcript, websocket):
        try:
            if not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
                self.music_sent_event.set()
                # Send the initial MUSIC track
                filename, category, score = self.assigner_Music.retrieve_src_file(transcript)
                if filename:
                    print(f"Sending initial MUSIC track for category '{category}' to client with score: {score} from {self.transcript}.")
                    audio_bytes = self.get_audio_bytes(os.path.join(filename))
                    await self.send_audio_with_header(websocket, audio_bytes, "MUSIC")
                    self.music_sent_event.set()  # Set flag to True after sending MUSIC
                else:
                    print("No MUSIC found for the given text.")
            else:
                # Send SFX track for subsequent messages
                filename, category, score = self.assigner_SFX.retrieve_src_file(transcript)
                if filename:
                    print(f"Sending SFX for category '{category}' to client from {transcript}.")
                    audio_bytes = self.get_audio_bytes(os.path.join(filename))
                    await self.send_audio_with_header(websocket, audio_bytes, "SFX")
                else:
                    print("No SFX found for the given text.")
        except Exception as e:
            print(f"Error: {e}")

    async def process_transcript_async(self, transcript, websocket):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(f"Recieved: {transcript.text}")
            # if len(self.transcript) < 20: # TODO fix this
            #     self.transcript += transcript.text
            #     return
            await self.send_audio_from_transcript(transcript.text, websocket)
            # send an audio snippet of a story
            audio, transcript = self.get_next_story_section(transcript.text)
            await self.send_audio_with_header(websocket, base64.b64decode(audio), "STORY", 24000)


    def on_data(self, transcript, websocket, loop):
        asyncio.run_coroutine_threadsafe(self.process_transcript_async(transcript, websocket), loop)

    async def audio_receiver(self, websocket, path):
        print("Client connected")
        loop = asyncio.get_running_loop()
        
        transcriber = aai.RealtimeTranscriber(
            on_data=lambda transcript: self.on_data(transcript, websocket, loop),
            on_error=self.on_error,
            sample_rate=44_100,
            on_open=lambda session: print("Session ID:", session.session_id),
            on_close=self.on_close,
            word_boost=self.all_sounds,
            end_utterance_silence_threshold=1000
        )

        try:
            transcriber.connect()
            async for message in websocket:
                transcriber.stream([message])
        except websockets.ConnectionClosed as e:
            print(f"Client disconnected: {e}")
        finally:
            transcriber.close()

    async def txt_reciever(self, websocket, path):
        async for message in websocket:
            print(f"Received message: {message}")
            await self.send_audio_from_transcript(message, websocket)

    @staticmethod
    async def connection_handler(websocket, path):
        """Handle a new client connection by creating an instance of AudioServer."""
        server_instance = AudioServer()  # Create a new instance for each connection
        await server_instance.audio_receiver(websocket, path)

    @staticmethod
    async def start_server(host="0.0.0.0", port=8765):
        print(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(AudioServer.connection_handler, host, port):
            await asyncio.Future()  # Run forever

    @staticmethod
    def on_error(error: aai.RealtimeError):
        print("An error occurred:", error)

    @staticmethod
    def on_close():
        print("Closing Session")


# Server entry point
def run():
    asyncio.run(AudioServer.start_server())


# Run the server if the script is executed directly
if __name__ == "__main__":
    run()
