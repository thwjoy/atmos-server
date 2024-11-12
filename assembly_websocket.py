from typing import Optional
import assemblyai as aai
import pydub
import asyncio
import websockets
import struct
import os
from assembly_db import SoundAssigner, OPENAI_API_KEY

class AudioServer:
    HEADER_FORMAT = '!5sI'  # 5-byte string (indicator) and 4-byte integer (audio size)

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.music_sent = False  # Track whether a MUSIC track has been sent
        aai.settings.api_key = "09485e2cc7b741d4aa2922da67f84094"
        self.assigner_SFX = SoundAssigner(chroma_path="ESC-50_db")
        self.assigner_Music = SoundAssigner(chroma_path="SA_db")
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

    async def send_audio_in_chunks(self, websocket, audio_bytes, chunk_size=1024 * 1000):
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await websocket.send(chunk)

    async def send_audio_with_header(self, websocket, audio_bytes, indicator):
        # Pack the indicator and audio size into a structured header
        header = struct.pack(self.HEADER_FORMAT, indicator.encode(), len(audio_bytes))
        message = header + audio_bytes  # Combine header and audio bytes
        await self.send_audio_in_chunks(websocket, message)

    async def process_transcript_async(self, transcript, websocket):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            filename, category = self.assigner_SFX.retrieve_src_file(transcript.text)
            audio_bytes = None
            indicator = None

            if filename:  # If SFX file found
                print(f"Sending SFX for category '{category}' to client.")
                audio_bytes = self.get_audio_bytes(os.path.join(filename))
                indicator = "SFX"
            else:  # Default to sending Music if no SFX file found
                filename, category = self.assigner_Music.retrieve_src_file(transcript.text)
                if filename:
                    print(f"Sending MUSIC for category '{category}' to client.")
                    audio_bytes = self.get_audio_bytes(os.path.join(filename))
                    indicator = "MUSIC"

            if audio_bytes:
                header = f"{indicator}:{len(audio_bytes)}".encode('utf-8')
                message = header + audio_bytes  # Combine header with actual audio bytes
                try:
                    await self.send_audio_in_chunks(websocket, message)
                    print("[DEBUG] Audio data sent to client.")
                except Exception as e:
                    print(f"[ERROR] Failed to send audio data: {e}")
            else:
                print("No sound found for the given text.")

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
            end_utterance_silence_threshold=500
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
            try:
                if not self.music_sent:
                    # Send the initial MUSIC track
                    filename, category = self.assigner_Music.retrieve_src_file(message)
                    if filename:
                        print(f"Sending initial MUSIC track for category '{category}' to client.")
                        audio_bytes = self.get_audio_bytes(os.path.join(filename))
                        await self.send_audio_with_header(websocket, audio_bytes, "MUSIC")
                        self.music_sent = True  # Set flag to True after sending MUSIC
                    else:
                        print("No MUSIC found for the given text.")
                else:
                    # Send SFX track for subsequent messages
                    filename, category = self.assigner_SFX.retrieve_src_file(message)
                    if filename:
                        print(f"Sending SFX for category '{category}' to client.")
                        audio_bytes = self.get_audio_bytes(os.path.join(filename))
                        await self.send_audio_with_header(websocket, audio_bytes, "SFX")
                    else:
                        print("No SFX found for the given text.")
            except Exception as e:
                print(f"Error: {e}")

    async def start_server(self):
        server = await websockets.serve(self.txt_reciever, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

    def run(self):
        asyncio.run(self.start_server())

    @staticmethod
    def on_error(error: aai.RealtimeError):
        print("An error occurred:", error)

    @staticmethod
    def on_close():
        print("Closing Session")

# Run the server
if __name__ == "__main__":
    server = AudioServer()
    server.run()
