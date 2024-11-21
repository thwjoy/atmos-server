import ast
import io
import re
from typing import Optional
import assemblyai as aai
import pydub
import asyncio
import websockets
import struct
import os
from whisper_realtime import RealTimeTranscriber
from data.assembly_db import SoundAssigner
from openai import OpenAI
import soundfile as sf
from keys import OPENAI_API_KEY

import librosa
import numpy as np


SAMPLE_RATE = 44100

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_wav_from_bytes(byte_data):
    # Use soundfile to read from the bytes object
    with io.BytesIO(byte_data) as bytes_io:
        audio_data, sample_rate = sf.read(bytes_io, dtype='float32')  # or 'float32' for normalized values
    return sample_rate, np.array(audio_data)

def audio_to_bytes(audio_data, sample_rate):
    # Create an in-memory byte buffer
    with io.BytesIO() as buffer:
        # Write the audio data to the buffer as a WAV file
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        
        # Retrieve the byte data from the buffer
        audio_bytes = buffer.getvalue()
        
    return audio_bytes

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
        self.assigner_SFX = SoundAssigner(chroma_name="SFX_db", data_root="./data/datasets")
        self.assigner_Music = SoundAssigner(chroma_name="SA_db", data_root="./data/datasets")
        self.sfx_score_threshold = 1.0
        self.music_score_threshold = 1.2
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
        
    async def get_audio_bytes_async(self, audio_path):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_audio_bytes, audio_path)
                       
    def get_next_story_section(self, transcript):
        # use ChatGPT to generate the next section of the story
        self.transcript += transcript
        print(self.transcript)
        chat = self.client.chat.completions.create(
            model="gpt-4o",
            modalities=["text"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {"role": "system", "content": f"""You are a helpful assistent
                 who is going to make a story with me. I will start the story
                 and you will continue it. Once you have writen a few sentences,
                 I will then take over, and we will keep going until we are finished.
                 Keep your sections to 2 or 3 sentences maximum.

                 The story is for children under 10, keep the language simple and the story fun.

                 Do not repeat the story I have already written. You should make new words.

                 I also want you to add sounds to each word you have written, where the first element is the 
                 word and the second element contains None. Do not add any other information
                 apart from this list. The response should look like this:

                 <your addition to the story>

                 output = [('word', None), ..., ('word', None)]

                 """
                 },
                {
                    "role": "user",
                    "content": self.transcript
                }
            ]
        )
        # print(f"ChatGPT response: {chat.choices[0].message.content}")
        # try get the literal
        try:
            response = chat.choices[0].message.content
            response = response.replace("“", "'").replace("”", "'")
            match = re.search(r"\[.*\]", response)
            if match:
                array_text = match.group(0)
                # Safely evaluate the array text as a Python literal
                array_literal = ast.literal_eval(array_text)
            else:
                print("No array found in the text.")
                response = "I didn't catch that, can you try again?".split(" ")
                array_literal = [(word, None) for word in response]
        except:
            response = "I didn't catch that, can you try again?".split(" ")
            array_literal = [(word, None) for word in response]

        # print(f"array_literal: {array_literal}")
        words = [word for word, sound in array_literal if sound is None]
        sounds = [sound for word, sound in array_literal if sound is not None]
        print(f"Sounds: {sounds}")
        sentence = " ".join(words)
        print(f"Words: {words}")
        # sounds = [sound for word, sound in chat.choices[0].message.content if sound is not None]
        # sentence = chat.choices[0].message.content
        # sounds = []

        audio = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            response_format="wav",
            input=sentence,
        )
        print(f"Generated narration")
        self.transcript += " " + sentence
        print("Transcript: ", self.transcript)

        return audio.content, sentence, sounds


    async def send_audio_in_chunks(self, websocket, audio_bytes, header, chunk_size=1024 * 1000):
        # Send the first chunk: header + initial audio chunk
        first_chunk = header + audio_bytes[:chunk_size]
        await websocket.send(first_chunk)
        # Send the remaining audio bytes in chunks
        for i in range(chunk_size, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await websocket.send(chunk)

    async def send_audio_with_header(self, websocket, audio_bytes, indicator, sample_rate=SAMPLE_RATE):
        # Pack the indicator and audio size into a structured header
        indicator = indicator[:5].ljust(5)
        header = struct.pack(self.HEADER_FORMAT, indicator.encode(), len(audio_bytes), sample_rate)

        # Send the audio in chunks, starting with the header
        await self.send_audio_in_chunks(websocket, audio_bytes, header)

    async def send_music_from_transcript(self, transcript, websocket):
        try:
            filename, category, score = self.assigner_Music.retrieve_src_file(transcript)
            if score < self.music_score_threshold:
                if filename:
                    print(f"Sending MUSIC track for category '{category}' to client with score: {score}.")
                    audio_bytes = await self.get_audio_bytes_async(os.path.join(filename))
                    print("Read MUSIC")
                    await self.send_audio_with_header(websocket, audio_bytes, "MUSIC")
                    print("Sent MUSIC track.")
                else:
                    print("No MUSIC found for the given text.")
            else:
                print(f"Not sending audio for category '{category}' to client with score: {score}.")
                self.music_sent_event.clear()
        except Exception as e:
            print(f"Error: {e}")

    async def send_sfx_from_transcript(self, transcript, websocket):
        try:
            filename, category, score = self.assigner_SFX.retrieve_src_file(transcript)
            if score < self.sfx_score_threshold:
                if filename:
                    print(f"Sending SFX for category '{category}' to client with score: {score}.")
                    audio_bytes = await self.get_audio_bytes_async(os.path.join(filename))
                    print("Read SFX")
                    await self.send_audio_with_header(websocket, audio_bytes, "SFX")
                    print("Sent SFX.")
                else:
                    print("No SFX found for the given text.")
            else:
                print(f"Not sending audio for category '{category}' to client with score: {score}.")
        except Exception as e:
            print(f"Error: {e}")

    def add_sounds_to_audio(self, audio, sounds, timestamps, sample_rate):
        """ for each sound, find the timestamp in timestamps df and insert into audio array"""
        # loop through each row of timestamps and use word column to find the sound
        for idx, row in timestamps.iterrows():
            # match the sound to the word
            filename, category, score = self.assigner_SFX.retrieve_src_file(row['word'])
        return audio


    async def send_audio_from_transcript(self, transcript, websocket):
        audio, transcript, sounds = self.get_next_story_section(transcript)
        # convert the audio bytes into a wavfile and load as numpy
        sample_rate, audio = load_wav_from_bytes(audio)
        whisper_sample_rate = 16000
        audio = resample_audio(audio, sample_rate, whisper_sample_rate)
        duration = min(10, np.floor(len(audio) / whisper_sample_rate))
        # try:
        #     transcriber = RealTimeTranscriber(
        #                 book=transcript,
        #                 line_offset=0,
        #                 chunk_duration=10,
        #                 audio_window=10,
        #     )
        #     transcriber.process_audio_file(audio)
        #     audio = self.add_sounds_to_audio(audio, sounds, transcriber.get_df(), whisper_sample_rate)
        # except Exception as e:
        #     print(f"Error: {e}")
        audio = audio_to_bytes(audio, whisper_sample_rate)
        print(f"Sending story snippet: {transcript}")
        await self.send_audio_with_header(websocket, audio, "STORY", whisper_sample_rate)


    async def process_transcript_async(self, transcript, websocket):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(f"Recieved: {transcript.text}")
            # if len(self.transcript) < 20: # TODO fix this
            #     self.transcript += transcript.text
            #     return
            if not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
                self.music_sent_event.set()
                asyncio.create_task(self.send_music_from_transcript(transcript.text, websocket))
                # await self.send_audio_from_transcript(transcript.text, websocket)
            else:
                asyncio.create_task(self.send_sfx_from_transcript(transcript.text, websocket))
                # await self.send_audio_from_transcript(transcript.text, websocket)


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
        except websockets.ConnectionClosedError as e:
            print(f"Client disconnected abruptly: {e}")
        except websockets.ConnectionClosedOK:
            print("Client disconnected gracefully.")
        finally:
            transcriber.close()
            print("Cleaned up resources after client disconnection")
            # Send a proper close frame to the client
            try:
                await websocket.close()
                print("WebSocket closed with close frame.")
            except Exception as e:
                print(f"Error sending close frame: {e}")

    # async def txt_reciever(self, websocket, path):
    #     async for message in websocket:
    #         print(f"Received message: {message}")
    #         # if len(self.transcript) < 20: # TODO fix this
    #         #     self.transcript += transcript.text
    #         #     return
    #         if not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
    #             self.music_sent_event.set()
    #             await self.send_music_from_transcript(message, websocket)
    #             # await self.send_audio_from_transcript(transcript.text, websocket)
    #         else:
    #             await self.send_sfx_from_transcript(message, websocket)
    #             # await self.send_audio_from_transcript(transcript.text, websocket)

    @staticmethod
    async def connection_handler(websocket, path):
        """Handle a new client connection by creating an instance of AudioServer."""
        server_instance = AudioServer()  # Create a new instance for each connection
        await server_instance.audio_receiver(websocket, path)

    @staticmethod
    async def start_server(host="0.0.0.0", port=8765):
        print(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(AudioServer.connection_handler, host, port, ping_interval=300, ping_timeout=300):
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
