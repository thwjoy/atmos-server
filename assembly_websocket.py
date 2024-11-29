
import io
import assemblyai as aai
import pydub
import asyncio
import websockets
import struct
import os
import jwt
from contextvars import ContextVar
import logging
import ssl
import threading
import sqlite3
import json
from datetime import datetime
import uuid
import soundfile as sf
import librosa

from data.assembly_db import SoundAssigner
from openai import OpenAI
import numpy as np

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# certfile = "/root/.ssh/myatmos_pro_chain.crt"
# keyfile = "/root/.ssh/myatmos.key"

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
ssl_context = None

HEADER_FORMAT = ">5sI16sIII"

# Create ContextVars for user_id and session_id
user_id_context: ContextVar[str] = ContextVar("user_id", default="None")
session_id_context: ContextVar[str] = ContextVar("session_id", default="None")

# Filter to inject user_id and session_id into every log record
class ContextFilter(logging.Filter):
    def filter(self, record):
        # Add user_id and session_id to the log record
        record.user_id = user_id_context.get("None")
        record.session_id = session_id_context.get("None")
        return True

# Custom Formatter to handle missing fields gracefully
class SafeFormatter(logging.Formatter):
    def format(self, record):
        # Ensure user_id and session_id exist in the record
        if not hasattr(record, "user_id"):
            record.user_id = "None"
        if not hasattr(record, "session_id"):
            record.session_id = "None"
        return super().format(record)

# Configure logging
def configure_logging():
    formatter = SafeFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - user_id=%(user_id)s - session_id=%(session_id)s - %(message)s"
    )    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Add the UserIDFilter to the root logger
    root_logger.addFilter(ContextFilter())
    return root_logger

# Set user_id in the ContextVar
def set_user_id(user_id: str):
    user_id_context.set(user_id)

# Set session_id in the ContextVar
def set_session_id(session_id: str):
    session_id_context.set(session_id)

# Reset user_id and session_id to their defaults
def reset_context():
    user_id_context.set("None")
    session_id_context.set("None")

RATE_LIMIT = 3  # Max 10 connections per IP/USER
RATE_LIMIT_WINDOW = 60  # In seconds
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

async def read_audio_in_chunks(audio_path, chunk_size):
        """Read an audio file in chunks asynchronously."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
    
        loop = asyncio.get_running_loop()
        with open(audio_path, "rb") as audio_file:
            while chunk := await loop.run_in_executor(None, audio_file.read, chunk_size):
                yield chunk

async def send_with_backpressure(websocket, data, buffer_limit=10_000_000):
    """Send data to the WebSocket with backpressure handling."""
    while websocket.transport.get_write_buffer_size() > buffer_limit:
        await asyncio.sleep(0.1)  # Wait for the buffer to drain
    await websocket.send(data)

async def send_audio_with_header(websocket, audio_path, indicator, chunk_size=1024 * 512):
    """Send audio in chunks with header containing packet size, packet count, sample rate, and unique sequence ID."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    total_packets = (file_size + chunk_size - 1) // chunk_size  # Calculate total packet count

    # Retrieve sample rate (assume WAV format)
    sample_rate = 44100

    # Generate a unique sequence ID for this transmission
    sequence_id = uuid.uuid4().bytes  # UUID as a 32-byte binary string

    queue = asyncio.Queue()

    async def producer():
        packet_count = 0
        async for chunk in read_audio_in_chunks(audio_path, chunk_size):
            await queue.put((packet_count, chunk))  # Add packet count with chunk
            packet_count += 1
        await queue.put(None)  # Signal completion

    async def consumer():
        packet_sent = 0

        while True:
            try:
                data = await queue.get()
                if data is None:  # Completion signal
                    break

                packet_count, chunk = data

                # Build the header
                ind = indicator[:5].ljust(5)  # Ensure the indicator is 5 bytes
                header = struct.pack(
                    HEADER_FORMAT,       # Updated format
                    ind.encode(),        # Indicator
                    file_size,          # Size of this packet
                    sequence_id,         # Unique sequence ID
                    packet_count,        # Packet count (sequence number)
                    total_packets,       # Total packets
                    sample_rate          # Sample rate
                )

                # Send the chunk with the header
                await send_with_backpressure(websocket, header + chunk)
                packet_sent += 1

            except websockets.ConnectionClosed as e:
                logger.error(f"WebSocket closed during send: {e}")
                raise

    try:
        await asyncio.gather(
            asyncio.create_task(producer()),
            asyncio.create_task(consumer())
        )
        logger.info(f"Successfully sent packets with Sequence ID: {uuid.UUID(bytes=sequence_id)}.")
    except Exception as e:
        logger.error(f"Error in send_audio_with_header: {e}")

async def send_transcript_audio_with_header(websocket, audio, sample_rate, chunk_size=1024 * 512):
    ind = "STORY"
    file_size = len(audio)
    sequence_id = uuid.uuid4().bytes  # UUID as a 32-byte binary string
    packet_count = 0
    total_packets = (file_size + chunk_size - 1) // chunk_size  # Calculate total packet count
    # Divide the audio into chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        # Add a header with the chunk size
        header = struct.pack(
            HEADER_FORMAT,       # Updated format
            ind.encode(),        # Indicator
            file_size,          # Size of this packet
            sequence_id,         # Unique sequence ID
            packet_count,        # Packet count (sequence number)
            total_packets,       # Total packets
            sample_rate          # Sample rate
        )
        packet_count += 1
        data_with_header = header + chunk
        await send_with_backpressure(websocket, data_with_header)

class TokenValidationError(Exception):
    """Custom exception for token validation failures."""
    pass

async def validate_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id")

        if is_rate_limited_user(user_id):
            raise TokenValidationError("Rate limit exceeded for user")
        
        return user_id
    except jwt.ExpiredSignatureError:
        raise TokenValidationError("Token expired")
    except jwt.InvalidTokenError as e:
        raise TokenValidationError(f"Invalid token: {str(e)}")

import time

connection_attempts_ip = {}

def is_rate_limited_ip(ip: str) -> bool:
    current_time = time.time()
    if ip not in connection_attempts_ip:
        connection_attempts_ip[ip] = [current_time]
        return False
    
    # Filter out old attempts
    connection_attempts_ip[ip] = [
        ts for ts in connection_attempts_ip[ip] if current_time - ts < RATE_LIMIT_WINDOW
    ]
    
    # Add the current attempt
    connection_attempts_ip[ip].append(current_time)
    
    # Check if rate limit is exceeded
    return len(connection_attempts_ip[ip]) > RATE_LIMIT

connection_attempts_user = {}

def is_rate_limited_user(user_id: str) -> bool:
    current_time = time.time()
    if user_id not in connection_attempts_user:
        connection_attempts_user[user_id] = [current_time]
        return False
    
    # Filter out old attempts
    connection_attempts_user[user_id] = [
        ts for ts in connection_attempts_user[user_id] if current_time - ts < RATE_LIMIT_WINDOW
    ]
    
    # Add the current attempt
    connection_attempts_user[user_id].append(current_time)
    
    # Check if rate limit is exceeded
    return len(connection_attempts_user[user_id]) > RATE_LIMIT

async def monitored_task(coro, name="Unnamed Task"):
    try:
        await coro
    except Exception as e:
        logger.error(f"Error in task {name}: {e}")

# class TranscriberWrapper:
#     def __init__(self, **kwargs):
#         self.transcriber = aai.RealtimeTranscriber(**kwargs)

#     async def connect(self):
#         self.transcriber.connect()
#         logger.info("Transcriber connected")
#         return self

#     async def close(self):
#         self.transcriber.close()
#         logger.info("Transcriber closed")

#     def stream(self, data):
#         self.transcriber.stream(data)

#     async def __aenter__(self):
#         return await self.connect()

#     async def __aexit__(self, exc_type, exc_value, traceback):
#         await self.close()


class DatabaseManager:
    def __init__(self, db_path="database.db"):
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage

    def connect(self):
        """Get a thread-local connection."""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
        return self.local.connection

    def initialize(self):
        """Create tables if they don't exist."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    connection_id TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            ''')
            conn.commit()
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    stop_time TEXT
                )
            ''')
            conn.commit()

    def insert_transcript_data(self, connection_id, data):
        """Insert data for a specific connection."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcripts (connection_id, data) VALUES (?, ?)",
                (str(connection_id), json.dumps(data))
            )
            conn.commit()

    def log_session_start(self, user_id):
        """Log the start of a session."""
        start_time = datetime.utcnow().isoformat()  # Use ISO 8601 format
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (user_id, start_time) VALUES (?, ?)",
                (user_id, start_time)
            )
            conn.commit()
            return cursor.lastrowid  # Return the session ID for reference


    def log_session_stop(self, session_id):
        """Log the stop of a session."""
        stop_time = datetime.utcnow().isoformat()
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET stop_time = ? WHERE id = ?",
                (stop_time, session_id)
            )
            conn.commit()

class SharedResources:
    def __init__(self):
        self.assigner_SFX = SoundAssigner(chroma_name="SFX_db", data_root="./data/datasets")
        self.assigner_Music = SoundAssigner(chroma_name="SA_db", data_root="./data/datasets")
        self.openai = OpenAI()

shared_resources = SharedResources()

class AudioServer:

    def __init__(self, user_id='-1', host="0.0.0.0", port=8765):
        self.user_id = user_id
        self.session_id = '-1'
        set_user_id(user_id)
        self.host = host
        self.port = port
        self.music_sent_event = asyncio.Event()  # Track whether a MUSIC track has been sent
        self.tasks = []  # Track active tasks
        aai.settings.api_key = "09485e2cc7b741d4aa2922da67f84094"
        self.assigner_SFX = shared_resources.assigner_SFX
        self.assigner_Music = shared_resources.assigner_Music
        self.client = shared_resources.openai
        self.sfx_score_threshold = 1.2
        self.music_score_threshold = 1.2
        self.narration_transcript = ""
        self.transcript = {
            "transcript": [],
            "sounds": [],
            "score": []
        }
        # self.story = ""
        # self.client = OpenAI()

    def fire_and_forget(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(task))  # Clean up task when done
        return task

    async def close_all_tasks(self):
        """Cancel all active tasks and ensure they complete."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info(f"All tasks have been canceled and cleaned up.")

    def get_next_story_section(self, transcript):
        # use ChatGPT to generate the next section of the story
        self.narration_transcript += transcript
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

                 Only output the story, do not include any other information.

                 """
                 },
                {
                    "role": "user",
                    "content": self.narration_transcript
                }
            ]
        )
        # logger.debug(f"ChatGPT response: {chat.choices[0].message.content}")
        # try get the literal
        try:
            response = chat.choices[0].message.content
            # response = response.replace("“", "'").replace("”", "'")
            # match = re.search(r"\[.*\]", response)
            # if match:
            #     text = match.group(0)
                # Safely evaluate the array text as a Python literal
                # array_literal = ast.literal_eval(array_text)
            # else:
            #     logger.warning("No array found in the text.")
            #     text = "I didn't catch that, can you try again?".split(" ")
                # array_literal = [(word, None) for word in response]
        except:
            text = "I didn't catch that, can you try again?".split(" ")
            # array_literal = [(word, None) for word in response]

        
        # words = [word for word, sound in array_literal if sound is None]
        # sounds = [sound for word, sound in array_literal if sound is not None]
        # logger.debug(f"Sounds: {sounds}")
        # sentence = " ".join(words)
        # logger.debug(f"Words: {words}")
        # sounds = [sound for word, sound in chat.choices[0].message.content if sound is not None]
        # sentence = chat.choices[0].message.content
        # sounds = []
        sentence = response

        audio = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            response_format="wav",
            input=sentence,
        )
        logger.debug(f"Generated narration")
        self.narration_transcript += " " + sentence
        logger.info(f"Narration so far: {self.narration_transcript}")

        return audio.content, sentence, None
    
    async def send_audio_from_transcript(self, transcript, websocket):
        audio, transcript, sounds = self.get_next_story_section(transcript)
        logger.info(f"Sending story snippet: {transcript}")
        await send_transcript_audio_with_header(websocket, audio, 24000)


        # whisper_sample_rate = 16000
        # audio = resample_audio(audio, sample_rate, whisper_sample_rate)
        # duration = min(10, np.floor(len(audio) / whisper_sample_rate))
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
        #     logger.error(f"Error: {e}")
        # audio = audio_to_bytes(audio, whisper_sample_rate)

    # def add_sounds_to_audio(self, audio, sounds, timestamps, sample_rate):
    #     """ for each sound, find the timestamp in timestamps df and insert into audio array"""
    #     # loop through each row of timestamps and use word column to find the sound
    #     for idx, row in timestamps.iterrows():
    #         # match the sound to the word
    #         filename, category, score = self.assigner_SFX.retrieve_src_file(row['word'])
    #     return audio

    def insert_transcript_section(self, transcript, sounds, score):
        self.transcript["transcript"].append(transcript),
        self.transcript["sounds"].append(sounds),
        self.transcript["score"].append(score)

    
    async def send_music_from_transcript(self, transcript, websocket):
        try:
            filename, category, score = self.assigner_Music.retrieve_src_file(transcript)
            if score < self.music_score_threshold:
                if filename:
                    logger.info(f"Sending MUSIC track for category '{category}' to client with score: {score}.")
                    self.insert_transcript_section(transcript, filename, score)
                    await send_audio_with_header(websocket, os.path.join(filename), "MUSIC")
                else:
                    logger.info("No MUSIC found for the given text.")
            else:
                logger.warning(f"Not sending audio for category '{category}' to client with score: {score}.")
                self.music_sent_event.clear()
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket closed during send_music_from_transcript: {e}")
        except FileNotFoundError as e:
            logger.error(f"File not found during send_music_from_transcript: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in send_music_from_transcript: {e}")

    async def send_sfx_from_transcript(self, transcript, websocket):
        try:
            filename, category, score = self.assigner_SFX.retrieve_src_file(transcript)
            if score < self.sfx_score_threshold:
                if filename:
                    logger.info(f"Sending SFX for category '{category}' to client with score: {score}.")
                    self.insert_transcript_section(transcript, filename, score)
                    await send_audio_with_header(websocket, os.path.join(filename), "SFX")
                else:
                    logger.info("No SFX found for the given text.")
            else:
                logger.warning(f"Not sending audio for category '{category}' to client with score: {score}.")
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket closed during send_sfx_from_transcript: {e}")
        except FileNotFoundError as e:
            logger.error(f"File not found during send_sfx_from_transcript: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in send_sfx_from_transcript: {e}")

    async def process_transcript_async(self, transcript, websocket):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            logger.info(f"Recieved: {transcript.text}")
            # if len(self.narration_transcript) < 20: # TODO fix this
            #     self.narration_transcript += transcript.text
            #     return
            if not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
                self.music_sent_event.set()
                self.fire_and_forget(self.send_music_from_transcript(transcript.text, websocket))
                # self.fire_and_forget(self.send_audio_from_transcript(transcript.text, websocket))
            else:
                self.fire_and_forget(self.send_sfx_from_transcript(transcript.text, websocket))
                # self.fire_and_forget(self.send_audio_from_transcript(transcript.text, websocket))

    async def send_message_async(self, message, websocket):
        await asyncio.create_task(websocket.send(message))
        logger.info(f"Message sent: {message}")

    def on_data(self, transcript, websocket, loop):
        asyncio.run_coroutine_threadsafe(self.process_transcript_async(transcript, websocket), loop)

    def on_open(self, session, websocket, loop):
        set_user_id(self.user_id)
        logger.info(f"Transcriber session ID: {session.session_id}")
        self.session_id = session.session_id
        set_session_id(session.session_id)
        asyncio.run_coroutine_threadsafe(self.send_message_async(str(session.session_id), websocket), loop)
    
    def on_error(self, error: aai.RealtimeError, websocket):
        logger.error(f"An error occurred in transcriber: {error}")

    def on_close(self, websocket):
        try:
            db_manager.insert_transcript_data(self.session_id, self.transcript)
        except Exception as e:
            logger.error(f"Failed to save transcript to db: {str(e)}")

    async def audio_receiver(self, websocket):
        logger.info("Client connected")
        loop = asyncio.get_running_loop()
        
        try:
            transcriber = aai.RealtimeTranscriber(
                on_data=lambda transcript: self.on_data(transcript, websocket, loop),
                on_error=lambda error : self.on_error(error, websocket),
                sample_rate=44_100,
                on_open=lambda session: self.on_open(session, websocket, loop),
                on_close=lambda : self.on_close(websocket), # why is this self?
                end_utterance_silence_threshold=100
            )
            # Start the connection
            transcriber.connect()

            # try:
            #     self.fire_and_forget(self.send_music_from_transcript("jungle", websocket))
            # except Exception as e:
            #     logger.error(f"Error sending files: {e}")

            # await asyncio.sleep(5)

            # try:
            #     self.fire_and_forget(self.send_sfx_from_transcript("horse", websocket))
            # except Exception as e:
            #     logger.error(f"Error sending files: {e}")
            set_session_id(self.session_id)
            while True:
                try:
                    message = await websocket.recv()
                    transcriber.stream([message])
                except Exception as e:
                    logger.error(f"Closing: {e}")
                    break  # Exit the loop if an error occurs
        finally:
            try:
                await asyncio.wait_for(asyncio.to_thread(transcriber.close), timeout=5) # TODO this is probably a bug... 
            except asyncio.TimeoutError:
                logger.error("transcriber.close() timed out. Proceeding with cleanup.")

    @staticmethod
    async def connection_handler(websocket):
        # Get the client IP address
        client_ip = websocket.remote_address[0]

        # # Check rate limits
        if is_rate_limited_ip(client_ip):
            await websocket.close(code=4290, reason="Rate limit exceeded")
            logger.warning(f"Connection rejected: Rate limit exceeded for IP {client_ip}")
            return
    
        # Extract the token from the headers
        try:
            token = websocket.request_headers.get("Authorization", "").replace("Bearer ", "")
            if not token:
                await websocket.close(code=4001, reason="Unauthorized: Missing token")
                logger.warning("Connection rejected: Missing token")
                return
        except Exception as e:
            logger.error(f"Exception {e}")
            return
        
        # Verify the token
        # Validate the token asynchronously
        try:
            user_id = await validate_token(token)
            logger.info(f"Connection accepted for user {user_id}")
            db_session = db_manager.log_session_start(user_id)
        except TokenValidationError as e:
            await websocket.close(code=4003, reason=str(e))
            logger.warning(f"Connection rejected: {str(e)}")
            return
        
        # Handle communication after successful authentication
        try:
            server_instance = AudioServer(user_id)  # Create a new instance for each connection
            await server_instance.audio_receiver(websocket)
        except websockets.ConnectionClosed as e:
            logger.error(f"Connection closed: {e}")
        except websockets.ConnectionClosedError as e:
            logger.error(f"Connection closed abruptly: {e}")
        except websockets.ConnectionClosedOK:
            logger.error("Connection closed gracefully.")
        except Exception as e:
            logger.error(f"Closing websocket: {e}")
        finally:
            db_manager.log_session_stop(db_session)
            await server_instance.close_all_tasks()
            await websocket.close()


    @staticmethod
    async def start_server(host="localhost", port=5001):
        logger.info(f"Starting WebSocket server on wss://{host}:{port}")
        async with websockets.serve(AudioServer.connection_handler, host, port, ping_interval=300, ping_timeout=300, ssl=ssl_context):
            await asyncio.Future()  # Run forever

# Server entry point
def run():
    asyncio.run(AudioServer.start_server())


# Run the server if the script is executed directly
if __name__ == "__main__":
    logger = configure_logging()
    # Create a global instance of DatabaseManager
    db_manager = DatabaseManager()
    db_manager.initialize()  # Ensure tables are created
    run()
