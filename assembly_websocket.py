
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

from data.assembly_db import SoundAssigner

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

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

# def load_wav_from_bytes(byte_data):
#     # Use soundfile to read from the bytes object
#     with io.BytesIO(byte_data) as bytes_io:
#         audio_data, sample_rate = sf.read(bytes_io, dtype='float32')  # or 'float32' for normalized values
#     return sample_rate, np.array(audio_data)

# def audio_to_bytes(audio_data, sample_rate):
#     # Create an in-memory byte buffer
#     with io.BytesIO() as buffer:
#         # Write the audio data to the buffer as a WAV file
#         sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        
#         # Retrieve the byte data from the buffer
#         audio_bytes = buffer.getvalue()
        
#     return audio_bytes

# def resample_audio(audio, orig_sample_rate, target_sample_rate=44100):
#     # Convert the audio to a numpy array if it's not already
#     if not isinstance(audio, np.ndarray):
#         audio = np.array(audio)
    
#     # Resample the audio to the target sample rate
#     resampled_audio = librosa.resample(audio, orig_sample_rate, target_sample_rate)
#     return resampled_audio

async def read_audio_in_chunks(audio_path, chunk_size=1024 * 1024):
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

async def send_audio_with_header(websocket, audio_path, indicator, chunk_size=1024 * 1024): # TODO fix this
    """Send audio in chunks using an asyncio.Queue."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    file_size = os.path.getsize(audio_path)

    # Handle small files directly
    if file_size <= chunk_size:
        ind = indicator[:5].ljust(5)
        header = struct.pack(AudioServer.HEADER_FORMAT, ind.encode(), file_size, SAMPLE_RATE)
        try:
            audio = pydub.AudioSegment.from_file(audio_path)
            with io.BytesIO() as buffer:
                audio.export(buffer, format="wav")  # Export the audio to a BytesIO buffer
                buffer.seek(0)
                await send_with_backpressure(websocket, header + buffer.read())
                logger.info(f"Sent small file ({file_size} bytes) in a single chunk with header.")
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket closed while sending small file: {e}")
            raise
        return

    # For larger files, use producer-consumer logic
    queue = asyncio.Queue()

    async def producer():
        async for chunk in read_audio_in_chunks(audio_path, chunk_size):
            await queue.put(chunk)
        await queue.put(None)  # Signal completion

    async def consumer():
        # Add header to the first chunk
        ind = indicator[:5].ljust(5)
        header = struct.pack(AudioServer.HEADER_FORMAT, ind.encode(), file_size, SAMPLE_RATE)
        first_chunk = await queue.get()
        if first_chunk is not None:
            try:
                await send_with_backpressure(websocket, header + first_chunk)
            except websockets.ConnectionClosed as e:
                logger.error(f"WebSocket closed during send: {e}")
                raise

        # Send remaining chunks
        while True:
            try:
                chunk = await queue.get()
                if chunk is None:  # Completion signal
                    break
                await send_with_backpressure(websocket, chunk)
            except websockets.ConnectionClosed as e:
                logger.error(f"WebSocket closed during send: {e}")
                raise

    try:
        await asyncio.gather(
            asyncio.create_task(producer()), 
            asyncio.create_task(consumer())
        )
    except Exception as e:
        logger.error(f"Error in send_audio_with_header: {e}")

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

class TranscriberWrapper:
    def __init__(self, **kwargs):
        self.transcriber = aai.RealtimeTranscriber(**kwargs)

    async def connect(self):
        self.transcriber.connect()
        logger.info("Transcriber connected")
        return self

    async def close(self):
        self.transcriber.close()
        logger.info("Transcriber closed")

    def stream(self, data):
        self.transcriber.stream(data)

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

class SharedResources:
    def __init__(self):
        self.assigner_SFX = SoundAssigner(chroma_name="SFX_db", data_root="./data/datasets")
        self.assigner_Music = SoundAssigner(chroma_name="SA_db", data_root="./data/datasets")

shared_resources = SharedResources()

class AudioServer:
    HEADER_FORMAT = '>5sII'  # 5-byte string (indicator) and 4-byte integer (audio size)

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
        self.sfx_score_threshold = 1.0
        self.music_score_threshold = 1.2
        # self.transcript = ""
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

    # def get_next_story_section(self, transcript):
    #     # use ChatGPT to generate the next section of the story
    #     self.transcript += transcript
    #     logger.debug(self.transcript)
    #     chat = self.client.chat.completions.create(
    #         model="gpt-4o",
    #         modalities=["text"],
    #         audio={"voice": "alloy", "format": "wav"},
    #         messages=[
    #             {"role": "system", "content": f"""You are a helpful assistent
    #              who is going to make a story with me. I will start the story
    #              and you will continue it. Once you have writen a few sentences,
    #              I will then take over, and we will keep going until we are finished.
    #              Keep your sections to 2 or 3 sentences maximum.

    #              The story is for children under 10, keep the language simple and the story fun.

    #              Do not repeat the story I have already written. You should make new words.

    #              I also want you to add sounds to each word you have written, where the first element is the 
    #              word and the second element contains None. Do not add any other information
    #              apart from this list. The response should look like this:

    #              <your addition to the story>

    #              output = [('word', None), ..., ('word', None)]

    #              """
    #              },
    #             {
    #                 "role": "user",
    #                 "content": self.transcript
    #             }
    #         ]
    #     )
    #     # logger.debug(f"ChatGPT response: {chat.choices[0].message.content}")
    #     # try get the literal
    #     try:
    #         response = chat.choices[0].message.content
    #         response = response.replace("“", "'").replace("”", "'")
    #         match = re.search(r"\[.*\]", response)
    #         if match:
    #             array_text = match.group(0)
    #             # Safely evaluate the array text as a Python literal
    #             array_literal = ast.literal_eval(array_text)
    #         else:
    #             logger.warning("No array found in the text.")
    #             response = "I didn't catch that, can you try again?".split(" ")
    #             array_literal = [(word, None) for word in response]
    #     except:
    #         response = "I didn't catch that, can you try again?".split(" ")
    #         array_literal = [(word, None) for word in response]

    #     
    #     words = [word for word, sound in array_literal if sound is None]
    #     sounds = [sound for word, sound in array_literal if sound is not None]
    #     logger.debug(f"Sounds: {sounds}")
    #     sentence = " ".join(words)
    #     logger.debug(f"Words: {words}")
    #     # sounds = [sound for word, sound in chat.choices[0].message.content if sound is not None]
    #     # sentence = chat.choices[0].message.content
    #     # sounds = []

    #     audio = self.client.audio.speech.create(
    #         model="tts-1",
    #         voice="alloy",
    #         response_format="wav",
    #         input=sentence,
    #     )
    #     logger.debug(f"Generated narration")
    #     self.transcript += " " + sentence
    #     logger.debug("Transcript: ", self.transcript)

    #     return audio.content, sentence, sounds
    
    # async def send_audio_from_transcript(self, transcript, websocket):
    #     audio, transcript, sounds = self.get_next_story_section(transcript)
    #     # convert the audio bytes into a wavfile and load as numpy
    #     sample_rate, audio = load_wav_from_bytes(audio)
    #     whisper_sample_rate = 16000
    #     audio = resample_audio(audio, sample_rate, whisper_sample_rate)
    #     duration = min(10, np.floor(len(audio) / whisper_sample_rate))
    #     # try:
    #     #     transcriber = RealTimeTranscriber(
    #     #                 book=transcript,
    #     #                 line_offset=0,
    #     #                 chunk_duration=10,
    #     #                 audio_window=10,
    #     #     )
    #     #     transcriber.process_audio_file(audio)
    #     #     audio = self.add_sounds_to_audio(audio, sounds, transcriber.get_df(), whisper_sample_rate)
    #     # except Exception as e:
    #     #     logger.error(f"Error: {e}")
    #     audio = audio_to_bytes(audio, whisper_sample_rate)
    #     logger.info(f"Sending story snippet: {transcript}")
    #     await self.send_audio_with_header(websocket, audio, "STORY", whisper_sample_rate)

    # def add_sounds_to_audio(self, audio, sounds, timestamps, sample_rate):
    #     """ for each sound, find the timestamp in timestamps df and insert into audio array"""
    #     # loop through each row of timestamps and use word column to find the sound
    #     for idx, row in timestamps.iterrows():
    #         # match the sound to the word
    #         filename, category, score = self.assigner_SFX.retrieve_src_file(row['word'])
    #     return audio
    
    async def send_music_from_transcript(self, transcript, websocket):
        try:
            filename, category, score = self.assigner_Music.retrieve_src_file(transcript)
            if score < self.music_score_threshold:
                if filename:
                    logger.info(f"Sending MUSIC track for category '{category}' to client with score: {score}.")
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
            # if len(self.transcript) < 20: # TODO fix this
            #     self.transcript += transcript.text
            #     return
            if not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
                self.music_sent_event.set()
                self.fire_and_forget(self.send_music_from_transcript(transcript.text, websocket))
                # await self.send_audio_from_transcript(transcript.text, websocket)
            else:
                self.fire_and_forget(self.send_sfx_from_transcript(transcript.text, websocket))
                # await self.send_audio_from_transcript(transcript.text, websocket)

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
        logger.info("An error occurred in transcriber:", error)
        self.fire_and_forget(websocket.close())

    def on_close(self, websocket):
        if not websocket.closed:
            self.fire_and_forget(websocket.close())

    async def audio_receiver(self, websocket, path):
        logger.info("Client connected")
        loop = asyncio.get_running_loop()
        
        async with TranscriberWrapper(
            on_data=lambda transcript: self.on_data(transcript, websocket, loop),
            on_error=lambda error : self.on_error(error, websocket),
            sample_rate=44_100,
            on_open=lambda session: self.on_open(session, websocket, loop),
            on_close=lambda : self.on_close(websocket), # why is this self?
            end_utterance_silence_threshold=500
        ) as transcriber:
            try:
                self.fire_and_forget(self.send_music_from_transcript("jungle", websocket))
            except Exception as e:
                logger.error(f"Error sending files: {e}")

            await asyncio.sleep(5)

            try:
                self.fire_and_forget(self.send_sfx_from_transcript("horse", websocket))
            except Exception as e:
                logger.error(f"Error sending files: {e}")
            set_session_id(self.session_id)
            async for message in websocket:
                transcriber.stream([message])



    # async def txt_reciever(self, websocket, path):
    #     async for message in websocket:
    #         logger.info(f"Received message: {message}")
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
        # Get the client IP address
        client_ip = websocket.remote_address[0]

        # Check rate limits
        if is_rate_limited_ip(client_ip):
            await websocket.close(code=4290, reason="Rate limit exceeded")
            logger.warning(f"Connection rejected: Rate limit exceeded for IP {client_ip}")
            return
    
        # Extract the token from the headers
        token = websocket.request_headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            await websocket.close(code=4001, reason="Unauthorized: Missing token")
            logger.warning("Connection rejected: Missing token")
            return
        
        # Verify the token
        # Validate the token asynchronously
        try:
            user_id = await validate_token(token)
            logger.info(f"Connection accepted for user {user_id}")
        except TokenValidationError as e:
            await websocket.close(code=4003, reason=str(e))
            logger.warning(f"Connection rejected: {str(e)}")
            return
        
        # Handle communication after successful authentication
        try:
            server_instance = AudioServer(user_id)  # Create a new instance for each connection
            await server_instance.audio_receiver(websocket, path)
        except websockets.ConnectionClosed as e:
            logger.error(f"Connection closed: {e}")
        except websockets.ConnectionClosedError as e:
            logger.error(f"Connection closed abruptly: {e}")
        except websockets.ConnectionClosedOK:
            logger.error("Connection closed gracefully.")
        except Exception as e:
            logger.error(f"Closing websocket: {e}")
        finally:
            await server_instance.close_all_tasks()
            await websocket.close()


    @staticmethod
    async def start_server(host="0.0.0.0", port=8765):
        logger.info(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(AudioServer.connection_handler, host, port, ping_interval=300, ping_timeout=300):
            await asyncio.Future()  # Run forever

# Server entry point
def run():
    asyncio.run(AudioServer.start_server())


# Run the server if the script is executed directly
if __name__ == "__main__":
    logger = configure_logging()
    run()
