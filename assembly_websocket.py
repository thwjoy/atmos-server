
import asyncio
import websockets
import os
import time
import base64
import uuid
import struct
import random


import assemblyai as aai
aai.settings.http_timeout = 30.0
from openai import AsyncOpenAI
import numpy as np
import ssl


from utils.session_utils import (is_rate_limited_ip, is_rate_limited_user,
                           validate_token, TokenValidationError,
                           DatabaseManager)

from utils.logging_utils import configure_logging, set_user_id, set_session_id
from utils.io_utils import send_audio_with_header, send_transcript_audio_with_header, HEADER_FORMAT, send_with_backpressure
from data.assembly_db import SoundAssigner
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = False

if not DEBUG:
    certfile = "/root/.ssh/myatmos_pro_chain.crt"
    keyfile = "/root/.ssh/myatmos.key"

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
else:
    ssl_context = None

class SharedResources:
    def __init__(self):
        self.assigner_SFX = SoundAssigner(chroma_name="SFX_db", data_root="./data/datasets")
        self.assigner_Music = SoundAssigner(chroma_name="SA_db", data_root="./data/datasets")
        self.openai = AsyncOpenAI()

shared_resources = SharedResources()

class AudioServer:

    def __init__(self, user_id='-1', co_auth=False, music=False, sfx=False, host="0.0.0.0", port=8765):
        self.user_id = user_id
        self.db_session = db_manager.start_session(self.user_id)
        self.session_start_time = time.monotonic()
        self.session_id = '-1'
        self.co_auth = co_auth
        self.music = music
        self.sfx = sfx
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
        self.music_score_threshold = 1e6
        self.narration_transcript = ""
        self.last_sfx_lock = asyncio.Lock()
        self.last_sfx = []
        self.last_sfx_time = 0
        self.last_narration_turn = ""
        self.last_narration_time = 0
        self.last_narration_lock = asyncio.Lock()
        self.ignore_transcripts = False 
        self.debounce_task = None
        self.debounce_time = 4
        self.time_limit = 30
        self.trigger_disconnect = False
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

    async def get_next_story_section(self, transcript):
        # use ChatGPT to generate the next section of the story
        p = random.uniform(0.0, 1.0)
        if p < 0.5:
            answer_format = "give me two options about where the story should go."
        else:
            answer_format = "tell me it's my turn and ask what should happen next?"

        chat = await self.client.chat.completions.create(
            model="gpt-4o",
            modalities=["text"],
            messages=[
                {"role": "system", "content": f"""You are a helpful assistent
                 who is going to make a story with me. I will start the story
                 and you will continue it. Keep your sections to 2 or 3 sentence. 
                 You should start your response with an acknowledgement of what I said and a summary, e.g. "Nice, <summary> or I like it <summary>.
                 After you've finished {answer_format}.
                 The story is for children under 10, keep the language simple and the story fun.
                 """
                 },
                {
                    "role": "user",
                    "content": transcript
                }
            ]
        )
        text = chat.choices[0].message.content
        audio_response = self.get_next_story_audio_response(text)
        return text, audio_response

    async def get_next_story_audio_response(self, text):
        """
        Async generator that yields data chunks as they are received.
        """
        async with self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="wav",
            input=text,
        ) as response:
            async for data in response.iter_bytes(32 * 1024): # TODO look into this
                yield data     # Yield data to the caller for further processing
        
    async def send_audio_from_transcript(self, transcript, websocket):
        """Handles incoming transcript and sends audio after a 3-second pause."""
        # If ignoring transcripts, exit early
        if self.ignore_transcripts:
            logger.info("Ignoring incoming transcript due to ongoing narration.")
            return

        self.last_narration_turn += " " + transcript.strip()
        current_time = time.monotonic()

        async with self.last_narration_lock:
            # Cancel any running debounce task
            if self.debounce_task:
                self.debounce_task.cancel()
                try:
                    await self.debounce_task
                except asyncio.CancelledError:
                    pass

            # Start a new debounce task
            self.debounce_task = asyncio.create_task(self._debounce_and_send(websocket))

    async def _debounce_and_send(self, websocket):
        """Debounce function that waits n seconds before sending narration."""
        try:
            await asyncio.sleep(self.debounce_time)  # Wait for the debounce duration

            async with self.last_narration_lock:
                if self.last_narration_turn.strip():
                    # Set the ignore flag to prevent new transcripts
                    self.ignore_transcripts = True
                    logger.info(f"Now starting narration after debounce.")
                    sample_rate = 24000
                    self.narration_transcript += f"\r\n{self.last_narration_turn}"
                    transcript, audio_generator = await self.get_next_story_section(self.narration_transcript)

                    count = 0
                    sequence_id = uuid.uuid4().bytes
                    ind = "STORY"[:5].ljust(5)  # Ensure the indicator is 5 bytes
                    async for chunk in audio_generator:
                        header = struct.pack(
                            HEADER_FORMAT,       # Updated format
                            ind.encode(),        # Indicator
                            len(chunk),          # Size of this packet
                            sequence_id,         # Unique sequence ID
                            count,        # Packet count (sequence number)
                            count + 1,       # Total packets
                            sample_rate          # Sample rate
                        )
                        await send_with_backpressure(websocket, header + chunk)
                        count += 1

                    if self.music and not self.music_sent_event.is_set():
                        self.music_sent_event.set()
                        self.fire_and_forget(self.send_music_from_transcript(transcript, websocket))

                    self.narration_transcript += f"\r\n{transcript}"
                    self.insert_transcript_section(self.last_narration_turn, "", 0.0)
                    self.last_narration_turn = ""
                    logger.info(f"Sending story snippet")
                    self.insert_transcript_section(transcript, "", 0.0)
                    await websocket.send("Play")

                    self.ignore_transcripts = False


        except asyncio.CancelledError:
            # Handle cancellation silently; new transcript triggered reset
            pass

    def insert_transcript_section(self, transcript, sounds, score):
        self.transcript["transcript"].append(transcript),
        self.transcript["sounds"].append(sounds),
        self.transcript["score"].append(score)

    
    async def send_music_from_transcript(self, transcript, websocket):
        # if len(self.transcript["transcript"]) < 3:
        #     self.insert_transcript_section(transcript, "", 0.0)
        #     self.music_sent_event.clear()
        #     return
        try:
            # transcript_insert = " ".join(self.transcript["transcript"][-10:]) + " " + transcript
            filename, category, score = self.assigner_Music.retrieve_src_file(transcript)
            if score < self.music_score_threshold:
                if filename:
                    logger.info(f"Sending MUSIC track for category '{category}' to client with score: {score}.")
                    await send_audio_with_header(websocket, os.path.join(filename), "MUSIC")
                else:
                    logger.info("No MUSIC found for the given text.")
                    filename = ""
                    score = 0.0
            else:
                logger.warning(f"Not sending audio for category '{category}' to client with score: {score}.")
                self.music_sent_event.clear()
                self.music_score_threshold += 0.03
            self.insert_transcript_section(transcript, filename, score)
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket closed during send_music_from_transcript: {e}")
        except FileNotFoundError as e:
            logger.error(f"File not found during send_music_from_transcript: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in send_music_from_transcript: {e}")

    async def send_sfx_from_transcript(self, transcript, websocket):
        # check if it has been at least 3s from last SFX send
        current_time = time.monotonic()
        async with self.last_sfx_lock:
            # Check if at least 3 seconds have passed since the last SFX
            if current_time - self.last_sfx_time < 3:
                logger.info(f"Not sending SFX: only {current_time - self.last_sfx_time:.2f}s since last send.")
                self.insert_transcript_section(transcript, "", 0.0)
                return
        try:
            transcript_insert = " ".join(self.transcript["transcript"][-2:]) + " " + transcript
            logger.info(f"Transcript insert {transcript_insert}")
            filename, category, score = self.assigner_SFX.retrieve_src_file(transcript_insert)

            async with self.last_sfx_lock:
                # Check for repeat filename
                if filename in self.transcript["sounds"][-10:]:
                    logger.info(f"Not sending repeat of {filename}")
                    # self.last_sfx.append(filename)
                    self.insert_transcript_section(transcript, filename, score)
                    return
                # else:
                #     self.last_sfx = filename

            if score < self.sfx_score_threshold:
                if filename:
                    logger.info(f"Sending SFX for category '{category}' to client with score: {score}.")
                    await send_audio_with_header(websocket, os.path.join(filename), "SFX")
                    async with self.last_sfx_lock:
                        self.last_sfx_time = current_time  # Update the last SFX time
                else:
                    logger.info("No SFX found for the given text.")
                    filename = ""
                    score = 0.0
            else:
                logger.warning(f"Not sending audio for category '{category}' to client with score: {score}.")
            self.insert_transcript_section(transcript, filename, score)
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
            # if self.music and not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
            #         self.music_sent_event.set()
            #         self.fire_and_forget(self.send_music_from_transcript(transcript.text, websocket))

            # if self.sfx:
            #     self.fire_and_forget(self.send_sfx_from_transcript(transcript.text, websocket))
            
            if self.co_auth:
                self.fire_and_forget(self.send_audio_from_transcript(transcript.text, websocket))

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
            db_manager.end_session(self.db_session,
                                   self.session_id,
                                   self.transcript,
                                   co_auth=self.co_auth,
                                   music=self.music,
                                   sfx=self.sfx)
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
                end_utterance_silence_threshold=500
            )
            # Start the connection
            transcriber.connect()

            try:
                self.fire_and_forget(send_audio_with_header(websocket, "./data/datasets/intro_narration.wav", "STORY", 24000))
            except Exception as e:
                logger.error(f"Error sending files: {e}")

            # # send the start file to 
            while True:
                try:
                    message = await websocket.recv()
                    if isinstance(message, str):
                        if message == "STOP":
                            self.debounce_time = 2
                        if message == "START":
                            self.debounce_time = 10
                    elif isinstance(message, bytes):
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
        
        # try:
        #     co_auth = websocket.request_headers.get("CO-AUTH", "")
        #     is_co_auth = co_auth.lower() == "true"
        # except Exception as e:
        #     logger.error(f"Unable to determinie CO-AUTH mode {e}")
        is_co_auth = True

        try:
            music = websocket.request_headers.get("MUSIC", "")
            is_music = music.lower() == "true"
        except Exception as e:
            logger.error(f"Unable to determinie MUSIC mode {e}")
        # is_music = True

        # try:
        #     sfx = websocket.request_headers.get("SFX", "")
        #     is_sfx = sfx.lower() == "true"
        # except Exception as e:
        #     logger.error(f"Unable to determinie SFX mode {e}")
        is_sfx = False

        try:
            user_name = websocket.request_headers.get("userName", "")
        except Exception as e:
            logger.error(f"No username in header")
            user_name = 'empty'

        # Validate the token asynchronously
        try:
            user_id = await validate_token(token)
            logger.info(f"Connection accepted for user {user_name}")  
        except TokenValidationError as e:
            await websocket.close(code=4003, reason=str(e))
            logger.warning(f"Connection rejected: {str(e)}")
            return
        
        # Handle communication after successful authentication
        try:
            server_instance = AudioServer(user_name,
                                          co_auth=is_co_auth,
                                          music=is_music,
                                          sfx=is_sfx)  # Create a new instance for each connection
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
            await server_instance.close_all_tasks()
            await websocket.close()


    @staticmethod
    async def start_server(host="0.0.0.0", port=8765):
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
