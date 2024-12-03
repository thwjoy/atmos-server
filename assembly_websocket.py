
import asyncio
import websockets
import os




import assemblyai as aai
from openai import AsyncOpenAI
import numpy as np
import ssl


from utils.session_utils import (is_rate_limited_ip, is_rate_limited_user,
                           validate_token, TokenValidationError,
                           DatabaseManager)

from utils.logging_utils import configure_logging, set_user_id, set_session_id
from utils.io_utils import send_audio_with_header, send_transcript_audio_with_header
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

    async def get_next_story_section(self, transcript):
        # use ChatGPT to generate the next section of the story
        self.narration_transcript += transcript
        chat = await self.client.chat.completions.create(
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
        sentence = response + " Now it's your turn."

        audio = await self.client.audio.speech.create(
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
        audio, transcript, sounds = await self.get_next_story_section(transcript)
        logger.info(f"Sending story snippet: {transcript}")
        self.insert_transcript_section(transcript, "", 0.0)
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
            if self.music and not self.music_sent_event.is_set(): # we need to accumulate messages until we have a good narrative
                    self.music_sent_event.set()
                    self.fire_and_forget(self.send_music_from_transcript(transcript.text, websocket))
            else:
                if self.sfx:
                    self.fire_and_forget(self.send_sfx_from_transcript(transcript.text, websocket))
            
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
            db_manager.insert_transcript_data(self.session_id,
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
                end_utterance_silence_threshold=1000
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
        
        try:
            co_auth = websocket.request_headers.get("CO-AUTH", "")
            is_co_auth = co_auth.lower() == "true"
        except Exception as e:
            logger.error(f"Unable to determinie CO-AUTH mode {e}")
            is_co_auth = False

        try:
            music = websocket.request_headers.get("MUSIC", "")
            is_music = music.lower() == "true"
        except Exception as e:
            logger.error(f"Unable to determinie MUSIC mode {e}")
            is_music = False

        try:
            sfx = websocket.request_headers.get("SFX", "")
            is_sfx = sfx.lower() == "true"
        except Exception as e:
            logger.error(f"Unable to determinie SFX mode {e}")
            is_sfx = False

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
            server_instance = AudioServer(user_id,
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
            db_manager.log_session_stop(db_session)
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
