
import asyncio
import websockets
import os
import time
import base64
import uuid
import struct
import random
import json

import assemblyai as aai
aai.settings.http_timeout = 30.0
from openai import AsyncOpenAI
import numpy as np
import ssl
import argparse


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

certfile = "/root/.ssh/myatmos_pro_chain.crt"
keyfile = "/root/.ssh/myatmos.key"

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)

temp_bytes = b'\x08\x00\x04\x00\x03\x00\x00\x00\x01\x00\x03\x00\xff\xff\x01\x00\x02\x00\x01\x00\x03\x00\x06\x00\x08\x00\x03\x00\xfc\xff\xfd\xff\xfe\xff\x05\x00\x0c\x00\x10\x00\x0f\x00\x0b\x00\x05\x00\x07\x00\x08\x00\x04\x00\xfb\xff\xf7\xff\xfb\xff\x07\x00\x10\x00\x1b\x00\x12\x00\x04\x00\xf8\xff\xf4\xff\xfb\xff\x02\x00\n\x00\n\x00\r\x00\n\x00\x08\x00\x00\x00\xfc\xff\xf4\xff\xf6\xff\xf9\xff\xfe\xff\x05\x00\x07\x00\x04\x00\x01\x00\xf5\xff\xf3\xff\xf2\xff\xf6\xff\xf8\xff\xf8\xff\xfa\xff\xfd\xff\xff\xff\xfe\xff\x01\x00\xf8\xff\xf9\xff\xf2\xff\xf5\xff\xf7\xff\xfa\xff\xfb\xff\xfe\xff\xfe\xff\xfe\xff\xf9\xff\xf6\xff\xf5\xff\xf4\xff\xf4\xff\xf8\xff\x06\x00\x07\x00\x02\x00\xf6\xff\xf4\xff\xf5\xff\xf8\xff\xf9\xff\xf7\xff\xfa\xff\xf3\xff\xf0\xff\xef\xff\xfb\xff\x03\x00\n\x00\xfd\xff\xf9\xff\xf9\xff\xfb\xff\xfc\xff\xf9\xff\xfa\xff\xfc\xff\xfd\xff\xfd\xff\x01\x00\x08\x00\x08\x00\x01\x00\xf5\xff\xf3\xff\xf6\xff\x06\x00\x06\x00\x07\x00\x07\x00\x02\x00\xfe\xff\xfa\xff\xfb\xff\xfb\xff\xff\xff\x04\x00\t\x00\x0f\x00\x11\x00\x0e\x00\x08\x00\x06\x00\x04\x00\x07\x00\x08\x00\x0c\x00\x11\x00\x17\x00\x14\x00\x14\x00\x10\x00\x0b\x00\x07\x00\x06\x00\t\x00\x10\x00\x13\x00\x14\x00\x11\x00\r\x00\x06\x00\r\x00\x0c\x00\r\x00\x0c\x00\x0b\x00\x05\x00\x01\x00\x00\x00\x03\x00\x13\x00\x0f\x00\r\x00\xfe\xff\xff\xff\x08\x00\n\x00\r\x00\x0e\x00\x0e\x00\x0f\x00\x07\x00\t\x00\x0c\x00\x12\x00\x14\x00\r\x00\x04\x00\xff\xff\x02\x00\x07\x00\x0c\x00\x10\x00\x0e\x00\x18\x00\x19\x00\x17\x00\x0b\x00\x02\x00\xfd\xff\n\x00\x1c\x00$\x00&\x00\x1f\x00\x1d\x00\x16\x00\x07\x00\x04\x00\x07\x00\x1b\x00 \x00\x1d\x00\x1d\x00\x1b\x00\x19\x00\x0c\x00\xfe\xff\xfa\xff\t\x00\x17\x00\x10\x00\r\x00\xff\xff\x05\x00\t\x00\x0b\x00\x0f\x00\r\x00\x17\x00\x15\x00\x12\x00\x0f\x00\x0f\x00\x0e\x00\x0e\x00\r\x00\x18\x00\x1d\x00\x1e\x00\x11\x00\xff\xff\xfe\xff\x06\x00\x19\x00\x1c\x00\x19\x00\x1d\x00 \x00!\x00\x14\x00\x1b\x00\x15\x00\x11\x00\x06\x00\x08\x00\x12\x00\x19\x00!\x00\x1c\x00\x1d\x00\x19\x00\x1a\x00\x18\x00\x13\x00\x13\x00\x13\x00\x14\x00\x0f\x00\x12\x00\x16\x00%\x001\x005\x00@\x008\x000\x00\x1e\x00\x16\x00\x1a\x00"\x00@\x00N\x00X\x00V\x00T\x00R\x00K\x00A\x006\x000\x001\x00D\x00B\x00P\x00P\x00Q\x00?\x00*\x00\x19\x00\x13\x00\x1c\x00"\x00!\x00\x1d\x00\x1c\x00\x1d\x00\x1a\x00\x10\x00\x15\x00\x19\x00&\x00\x1d\x00\x1d\x00\r\x00\xfa\xff\xf0\xff\xf1\xff\x08\x00)\x00\\\x00m\x00j\x00D\x00\x17\x00\xfe\xff\xf3\xff\x07\x00\x1f\x00H\x00f\x00u\x00e\x00C\x00)\x00\x15\x00\t\x00\x0b\x00\x15\x00\x1d\x00&\x00!\x000\x007\x00>\x00#\x00\x0c\x00\xf1\xff\xdc\xff\xd1\xff\xc5\xff\xc9\xff\xe5\xff\xfc\xff\x04\x00\x02\x00\xf9\xff\xe6\xff\xd2\xff\xc2\xff\xb8\xff\xb0\xff\xb0\xff\xc3\xff\xdc\xff\xf4\xff\xf7\xff\xf9\xff\xf6\xff\x08\x00\x11\x00\xf6\xff\xc6\xff\x9d\xff\x9b\xff\xda\xff5\x00\x9d\x00\xdd\x00\xec\x00\xce\x00\xa0\x00\x9d\x00\x95\x00U\x00\xa6\x00\xe8\x00q\x01\xfc\x01\x01\x02\x13\x02\xb5\x01\xa3\x01\x08\x01\x06\x01\xf2\x01\x93\x03\x02\x04 \x01\xd8\xfd\x01\xfc\x10\xfeA\x02l\x05\xc5\x06`\x04\x1d\x01M\xfd\x96\xfb\xae\xfby\xfb~\xfb\xf4\xfb\xbb\xfeO\x03\xfe\x05\xed\x04\x91\xffz\xf9\xcc\xf5\x16\xf6\xa6\xf9 \xfe\xba\x00\xf5\x00\x89\x00\xf9\x00\x86\x02\xb3\x02\xd0\xff\xc5\xfb\x0e\xf9\xe7\xf9y\xfd7\x01\x98\x03\xbc\x04\xc8\x04\x18\x04\xf2\x02\x06\x011\xff\xc4\xfd\xa9\xfdW\xffi\x02\x1f\x05?\x06\xe0\x04^\x02\x99\x00y\xff\x05\xff)\xff\xe9\xffx\x00\xbb\x00]\x00\x81\xff\x9e\xffJ\x00\xe5\x00y\x00\xd9\xfe\n\xfd\xc6\xfbt\xfb\xdb\xfb\xa4\xfc\x1b\xfe\xf7\xff4\x01\xf7\x00"\xff\xae\xfcv\xfal\xf9b\xfa\x11\xfd\t\x00\xd3\x01\xb9\x01\xd6\x00!\x00\xa5\xff\xdc\xfeq\xfd*\xfc\x19\xfc&\xfe\xcd\x01\\\x05k\x06?\x04m\x00\x96\xfd\xfb\xfcR\xfe/\x00\x82\x01\x80\x02\xd8\x02!\x03\xe1\x021\x02\xa1\x00\xd5\xfe\xc3\xfdg\xfe\xba\x00\xb5\x02L\x03\xe2\x01\xdf\xff|\xfeL\xfe)\xff\x1a\x00\x99\x00G\x00\xa3\xff!\xff\xdc\xfe\xb5\xfen\xfe:\xfe\x82\xfeF\xff:\x00\xa6\x00\x1d\x00\xe6\xfe\x8d\xfd\xe0\xfc*\xfdT\xfe\xce\xff\xfa\x00S\x01\xd8\x00\xa9\xffj\xfe\x80\xfd>\xfd\xbb\xfd\xe8\xfeN\x00t\x01\xd1\x01A\x01\xff\xff\xa3\xfe\xa9\xfd\x8b\xfd/\xfeE\xffZ\x00,\x01x\x01\xeb\x00\xce\xff\x99\xfe\x1a\xfe@\xfe\xe1\xfe\x8e\xff!\x00i\x00+\x00\x90\xff&\xff\x0c\xff\x1e\xff\x14\xff\x19\xffY\xff\xa1\xff\xcf\xff\x9c\xffj\xffI\xff_\xff\x90\xff\xdc\xff"\x00L\x00S\x00,\x00\xf1\xff\xc4\xff\xe1\xffI\x00\xbc\x00\xfc\x00\xf2\x00\xd3\x00\x9a\x00M\x00\xfe\xff\x07\x00k\x00\xf0\x00D\x01`\x01>\x01\xe5\x00[\x00\xdc\xff\xda\xff8\x00\xcd\x008\x01`\x01,\x01\xaf\x00\x1a\x00\x91\xff\x80\xff\xbb\xff4\x00\xc8\x00.\x01s\x01F\x01\x95\x00\xb2\xff \xff-\xff\x9c\xff\x14\x00\x88\x00\x15\x01|\x01n\x01\xd7\x00\x1f\x00\x9e\xffU\xffP\xff\x9f\xff-\x00\xc6\x00\x06\x01"\x01\xe5\x00\x9b\x00\x17\x00\xc2\xff\xa7\xff\xbf\xff\xf2\xff4\x00\x93\x00\xd5\x00\xd2\x00\x9a\x00B\x00\xf4\xff\xbc\xff\xbf\xff$\x00\xb3\x00#\x015\x01\xda\x00L\x00\xaf\xffL\xffo\xff\x1d\x00\x12\x01\xb0\x01\xde\x01\x95\x01\x07\x01U\x00\xde\xff\xb3\xff?\x00\xe3\x00~\x01\xec\x01\x19\x022\x02\xb6\x01\x03\x01M\x00\'\x00y\x00\xe9\x00M\x01\xa1\x01\xf1\x01\x18\x02\xda\x01h\x01\xd6\x00m\x00"\x00\xf1\xff\x0b\x00\x81\x00\x1b\x01\xb0\x01\xc2\x01\x88\x01\xf8\x00F\x00\x8b\xff\x08\xff\xf1\xfe"\xff\x87\xff8\x00\x01\x01\x9c\x01W\x01-\x00\xc5\xfe\xf2\xfd\xf9\xfd\x93\xfeD\xff\xcb\xff\x03\x00\x00\x00\xd9\xff\x8d\xffY\xff\xfd\xfe\xa2\xfe\x90\xfe\xbd\xfej\xff\xd1\xff\xbc\xff\x19\xff4\xfe\xcc\xfd\x1c\xfe:\xff\xfe\xff\x1b\x00%\xff5\xfe\xdc\xfd\xfa\xfdU\xfe\xa1\xfe5\xff\x96\xff\x93\xffV\xff\x19\xff\x0c\xff\xd4\xfe\x92\xfej\xfe*\xff\xdb\xff|\x00;\x00\xcc\xff3\xff4\xffu\xff\x9b\xff\xe6\xff\xe3\xff"\x00\xff\xff\xa7\xffB\xff\x1e\xffF\xff~\xff\x95\xff\xbc\xff\xc3\xff\x04\x00l\xff\xef\xfe\n\xfe\xfa\xfd\\\xfeW\xff\x11\x006\x00\xae\xff\xd9\xfe8\xfe\x0c\xfe\xa6\xfe\'\xff\x85\xffH\xff\x00\xffQ\xff\x06\x00i\x008\x00\x89\xff1\xff\x81\xff\xcf\xffa\x00V\x00m\x00k\x00\xb1\x00a\x01\xee\x01\x0b\x02t\x01\xea\xff\xfa\xfe<\xff\x01\x00\xc5\x01"\x02\xe6\x02\xea\x02\x9c\x02\xfd\x01`\x00\xc0\xff"\xff\xbf\xff\x91\x00\xa5\x01\xec\x02\xfb\x03\xf7\x03\xd5\x02\xb4\x006\xffW\xfer\xfe\xf8\xfe\xee\x00\xdb\x02#\x04\xb6\x03x\x01k\xff\xf1\xfd4\xfdG\xfdj\xfe\x9a\x00\x9a\x02\xdd\x02\x98\x01\xca\xff\x8c\xfeM\xfe\xc1\xfeA\xff\xaa\xff\xdc\xff\xb0\xff{\x00G\x01y\x02\x90\x02\xcf\x01\x1b\x00\x8f\xfe\xa1\xfd\x8c\xfd\x1b\xffS\x00+\x02\x96\x03;\x04\xe1\x03i\x01\xa9\xfe\x15\xfc\xc0\xfb\x90\xfc\xad\xfe\x05\x01w\x02\xdd\x02\xd9\x01\xbe\xff\xb6\xfd\xf8\xfbT\xfa\x88\xfa\x8f\xfb\xa0\xfe\xe4\x00\xb0\x01`\x00s\xfd\xd4\xfa\x96\xf9?\xfaE\xfd\x1e\x00}\x02\xe0\x02\xfc\x00\xed\xff\x91\xfd\x82\xfd"\xfe\x16\x00\xf9\x02H\x05\x1c\x06X\x05\xae\x03\x9f\x01\xcd\x00\xfc\x00\x95\x02\xa1\x04\x80\x06\x1d\x07\xd4\x06H\x05P\x03#\x02(\x01A\x02\xe6\x02"\x04\xb2\x04n\x05\xd8\x04i\x03C\x00\x80\xfe\x8a\xfd\x81\xfe"\x00\x8d\x01<\x02\xbb\x01\x94\xff\xa1\xfd\x07\xfc\xc4\xfb\n\xfc\xd1\xfb1\xfc\xcb\xfc\x15\xfe\x05\xfeT\xfd\xb1\xfb\xe0\xfa\n\xfb\xc2\xfb\xeb\xfc\x16\xfdn\xfd\xac\xfc{\xfc\xb0\xfc\xbf\xfd%\xff\xfa\xffv\x00~\x00\xb1\x00\xb8\x00I\x00\xed\xff\xce\xff\x11\x00\xc7\x00\x93\x014\x03\xf5\x03F\x04\x17\x03i\x01\xa1\xff\xe8\xfe\xd6\xfe\x06\x00\xda\x00\xfb\x01\xb6\x02P\x02\xef\x01\x05\x00\x90\xfe\xa2\xfcd\xfc\xd2\xfc7\xfe\xd4\xffk\x00\x14\x00\x8b\xfe\xd8\xfc\xca\xfb\x00\xfc\xe1\xfc\xaf\xfd\xdd\xfd9\xfd\xfc\xfd-\xfd\x92\xfd\xc2\xfb\xf2\xfa\xd8\xfa\x97\xfa\x80\xfc\xa5\xfc\x10\xfe\xbb\xfd2\xfd\x8c\xfcs\xfc\x03\xfd\x0c\xfe\xee\xfe\x8b\x00\x14\x00|\xff\xa4\xfc\xa9\xfaw\xfa\xa6\xfa\xdc\xfe\x16\x03t\x040\x06H\x01'

async def insert_temp_bytes_periodically(transcriber, interval=10):
    """
    Periodically inserts temp_bytes into the transcriber every `interval` seconds.
    """
    while True:
        transcriber.stream([temp_bytes])
        await asyncio.sleep(interval)


STORY_ARCS = """
1) Stasis: The “normal world” before the story begins.
2) Trigger/Inciting Incident: An event disrupts the status quo and sets the story in motion.
3) The Quest: The protagonist begins their journey toward a goal or resolution.
4) Surprise: Complications, twists, and smaller challenges along the way.
5) Critical Choice: The protagonist faces a defining decision that impacts the outcome.
6) Climax: The most intense or pivotal moment in the story.
7) Resolution: The protagonist achieves or fails their goal, leading to a new stasis.
"""


class SharedResources:
    def __init__(self):
        self.assigner_SFX = SoundAssigner(chroma_name="SFX_db", data_root="./data/datasets")
        self.assigner_Music = SoundAssigner(chroma_name="SA_db", data_root="./data/datasets")
        self.openai = AsyncOpenAI()

shared_resources = SharedResources()

class AudioServer:

    def __init__(self, user_id='-1', co_auth=False, music=False, sfx=False, story_id=None,
                 arc_section=0, last_narration_turn="", host="0.0.0.0", port=8765):
        self.user_id = user_id
        self.db_session = db_manager.start_session(self.user_id)
        self.session_start_time = time.monotonic()
        self.session_id = '-1'
        self.story_id = story_id
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
        self.resume_story = True if last_narration_turn else False
        self.last_narration_turn = last_narration_turn if last_narration_turn else ""
        self.last_narration_time = 0
        self.last_narration_lock = asyncio.Lock()
        self.response_count = 0
        self.last_arc_change = 0
        self.arc_number = arc_section
        self.init_arc_number = arc_section
        self.ignore_transcripts = False 
        self.debounce_task = None
        self.debounce_time = 4
        self.process_audio = asyncio.Event()
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

    async def neaten_story(self, transcript):
        try:
            chat = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                modalities=["text"],
                messages=[
                    {"role": "system", "content": """You are a helpful assistent.
                    I will pass in a transcript of two people co-creating a story.
                    Do not add anything to the story, you should only format it so that it sounds coherent to read.
                    Please provide me a JSON like this {"name": "<story_name>", "story": <"story">}
                    """
                    },
                    {
                        "role": "user",
                        "content": transcript
                    }
                ],
                response_format={"type": "json_object" },
            )
            event = json.loads(chat.choices[0].message.content)
            name = event['name']
            story = event['story']
        except Exception as e:
            logger.warning(f"Unable to summarise: {e}")
            story = transcript
            name = "Your Story"
        return name, story

    async def get_next_story_section(self, transcript, arc_number):
        # use ChatGPT to generate the next section of the story
        p = random.uniform(0.0, 1.0)
        if p < 0.35:
            answer_format = "give me two options about where the story should go."
        else:
            answer_format = "tell me it's my turn and ask what should happen next?"

        if arc_number == 7:
            answer_format = "you should summarise the story and end the story without giving anymore options."

        chat = await self.client.chat.completions.create(
            model="gpt-4o",
            modalities=["text"],
            messages=[
                {"role": "system", "content": f"""You are a helpful assistent
                 who is going to make a story with me. I will start the story
                 and you will continue it. You should start your response with 
                 a small acknowledgement of what I said, but then proceed to 
                 add new sections of the story and give me lots of surprises,
                 twists and turns. Keep your responses to 3 or 4 sentences.
                 After you've finished {answer_format}.
                 The story is for children under 10, keep the language simple and 
                 the story fun, let your imagination run wild with fantasy.
                 
                 You should try and guide the story according to the story arc below:
                 {STORY_ARCS}
                 The story is currently at story arc {self.arc_number}. If they 
                 ask you for help or say I want to move on, you should do this for them.
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

    async def analyse_story_arc(self, transcript):
        try:
            chat = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                modalities=["text"],
                messages=[
                    {"role": "system", "content": f"""You are a helpful assistent.

                    I will pass you a transcript of two people co-creating a story.

                    You need to decide at what part of the story arc the transcript it. Choose from the following
                    
                    Template of story arc: 
                    <story_section_number>) arc_description
                    {STORY_ARCS}

                    The response should be in JSON format like this {{"story_section_number": "<story_section_number>", "arc_description": <"arc_description">}}
                    """
                    },
                    {
                        "role": "user",
                        "content": transcript
                    }
                ],
                response_format={"type": "json_object" },
            )
            event = json.loads(chat.choices[0].message.content)
            number = event['story_section_number']
            section = event['arc_description']
        except Exception as e:
            logger.warning(f"Unable to analyse story arc: {e}")
            number = -1
            section = "NA"
        return number, section

        
    def insert_transcript_section(self, transcript, sounds, score):
        self.transcript["transcript"].append(transcript),
        self.transcript["sounds"].append(sounds),
        self.transcript["score"].append(score)

    async def play_music_periodically(self, websocket, filename, t):
        while True:
            await send_audio_with_header(websocket, os.path.join(filename), "MUSIC")
            # Wait for 3 minutes (180 seconds) before sending again
            await asyncio.sleep(t)


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
                    self.fire_and_forget(self.play_music_periodically(websocket, filename, 180))
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

    async def send_audio_from_transcript(self, websocket):

        try:
            # Wait for the event with a timeout of 10 seconds
            await asyncio.wait_for(self.process_audio.wait(), timeout=5.0)
            logger.info("Event completed within the time limit.")
        except asyncio.TimeoutError:
            logger.info("Timeout reached! The event did not complete in time.")

        self.narration_transcript += f"\r\n{self.last_narration_turn}"
        self.response_count += 1
        arc_number, arc_desc = await self.analyse_story_arc(self.narration_transcript)
        try:
            arc_number = int(arc_number)
        except ValueError:
            logger.error("Invalid conversion from st r to int in arc_number")

        if arc_number > self.arc_number:
            self.arc_number = arc_number
            self.last_arc_change = self.response_count
        elif self.last_arc_change + 2 < self.response_count:
            print("Forcing move on")
            self.arc_number = self.arc_number + 1 
            self.last_arc_change = self.response_count

        await websocket.send(f"ARCNO: {self.arc_number}")
        score = self.get_streak_score(self.init_arc_number, self.arc_number)
        await websocket.send(f"Streak: {score}")

        # if self.arc_number == 7:
        #     try:
        #         self.fire_and_forget(send_audio_with_header(websocket, "./data/datasets/finished_story.wav", "STORY", 24000))
        #     except Exception as e:
        #         logger.error(f"Error sending files: {e}")
        #     finally:
        #         return
    
        sample_rate = 24000
        transcript, audio_generator = await self.get_next_story_section(self.narration_transcript, self.arc_number)

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

    def get_streak_score(self, init_arc_number, arc_number):
        """
        Calculate the streak score based on how far the user has advanced in arc_number.

        Parameters:
        - init_arc_number (int): The starting arc number.
        - arc_number (int): The current arc number.

        Returns:
        - int: The score based on the advancement.
        """
        if init_arc_number < 0 or init_arc_number > 7 or arc_number < 0 or arc_number > 7:
            raise ValueError("arc_number and init_arc_number must be between 0 and 7")

        if arc_number <= init_arc_number:
            # No progress or regression
            return 0

        # Define the points for each incremental advance
        arc_progression_points = {
            1: 5,   # Points for advancing 1 arc
            2: 10,  # Points for advancing 2 arcs
            3: 15,  # Points for advancing 3 arcs
            4: 20,  # Points for advancing 4 arcs
            5: 30,  # Points for advancing 5 arcs
            6: 45,  # Points for advancing 6 arcs
            7: 80   # Points for advancing 7 arcs (full progress)
        }

        # Calculate how far the user has advanced
        arc_progress = arc_number - init_arc_number

        # Return the corresponding points based on progress
        return arc_progression_points.get(arc_progress, 0)
            
    async def send_message_async(self, message, websocket):
        await asyncio.create_task(websocket.send(message))
        logger.info(f"Message sent: {message}")

    def on_data(self, transcript, websocket, loop):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            logger.debug(f"Recieved: {transcript.text}")
            self.last_narration_turn += " " + transcript.text.strip()
            self.process_audio.set()
        elif isinstance(transcript, aai.RealtimePartialTranscript):
            self.process_audio.clear()
        # asyncio.run_coroutine_threadsafe(self.process_transcript_async(transcript, websocket), loop)



    def on_open(self, session, websocket, loop):

        set_user_id(self.user_id)
        logger.info(f"Transcriber session ID: {session.session_id}")
        self.session_id = session.session_id
        self.story_id = self.session_id if not self.story_id else self.story_id
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

            if not self.resume_story:
                try:
                    self.fire_and_forget(send_audio_with_header(websocket, "./data/datasets/intro_narration.wav", "STORY", 24000))
                except Exception as e:
                    logger.error(f"Error sending files: {e}")
            else:
                self.process_audio.set()
                await websocket.send(f"ARCNO: {self.arc_number}")
                self.fire_and_forget(self.send_audio_from_transcript(websocket))

            # hack to keep transcriber online
            self.fire_and_forget(insert_temp_bytes_periodically(transcriber))

            # # send the start file to 
            while True:
                try:
                    message = await websocket.recv()
                    if isinstance(message, str):
                        if message == "STOP":
                            self.fire_and_forget(self.send_audio_from_transcript(websocket))
                    elif isinstance(message, bytes):
                        transcriber.stream([message])
                except Exception as e:
                    logger.error(f"Closing: {e}")
                    break  # Exit the loop if an error occurs
        finally:
            if len(self.transcript["transcript"]) > 0:
                score = self.get_streak_score(self.init_arc_number, self.arc_number)
                db_manager.update_streak(contact_email=self.user_id,
                                         points=score)

                name, story = await self.neaten_story("\r\n".join(self.transcript["transcript"]))
                if self.resume_story:
                    db_manager.update_story(story_id=self.story_id,
                                        user=self.user_id,
                                        story=story,
                                        visible=1,
                                        arc_section=self.arc_number)
                else:
                    db_manager.add_story(story_id=self.story_id,
                                        user=self.user_id,
                                        story=story,
                                        story_name=name,
                                        visible=1,
                                        arc_section=self.arc_number)
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
            user_id = user_name
            logger.info(f"Connection accepted for user {user_name}")  
        except TokenValidationError as e:
            await websocket.close(code=4003, reason=str(e))
            logger.warning(f"Connection rejected: {str(e)}")
            return

        try:
            story_id = websocket.request_headers.get("storyId", "")
            story = db_manager.get_story(story_id, user_id)
            if not story:
                story = {
                    'id': None,
                    'story': None,
                    'arc_section': 0,
                }
        except Exception as e:
            logger.error(f"No story_id in header")
            story = {
                    'id': None,
                    'story': None,
                    'arc_section': 0,
                }
        
        # Handle communication after successful authentication
        try:
            server_instance = AudioServer(user_name,
                                          co_auth=is_co_auth,
                                          music=is_music,
                                          sfx=is_sfx,
                                          story_id=story['id'],
                                          last_narration_turn=story['story'],
                                          arc_section=story['arc_section'])  # Create a new instance for each connection
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
            await websocket.close(code=1000, reason="Server shutting down gracefully")



    @staticmethod
    async def start_server(host="0.0.0.0", port=8765):
        logger.info(f"Starting WebSocket server on wss://{host}:{port}")
        async with websockets.serve(AudioServer.connection_handler, host, port, ping_interval=300, ping_timeout=300, ssl=ssl_context):
            await asyncio.Future()  # Run forever

# Server entry point
def run(port):
    asyncio.run(AudioServer.start_server(port=port))


# Run the server if the script is executed directly
if __name__ == "__main__":
    # Add argparse for port argument
    parser = argparse.ArgumentParser(description="Run the server with a specified port.")
    parser.add_argument("--port", type=int, default=8765, help="The port on which the server will run (default: 8000).")
    
    args = parser.parse_args()
    port = args.port

    logger = configure_logging()
    # Create a global instance of DatabaseManager
    db_manager = DatabaseManager()
    db_manager.initialize()  # Ensure tables are created
    run(port)
