from asyncio.log import logger
from datetime import datetime
import json
import os
import sqlite3
import threading
import time
import jwt

from dotenv import load_dotenv
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

RATE_LIMIT = 3  # Max 10 connections per IP/USER
RATE_LIMIT_WINDOW = 60  # In seconds
SAMPLE_RATE = 44100

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