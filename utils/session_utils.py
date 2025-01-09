from asyncio.log import logger
from datetime import datetime
import json
import os
import sqlite3
import sqlite_history
import threading
import time
import jwt
import uuid

from dotenv import load_dotenv
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

RATE_LIMIT = 60  # Max 10 connections per IP/USER
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
        """Create the merged table if it doesn't exist."""
        with self.connect() as conn:
            cursor = conn.cursor()

            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    connection_id TEXT,
                    data TEXT,
                    co_auth BOOLEAN NOT NULL DEFAULT 0,
                    music BOOLEAN NOT NULL DEFAULT 0,
                    sfx BOOLEAN NOT NULL DEFAULT 0,
                    start_time TEXT NOT NULL,
                    stop_time TEXT
                )
            ''')
            try:
                sqlite_history.configure_history(conn, "sessions")
            except:
                logger.debug("History already exisits")

            # Create stories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stories (
                    id TEXT PRIMARY KEY,  -- UUID
                    user TEXT NOT NULL,   -- User who created the story
                    story_name TEXT NOT NULL, -- The name of the story
                    story TEXT NOT NULL,  -- The story content
                    visible BOOLEAN NOT NULL DEFAULT 1,  -- Visibility flag
                    arc_section INTEGER NOT NULL DEFAULT 0  -- Arc section from 0 to 7
                )
            ''')
            try:
                sqlite_history.configure_history(conn, "stories")
            except:
                logger.debug("stories already exisits")

            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    uuid TEXT PRIMARY KEY,            -- Unique UUID for each user
                    contact_email TEXT NOT NULL UNIQUE, -- Unique email identifier
                    streak INTEGER NOT NULL DEFAULT 0  -- Integer to store the points (default is 0)
                )
            ''')
            try:
                sqlite_history.configure_history(conn, "users")
            except:
                logger.debug("users already exisits")

            conn.commit()

    def start_session(self, user_id):
        """Start a session by setting start_time."""
        start_time = datetime.utcnow().isoformat()  # ISO 8601 format
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO sessions (user_id, start_time)
                VALUES (?, ?)
                ''',
                (user_id, start_time)
            )
            conn.commit()
            return cursor.lastrowid  # Return the record ID

    def end_session(
        self, record_id, connection_id, data, co_auth=False, music=False, sfx=False
    ):
        """End a session by updating connection_id, data, and stop_time."""
        stop_time = datetime.utcnow().isoformat()  # ISO 8601 format
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                UPDATE sessions
                SET connection_id = ?, data = ?, co_auth = ?, music = ?, sfx = ?, stop_time = ?
                WHERE id = ?
                ''',
                (str(connection_id), json.dumps(data), co_auth, music, sfx, stop_time, record_id)
            )
            conn.commit()

    def add_story(self, story_id, user, story_name, story, visible=True, arc_section=0):
        """Add a new story to the stories table."""
        insert_query = """
        INSERT INTO stories (id, user, story_name, story, visible, arc_section)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(insert_query, (str(story_id), user, story_name, story, visible, arc_section))
            conn.commit()

    def update_story(self, story_id, user, story_name=None, story=None, visible=None, arc_section=None):
        """
        Update the story_name, story content, visibility, and arc_section for a specific story ID.
        
        Parameters:
            - story_id: The ID of the story to update.
            - user: The user who owns the story.
            - story_name: (Optional) The new name for the story.
            - story: (Optional) The new content for the story.
            - visible: (Optional) The visibility status (e.g., 1 for visible, 0 for hidden).
            - arc_section: (Optional) The new arc section value.
        """
        # Build the base query
        update_query = "UPDATE stories SET "
        params = []

        # Add fields to update only if they are not None
        if story_name is not None:
            update_query += "story_name = ?, "
            params.append(story_name)
        if story is not None:
            update_query += "story = ?, "
            params.append(story)
        if visible is not None:
            update_query += "visible = ?, "
            params.append(visible)
        if arc_section is not None:
            update_query += "arc_section = ?, "
            params.append(arc_section)

        # Remove the trailing comma and space
        update_query = update_query.rstrip(", ")

        # Add the WHERE clause
        update_query += " WHERE id = ? AND user = ?"
        params.extend([story_id, user])

        # Execute the query
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, params)
            if cursor.rowcount == 0:
                raise ValueError("No story found with the given ID.")
            conn.commit()


    def get_stories(self, user, visible_only=True):
        """Retrieve stories for a specific user, optionally filtering by visibility."""
        if not user:
            raise ValueError("User must be supplied to retrieve stories.")

        query = "SELECT * FROM stories WHERE user = ?"
        params = [user]

        if visible_only:
            query += " AND visible = 1"

        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_story(self, story_id, user):
        """Retrieve a single story by story_id and user."""
        query = """
        SELECT id, user, story_name, story, visible, arc_section
        FROM stories
        WHERE id = ? AND user = ?
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (story_id, user))
            result = cursor.fetchone()

        if result:
            return {
                "id": result[0],
                "user": result[1],
                "story_name": result[2],
                "story": result[3],
                "visible": bool(result[4]),
                "arc_section": result[5],
            }
        else:
            return None

    def add_user(self, contact_email, streak=0):
        """Add a new user to the users table."""
        user_uuid = str(uuid.uuid4())  # Generate a new UUID
        insert_query = """
        INSERT INTO users (uuid, contact_email, streak)
        VALUES (?, ?, ?);
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(insert_query, (user_uuid, contact_email, streak))
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Error adding user: {e}")

    def get_user(self, contact_email):
        """Retrieve a user's details by contact_email."""
        query = "SELECT uuid, contact_email, streak FROM users WHERE contact_email = ?"
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (contact_email,))
            result = cursor.fetchone()

        if result:
            return {
                "uuid": result[0],  # Return the UUID as well
                "contact_email": result[1],
                "streak": result[2],
            }
        else:
            return None

    def update_streak(self, contact_email, points):
        """Update a user's streak by adding points."""
        update_query = """
        UPDATE users
        SET streak = streak + ?
        WHERE contact_email = ?
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, (points, contact_email))
            if cursor.rowcount == 0:
                raise ValueError("No user found with the given email.")
            conn.commit()

    def reset_streak(self, contact_email):
        """Reset a user's streak to 0."""
        update_query = """
        UPDATE users
        SET streak = 0
        WHERE contact_email = ?
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, (contact_email,))
            if cursor.rowcount == 0:
                raise ValueError("No user found with the given email.")
            conn.commit()

    