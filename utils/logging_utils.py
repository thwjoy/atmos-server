import logging
from contextvars import ContextVar


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
