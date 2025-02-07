from flask import Flask, request, jsonify
import threading
import sqlite3
import uuid
from datetime import datetime
import json
import ssl
import jwt
import os
from utils.session_utils import DatabaseManager, validate_token, TokenValidationError, is_rate_limited_ip, is_rate_limited_user
from functools import wraps

from dotenv import load_dotenv
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

# Initialize Flask app and DatabaseManager
app = Flask(__name__)
db_manager = DatabaseManager()
db_manager.initialize()

certfile = "/root/.ssh/myatmos_pro_chain.crt"
keyfile = "/root/.ssh/myatmos.key"

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)

# Token Verification Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        
        client_ip = request.remote_addr
        if is_rate_limited_ip(client_ip):
            return jsonify({"error": "Rate limit exceeded for IP"}), 429

        try:
            user_id = jwt.decode(token, SECRET_KEY, algorithms=["HS256"]).get("user_id")
            if is_rate_limited_user(user_id):
                return jsonify({"error": "Rate limit exceeded for user"}), 429

            request.user_id = user_id  # Attach user_id to the request context
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)

    return decorated

@app.route('/stories/get_stories', methods=['GET'])
@token_required
def get_stories():
    # Extract user_id from the request headers
    user_id = request.headers.get("username")


    try:
        # Fetch stories for the user
        stories = db_manager.get_stories(user_id, visible_only=True)

        # Format the response
        formatted_stories = [
            {
                "id": story[0],
                "user": story[1],
                "story_name": story[2],
                "story": story[3],
                "visible": bool(story[4]),
                "arc_section": story[5]
            }
            for story in stories
        ]

        return jsonify({"stories": formatted_stories}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stories/<story_id>', methods=['PUT'])
@token_required
def update_story(story_id):
    """Update an existing story."""
    try:
        # Parse the incoming JSON request body
        data = request.get_json()

        # Extract the fields to update
        user = request.headers.get("username") # request.user_id  # Retrieved from the token by the token_required decorator
        story_name = data.get("story_name")
        story_content = data.get("story")
        visible = data.get("visible", True)

        # Validate required fields
        if not story_name or not story_content:
            return jsonify({"error": "Fields 'story_name' and 'story' are required"}), 400

        # Update the story in the database
        db_manager.update_story(story_id, user, story_name, story_content, visible)

        return jsonify({"message": "Story updated successfully", "id": story_id}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404  # If the story or user does not exist
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # General server error

@app.route('/stories/streak', methods=['GET'])
def get_streak():
    """
    Fetch the streak for a specific user by contact_email from headers.
    """
    try:
        # Get the contact_email from the request headers
        contact_email = request.headers.get('username')  # 'username' header carries the email

        if not contact_email:
            return jsonify({"success": False, "error": "Username (email) header is required"}), 400

        # Use the DatabaseManager to fetch the user
        user = db_manager.get_user(contact_email)

        if user:
            return jsonify({
                "success": True,
                "contact_email": user["contact_email"],
                "streak": user["streak"]
            }), 200  # HTTP 200 OK
        else:
            return jsonify({
                "success": False,
                "error": "User not found"
            }), 404  # HTTP 404 Not Found
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500  # HTTP 500 Internal Server Error

@app.route('/stories/streak', methods=['POST'])
def update_streak():
    """
    Update the streak for a specific user by adding points.
    """
    try:
        # Get the contact_email from the request headers
        contact_email = request.headers.get('username')  # 'username' header carries the email

        if not contact_email:
            return jsonify({"success": False, "error": "Username (email) header is required"}), 400

        # Parse the points from the request body
        data = request.get_json()
        if not data or 'points' not in data:
            return jsonify({"success": False, "error": "Points are required"}), 400

        points = data['points']

        # Fetch the user
        user = db_manager.get_user(contact_email)

        if user:
            # Update the streak in the database
            db_manager.update_streak(contact_email, points)

            # Fetch the updated user details
            updated_user = db_manager.get_user(contact_email)

            return jsonify({
                "success": True,
                "message": "Streak updated successfully.",
                "contact_email": updated_user["contact_email"],
                "streak": updated_user["streak"],
                "points_earned": points
            }), 200  # HTTP 200 OK
        else:
            return jsonify({
                "success": False,
                "error": "User not found"
            }), 404  # HTTP 404 Not Found
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500  # HTTP 500 Internal Server Error


@app.route('/stories/login', methods=['POST'])
def login_or_register():
    """
    Check if the user exists. If not, register the user.
    """
    try:
        data = request.get_json()
        contact_email = data.get('contact_email')

        if not contact_email:
            return jsonify({"success": False, "error": "Email is required."}), 400

        # Check if user exists
        user = db_manager.get_user(contact_email)
        if user:
            # User already exists
            return jsonify({"success": True, "message": "User logged in successfully."}), 200
        else:
            # Register new user
            db_manager.add_user(contact_email=contact_email)
            return jsonify({"success": True, "message": "User registered successfully."}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/stories/characters', methods=['GET'])
def get_characters():
    """
    Endpoint to fetch characters from the database.
    Query Parameters:
        - owner_id (required): The ID of the user owning the characters.
        - visible_only (optional): Whether to filter by visibility (default is True).
    """
    try:
        # Get query parameters
        owner_id = request.args.get('owner_id')
        if not owner_id:
            return jsonify({"error": "owner_id is required"}), 400

        visible_only = request.args.get('visible_only', 'true').lower() == 'true'

        # Fetch characters from the database
        characters = db_manager.fetch_characters(owner_id=owner_id, visible_only=visible_only)

        # Return the characters as JSON
        return jsonify(characters), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/stories/characters', methods=['POST'])
def add_character():
    """
    Endpoint to add a new character.
    Expects a JSON payload with the following fields:
        - id (string, required): The unique ID for the character (UUID).
        - name (string, required): The name of the character.
        - description (string, required): The description of the character.
        - owner_id (string, required): The ID of the user who owns the character.
        - visible (boolean, optional): Whether the character is visible (default is True).
    """
    try:
        # Parse the JSON payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["id", "name", "description", "owner_id"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Extract data
        character_id = data["id"]
        name = data["name"]
        description = data["description"]
        owner_id = data["owner_id"]
        visible = data.get("visible", True)  # Default to True if not provided

        # Save the character to the database
        db_manager.save_character(
            asset_id=character_id,
            name=name,
            description=description,
            owner_id=owner_id,
            visible=visible
        )

        return jsonify({"success": True, "message": "Character created successfully"}), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


@app.route('/stories/characters/<string:character_id>', methods=['PUT'])
def delete_character(character_id):
    """
    Endpoint to "delete" a character by setting its visibility to 0.
    Path Parameters:
        - character_id (string, required): The unique ID of the character to delete.
    """
    try:
        # Update the visibility of the character to 0

        db_manager.update_character(asset_id=character_id.lower(), visible=0)

        return jsonify({"success": True, "message": "Character visibility updated to 0"}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404  # Character not found or invalid input
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


if __name__ == '__main__':
    # Run the server on HTTPS (you need to provide SSL certificates)
    app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context)