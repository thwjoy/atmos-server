import secrets
import jwt
import datetime
from dotenv import load_dotenv
import os

"""
Generated Secret Key: 08954bc12212acf67aa177bf177040ae38371f84cc491ad69b3d24632deb9e17
Token for user 1: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJleHAiOjE3MzIyMDQ1MjMsImlhdCI6MTczMjIwMDkyMywiaXNzIjoieW91ci1hcHAtbmFtZSJ9.nvEtuOIUGZ12Vj4Z8My2gBEtN255R8MtbLRhbuDNYsg
"""

def generate_secret_key():
    """
    Generates a 32-byte secure random secret key.
    """
    secret_key = secrets.token_hex(32)  # 64-character hexadecimal key
    print("Generated Secret Key:", secret_key)
    return secret_key

def generate_user_token(user_id, secret_key, expiration_hours=240):
    """
    Generates a JWT token for a user with the given secret key.
    
    Args:
        user_id (int): The ID of the user.
        secret_key (str): The secret key used for signing the token.
        expiration_hours (int): Expiration time in hours for the token.

    Returns:
        str: A JWT token.
    """
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(hours=expiration_hours)
    payload = {
        "user_id": user_id,
        "exp": expiration_time,  # Expiration time
        "iat": datetime.datetime.utcnow(),  # Issued at
        "iss": "your-app-name"  # Issuer
    }

    # Generate the token
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token

# Example Usage
if __name__ == "__main__":
    # Generate a secret key (store securely in practice)
    load_dotenv()
    secret_key = os.getenv("SECRET_KEY")

    # Generate a token for a user with ID 1
    user_id = 1
    token = generate_user_token(user_id, secret_key)

    print(f"Token for user {user_id}: {token}")
