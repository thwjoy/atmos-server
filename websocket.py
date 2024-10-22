import asyncio
import json
import struct
import websockets
import ssl
from test_RAG import OPENAI_API_KEY, RAGAutoComplete
import os
import base64

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# certfile = "/home/ubuntu/certs/myatmos_chain.crt"
# keyfile = "/home/ubuntu/certs/private.key"

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)

rag_system = RAGAutoComplete(doc_path=os.path.join("harry_potter_1_db", "harry_potter_1.txt"),
                             chroma_path="harry_potter_1_db",
                             api_key=OPENAI_API_KEY)

# Define the WebSocket handler
async def echo(websocket, path):
    print(f"New connection established: {path}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            # Echo the message back to the client
            chunk_data = rag_system.get_audio(message)

                # check if the prompt contains silence
            if chunk_data["audio_db"][0]['prompt'] == "silence":
                payload = "Silence detected"
                print(payload)

            if len(chunk_data["audio_db"]) == 0:
                payload = "No audio found for the given text."
                print(payload)

            if chunk_data["chunks_scores"] > 0.25:
                payload = "Not sending audio, bad score %.3f" % chunk_data["chunks_scores"]
                print(payload)
                print("Audio data:", chunk_data["audio_db"][0])

            else:
                print("Sending the following data:")
                print("Score %.3f" % chunk_data["chunks_scores"])
                print("Audio data:", chunk_data["audio_db"][0])

                # Open and encode the audio file in base64
                with open(chunk_data["audio_db"][0]['audio'], 'rb') as file:
                    file_data = file.read()

                metadata = {
                    "prompt": chunk_data["audio_db"][0]['prompt'],
                    "score": chunk_data["chunks_scores"],
                    "chunk": chunk_data["audio_db"][0]['chunk'],

                }

                # payload = {
                #     "audio": file_data,
                #     "metadata": metadata
                # }
                metadata_json = json.dumps(metadata).encode('utf-8')

                # Create the payload (4-byte metadata length + metadata + binary file data)
                metadata_len = struct.pack('>I', len(metadata_json))
                payload = metadata_len + metadata_json + file_data

                total_size = len(metadata_len) + len(metadata_json) + len(file_data)
                print(f"Total size of data to send: {total_size} bytes. Metadata len: {len(metadata_json)}")

            await websocket.send(payload)

    except websockets.ConnectionClosed as e:
        print(f"Connection closed with code {e.code}, reason: {e.reason}")
    finally:
        print(f"Connection closed for {path}")

def handle_audio_chunk(message):
    print('message received', len(message), type(message))
    chunk_data = rag_system.get_audio(message['msg'])

    # check if the prompt contains silence
    if chunk_data["audio_db"][0]['prompt'] == "silence":
        print("Silence detected")
        return
    if len(chunk_data["audio_db"]) == 0:
        print("No audio found for the given text.")
        return
    if chunk_data["chunks_scores"] > 0.25:
        print("Not sending audio, bad score %.3f" % chunk_data["chunks_scores"])
        print("Audio data:", chunk_data["audio_db"][0])
        return
    else:
        print("Sending the following data:")
        print("Score %.3f" % chunk_data["chunks_scores"])
        print("Audio data:", chunk_data["audio_db"][0])

        with open(chunk_data["audio_db"][0]['audio'], 'rb') as file:
            file_data = file.read()

        metadata = {
            "prompt": chunk_data["audio_db"][0]['prompt'],
            "score": chunk_data["chunks_scores"],
            "chunk": chunk_data["audio_db"][0]['chunk'],

        }
        
        # Send the zip file as a response
        emit('file_data', {"audio.wav": file_data, "metadata.json": metadata})

# Start the WebSocket server
async def start_server():
    async with websockets.serve(echo, "0.0.0.0", 5001, ssl=None):
        print("WebSocket server started on ws://0.0.0.0:5001")
        await asyncio.Future()  # Keep the server running indefinitely

if __name__ == "__main__":
    asyncio.run(start_server())