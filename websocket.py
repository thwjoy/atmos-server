import asyncio
import websockets
import ssl

certfile = "/home/ubuntu/certs/myatmos_chain.crt"
keyfile = "/home/ubuntu/certs/private.key"

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)

# Define the WebSocket handler
async def echo(websocket, path):
    print(f"New connection established: {path}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            # Echo the message back to the client
            await websocket.send(f"Echo: {message}")
    except websockets.ConnectionClosed as e:
        print(f"Connection closed with code {e.code}, reason: {e.reason}")
    finally:
        print(f"Connection closed for {path}")

# Start the WebSocket server
async def start_server():
    async with websockets.serve(echo, "0.0.0.0", 5001, ssl=ssl_context):
        print("WebSocket server started on ws://0.0.0.0:5001")
        await asyncio.Future()  # Keep the server running indefinitely

if __name__ == "__main__":
    asyncio.run(start_server())