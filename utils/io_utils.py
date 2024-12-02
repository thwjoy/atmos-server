import asyncio
import os
import struct
import uuid
# import librosa
import numpy as np
import logging 

HEADER_FORMAT = ">5sI16sIII"


# def load_wav_from_bytes(byte_data):
#     # Use soundfile to read from the bytes object
#     with io.BytesIO(byte_data) as bytes_io:
#         audio_data, sample_rate = sf.read(bytes_io, dtype='float32')  # or 'float32' for normalized values
#     return sample_rate, np.array(audio_data)

# def audio_to_bytes(audio_data, sample_rate):
#     # Create an in-memory byte buffer
#     with io.BytesIO() as buffer:
#         # Write the audio data to the buffer as a WAV file
#         sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        
#         # Retrieve the byte data from the buffer
#         audio_bytes = buffer.getvalue()
#        
#    return audio_bytes

logger=logging.getLogger(__name__)

# def resample_audio(audio, orig_sample_rate, target_sample_rate=44100):
#     # Convert the audio to a numpy array if it's not already
#     if not isinstance(audio, np.ndarray):
#         audio = np.array(audio)
    
#     # Resample the audio to the target sample rate
#     resampled_audio = librosa.resample(audio, orig_sample_rate, target_sample_rate)
#     return resampled_audio

async def read_audio_in_chunks(audio_path, chunk_size):
        """Read an audio file in chunks asynchronously."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
    
        loop = asyncio.get_running_loop()
        with open(audio_path, "rb") as audio_file:
            while chunk := await loop.run_in_executor(None, audio_file.read, chunk_size):
                yield chunk

async def send_with_backpressure(websocket, data, buffer_limit=10_000_000):
    """Send data to the WebSocket with backpressure handling."""
    while websocket.transport and websocket.transport.get_write_buffer_size() > buffer_limit:
        await asyncio.sleep(0.1)  # Wait for the buffer to drain
    
    if websocket.transport:
        await websocket.send(data)
    else:
        raise RuntimeError("WebSocket transport is not available. Connection might be closed.")

async def send_audio_with_header(websocket, audio_path, indicator, chunk_size=1024 * 512):
    """Send audio in chunks with header containing packet size, packet count, sample rate, and unique sequence ID."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    total_packets = (file_size + chunk_size - 1) // chunk_size  # Calculate total packet count

    # Retrieve sample rate (assume WAV format)
    sample_rate = 44100

    # Generate a unique sequence ID for this transmission
    sequence_id = uuid.uuid4().bytes  # UUID as a 32-byte binary string

    queue = asyncio.Queue()

    async def producer():
        packet_count = 0
        async for chunk in read_audio_in_chunks(audio_path, chunk_size):
            await queue.put((packet_count, chunk))  # Add packet count with chunk
            packet_count += 1
        await queue.put(None)  # Signal completion

    async def consumer():
        packet_sent = 0

        while True:
            try:
                data = await queue.get()
                if data is None:  # Completion signal
                    break

                packet_count, chunk = data

                # Build the header
                ind = indicator[:5].ljust(5)  # Ensure the indicator is 5 bytes
                header = struct.pack(
                    HEADER_FORMAT,       # Updated format
                    ind.encode(),        # Indicator
                    file_size,          # Size of this packet
                    sequence_id,         # Unique sequence ID
                    packet_count,        # Packet count (sequence number)
                    total_packets,       # Total packets
                    sample_rate          # Sample rate
                )

                # Send the chunk with the header
                await send_with_backpressure(websocket, header + chunk)
                packet_sent += 1

            except websocket.ConnectionClosed as e:
                logger.error(f"WebSocket closed during send: {e}")
                raise

    try:
        await asyncio.gather(
            asyncio.create_task(producer()),
            asyncio.create_task(consumer())
        )
        logger.info(f"Successfully sent packets with Sequence ID: {uuid.UUID(bytes=sequence_id)}.")
    except Exception as e:
        logger.error(f"Error in send_audio_with_header: {e}")

async def send_transcript_audio_with_header(websocket, audio, sample_rate, chunk_size=1024 * 512):
    ind = "STORY"
    file_size = len(audio)
    sequence_id = uuid.uuid4().bytes  # UUID as a 32-byte binary string
    packet_count = 0
    total_packets = (file_size + chunk_size - 1) // chunk_size  # Calculate total packet count
    # Divide the audio into chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        # Add a header with the chunk size
        header = struct.pack(
            HEADER_FORMAT,       # Updated format
            ind.encode(),        # Indicator
            file_size,          # Size of this packet
            sequence_id,         # Unique sequence ID
            packet_count,        # Packet count (sequence number)
            total_packets,       # Total packets
            sample_rate          # Sample rate
        )
        packet_count += 1
        data_with_header = header + chunk
        await send_with_backpressure(websocket, data_with_header)