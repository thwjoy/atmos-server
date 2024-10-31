from pathlib import Path
from openai import OpenAI
import os 
import io
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

with open("harry_potter_1_db/harry_potter_1.txt", "r") as file:
    book = file.read()

book_lines = book.split("\n")

client = OpenAI()
output_file_path = Path(__file__).parent / "output_audio.wav"

# Initialize an empty array to store combined audio
combined_audio = np.array([], dtype=np.int16)


for b in tqdm(range(122, 235)):
    if not book_lines[b] == "":
        print("Transcribing line:", book_lines[b])
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            response_format="wav",
            input=book_lines[b],
        )

        # Load the audio response into a BytesIO buffer and read it
        audio_data = io.BytesIO(response.content)
        sample_rate, audio_segment = wavfile.read(audio_data)

        # Concatenate the new audio segment with the existing combined audio
        combined_audio = np.concatenate((combined_audio, audio_segment), axis=0)

        # Save the combined audio to the output file
        wavfile.write(output_file_path, sample_rate, combined_audio)
        print(f"Combined audio saved to {output_file_path}")




