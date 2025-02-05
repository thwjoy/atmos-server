from pathlib import Path
from openai import OpenAI
import os 
import io
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()
output_file_path = Path(__file__).parent / "wicked_hw_ending.wav"

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    response_format="wav",
    input="""The Golden Amulet.
With the final riddle solved, the Golden Amulet appeared in a flash of light. Its power coursed through the air, restoring balance to the land. The witches stood together, their shared triumph bridging the rift between them.
“You’re not alone anymore, Elphie,” Glinda said softly.
Elphaba hesitated, then allowed herself a small smile. “Maybe not. But there’s still so much to fix.”
Hand in hand, they walked into the future, ready to rebuild Oz—not just its roads and gears, but the trust and hope its people had lost.
The End.
""",
)

# Load the audio response into a BytesIO buffer and read it
audio_data = io.BytesIO(response.content)
sample_rate, audio_segment = wavfile.read(audio_data)

# Save the combined audio to the output file
wavfile.write(output_file_path, sample_rate, audio_segment)
print(f"Combined audio saved to {output_file_path}")




