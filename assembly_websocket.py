from typing import Optional
import assemblyai as aai
import pydub
import soundfile as sf
import sounddevice as sd
import pyaudio
import os

from assembly_db import SoundAssigner, OPENAI_API_KEY

aai.settings.api_key = "09485e2cc7b741d4aa2922da67f84094" 

assigner = SoundAssigner(chroma_path="ESC-50")

all_sounds = [
    "dog", "rain", "crying_baby", "door_knock", "helicopter",
    "rooster", "sea_waves", "sneezing", "mouse_click", "chainsaw",
    "pig", "crackling_fire", "clapping", "keyboard_typing", "siren",
    "cow", "crickets", "breathing", "door_wood_creaks", "car_horn",
    "frog", "chirping_birds", "coughing", "can_opening", "engine",
    "cat", "water_drops", "footsteps", "washing_machine", "train",
    "hen", "wind", "laughing", "vacuum_cleaner", "church_bells",
    "insects", "pouring_water", "brushing_teeth", "clock_alarm", "airplane",
    "sheep", "toilet_flush", "snoring", "clock_tick", "fireworks",
    "crow", "thunderstorm", "drinking_sipping", "glass_breaking", "hand_saw"
]

class MicrophoneStream:
    def __init__(self, sample_rate: int = 44_100, device_index: Optional[int] = None):
        """
        Creates a stream of audio from the microphone.

        Args:
            sample_rate: The sample rate to record audio at.
            device_index: The index of the input device to use. If None, uses the default device.
        """
        self._pyaudio = pyaudio.PyAudio()
        self.sample_rate = sample_rate

        self._chunk_size = int(self.sample_rate * 0.1)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=device_index,
        )

        self._open = True

    def __iter__(self):
        """
        Returns the iterator object.
        """

        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.
        """
        if not self._open:
            raise StopIteration

        try:
            return self._stream.read(self._chunk_size)
        except KeyboardInterrupt:
            raise StopIteration

    def close(self):
        """
        Closes the stream.
        """

        self._open = False

        if self._stream.is_active():
            self._stream.stop_stream()

        self._stream.close()
        self._pyaudio.terminate()



def play_audio(audio_path):
    "This function plays the audio file at the given path."

    
    pydub.AudioSegment.from_file(audio_path).export("temp.wav", format="wav")
    

    data, samplerate = sf.read("temp.wav")
    sd.play(data, samplerate)

def on_open(session_opened: aai.RealtimeSessionOpened):
    "This function is called when the connection has been established."

    print("Session ID:", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
    "This function is called when a new transcript has been received."

    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end="\r\n")
        filename, category = assigner.retrieve_src_file(transcript.text)
        if filename is not None:
            print(f"Playing sound for category '{category}'")
            play_audio(os.path.join("ESC-50-master/audio", filename))
        else:
            print("No sound found for the given text.")


def on_error(error: aai.RealtimeError):
    "This function is called when the connection has been closed."

    print("An error occured:", error)

def on_close():
    "This function is called when the connection has been closed."

    print("Closing Session")


transcriber = aai.RealtimeTranscriber(
    on_data=on_data,
    on_error=on_error,
    sample_rate=44_100,
    on_open=on_open, # optional
    on_close=on_close, # optional
    word_boost=all_sounds,
    end_utterance_silence_threshold=500
)

# Start the connection
transcriber.connect()

# Open a microphone stream
microphone_stream = MicrophoneStream()

# Press CTRL+C to abort
transcriber.stream(microphone_stream)

transcriber.close()