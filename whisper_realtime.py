import io
import os
import time
import numpy as np
import pandas as pd
import torch
import string

# import urllib
# import tarfile
# import torchaudio
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import matplotlib.ticker as ticker
from scipy.io import wavfile
# from tqdm.notebook import tqdm

import webrtcvad
import soundfile as sf
from dtw import dtw
from whisper.tokenizer import get_tokenizer
from scipy.ndimage import median_filter
from IPython.display import display, HTML
import sounddevice as sd


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000

lang = "en_us"
language = "English"

from whisper import whisper
whisper.model.MultiHeadAttention.use_sdpa = False
model = whisper.load_model("base")


# from IPython.display import display, HTML
# from whisper.tokenizer import get_tokenizer
# from dtw import dtw
# from scipy.ndimage import median_filter

AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

medfilt_width = 7
qk_scale = 1.0

tokenizer = get_tokenizer(model.is_multilingual, language=language)

def split_tokens_on_unicode(tokens: torch.Tensor):
    words = []
    word_tokens = []
    current_tokens = []
    
    for token in tokens.tolist():
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps(current_tokens)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []
    
    return words, word_tokens

def split_tokens_on_spaces(tokens: torch.Tensor):
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens)
    words = []
    word_tokens = []
    
    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eot and subword_tokens[0] is not tokenizer.no_speech
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
    
    return words, word_tokens

# install hooks on the cross attention layers to retrieve the attention weights
QKs = [None] * model.dims.n_text_layer

# Define the hook function explicitly
def save_attention_weights(module, input, output, index):
    QKs[index] = output[-1]  # Save the attention weights for each layer

# Register the forward hooks explicitly
for i, block in enumerate(model.decoder.blocks):
    block.cross_attn.register_forward_hook(lambda module, input, output, i=i: save_attention_weights(module, input, output, i))

# for the first 10 examples in the dataset
sampling_rate = 16000
chunk_duration = 1
chunk_size = sampling_rate * chunk_duration  # Number of samples per chunk
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE
transcription = "   When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts.  "

# Initialize an empty list to store audio data in real-time
buffer_size = int(chunk_size)
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
buffer_fill = 0 # Track the buffer fill level
accumulated_time = 0.0  # Initialize accumulated time in seconds


# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(0)  # 0-3; higher values are more aggressive at filtering non-speech

# Audio callback function to store audio in the buffer
def audio_callback(indata, frames, time, status):
    global audio_buffer, buffer_fill, accumulated_time
    if status:
        print(f"Audio callback status: {status}")

    # Convert to int16 for VAD processing
    audio_chunk = indata[:, 0].astype(np.float32)
    audio_chunk_int16 = (audio_chunk * 32767).astype(np.int16)  # Convert to int16 for VAD

    # Define 30ms frame size
    frame_duration = 30  # ms
    frame_size = int(16000 * (frame_duration / 1000.0))  # Samples per 30ms frame

    # Loop through the 30ms frames in the chunk
    for i in range(0, len(audio_chunk_int16), frame_size):
        frame = audio_chunk_int16[i:i + frame_size]
        accumulated_time += frame_duration / 1000.0  # Update the accumulated time
        # Check if the frame contains speech
        if vad.is_speech(frame.tobytes(), sample_rate=16000):
            # Only add frames with speech to the buffer
            frames_to_fill = min(buffer_size - buffer_fill, len(frame))
            audio_buffer[buffer_fill:buffer_fill + frames_to_fill] = frame[:frames_to_fill] / 32767.0  # Convert back to float32
            buffer_fill += frames_to_fill
            
            # Stop adding if the buffer is full
            if buffer_fill >= buffer_size:
                break

def get_audio_index_at_time(time: float):
    return int(time * whisper.audio.SAMPLE_RATE)

transcipt_tokens = tokenizer.encode(transcription)

audio_file = "buffers/buffer.wav"
audio_data = wavfile.read(audio_file)[1]
total_samples = len(audio_data)

count = 0
# Real-time recording and processing
# with sd.InputStream(callback=audio_callback, channels=1, samplerate=sampling_rate):
#     print("Starting real-time transcription...")
#     audio_full = torch.empty(0)  # to store the entire audio so far
#     # Loop until the desired duration or break condition
#     while True:
#         # Wait until the audio buffer is full

audio_full = torch.empty(0)  # to store the entire audio so far
for start in range(0, total_samples, chunk_size):
    end = min(start + chunk_size, total_samples)
    chunk = audio_data[start:end].reshape(-1, 1)
    
    # Invoke the callback manually with the current chunk
    audio_callback(chunk, len(chunk), None, None)

    if buffer_fill >= buffer_size:
        audio_chunk = audio_buffer.copy()

        new_audio = torch.tensor(np.array(audio_chunk), dtype=torch.float32)
        audio_full = torch.cat([audio_full, new_audio])
        
        duration = len(audio_full)
        audio = audio_full[:duration]

        # save the audio to a file
        wavfile.write("buffers/buffer_vad.wav", whisper.audio.SAMPLE_RATE, audio.cpu().numpy())

        # simulate the rest of the audio being silence at time 5s
        duration = len(audio)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        tokens = torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.timestamp_begin,
            ] + transcipt_tokens + [
                tokenizer.no_speech,
                tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN,
                tokenizer.eot,
            ]
        )
        with torch.no_grad():
            logits = model(mel.unsqueeze(0), tokens.unsqueeze(0)).squeeze(0)
            # print average entropy accross channels enropy of logits

        print(mel.shape, tokens.shape)
        weights = torch.cat(QKs)  # layers * heads * tokens * frames    
        weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
        weights = median_filter(weights, (1, 1, 1, medfilt_width))
        weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
        w = weights / weights.norm(dim=-2, keepdim=True)
        matrix = w[-6:].mean(axis=(0, 1))

        alignment = dtw(-matrix.double().numpy())

        jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
        jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN
        words, word_tokens = split_tokens_on_spaces(tokens) 

        # display the word-level timestamps in a table
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        begin_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]

        # we only want to display words up to the current time
        stop_ind = np.argmax(end_times[np.where(end_times < duration)[0]])

        # TODO find a better way of doing this
        if jump_times.mean() > 0.5:
            print("Skipping due to too frequent words")
        
            data = [
                dict(word=word, begin=begin, end=end)
                for word, begin, end in zip(words[:stop_ind], begin_times[:stop_ind], end_times[:stop_ind])
                if not word.startswith("<|") and word.strip() not in ".,!?、。" 
            ]

            print("Time:", duration)   
            display(pd.DataFrame(data))

        buffer_fill = 0


