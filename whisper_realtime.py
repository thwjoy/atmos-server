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
from dtaidistance.dtw import warping_paths, best_path
from whisper.tokenizer import get_tokenizer
from scipy.ndimage import median_filter
from IPython.display import display, HTML
import sounddevice as sd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d


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
transcription = """ When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.

None of them noticed a large, tawny owl flutter past the window.

At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. "Little tyke," chortled Mr. Dursley as he left the house. He got into his car and backed out of number four's drive.

It was on the corner of the street that he noticed the first sign of something peculiar -- a cat reading a map. For a second, Mr. Dursley didn't realize what he had seen -- then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive -- no, looking at the sign; cats couldn't read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day. """

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
        if True: #vad.is_speech(frame.tobytes(), sample_rate=16000):
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

audio_file = "buffers/buffer_full.wav"
audio_data = wavfile.read(audio_file)[1]
total_samples = len(audio_data)

count = 0

# record an audio sample and save it as a wav file

# Function to record and save audio
def record_audio(sampling_rate, duration, output_file):
    print("Recording started...")
    # Record audio
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    
    # Normalize to int16 format for saving as WAV
    audio_data_int16 = np.int16(audio_data * 32767)
    
    # Save as WAV file
    wavfile.write(output_file, sampling_rate, audio_data_int16)
    print(f"Audio saved as {output_file}")

# record_audio(16000, 120, "buffers/buffer_full.wav")

duration_offset = 0
token_offset = 0

# Function to calculate DTW distance between subsequences
def find_best_start_index(full_list, approximate):
    min_distance = float('inf')
    best_start_index = None

    for start in range(len(full_list) - len(approximate) + 1):
        # Extract the subsequence from the full list
        sub_list = full_list[start:start + len(approximate)]
        
        # Calculate DTW distance between approximate and sub_list
        distance, _ = fastdtw(sub_list, approximate)
        
        # Update the minimum distance and best start index
        if distance < min_distance:
            min_distance = distance
            best_start_index = start

    return best_start_index, min_distance

# full_data with empty data frame
full_data = pd.DataFrame({
    "word": [],
    "begin": [],
    "end": [],
    "diff": []
})

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
        if duration // AUDIO_SAMPLES_PER_TOKEN > 1500:
            # shift the audio buffer to the right
            duration_offset += len(new_audio)

            # get words from data frame
            duration_s = (duration_offset + chunk_size) / sampling_rate 
            times = data.end.tolist()
            # get index where times are less than duration_s
            index = np.argmax(np.array(times) > duration_s)

            # insert data from Pandas data up to index index into data_full
            full_data = pd.concat([full_data, data.iloc[:index]], ignore_index=True)

            # now we need to remove these tokens from the transcript_tokens
            full_words = full_data.word.tolist()
            
            curr_toks = tokenizer.encode("".join(full_words))

            # Compute DTW warping paths
            distance, paths = warping_paths(transcipt_tokens, curr_toks, psi=0)
            alignment_path = best_path(paths)

            # Find the minimum distance in the last row
            min_distances = paths[:, -1]  # Get matching for first index
            min_index = int(np.argmin(min_distances))
            min_distance = min_distances[min_index]
            token_offset = min_index
            # print(f"Duration s: {duration_s}")
            # print(f"Time: {times}")
            print(f"Best Start Index: {token_offset}")
            # print(f"Words: {full_words}")


        audio = audio_full[duration_offset:]

        # simulate the rest of the audio being silence at time 5s
        duration = len(audio)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        tokens = torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.timestamp_begin,
            ] + transcipt_tokens[token_offset:token_offset+200] + [
                tokenizer.no_speech,
                tokenizer.timestamp_begin + (duration - duration_offset) // AUDIO_SAMPLES_PER_TOKEN,
                tokenizer.eot,
            ]
        )
        with torch.no_grad():
            logits = model(mel.unsqueeze(0), tokens.unsqueeze(0)).squeeze(0)
            
        weights = torch.cat(QKs)  # layers * heads * tokens * frames    
        weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
        weights = median_filter(weights, (1, 1, 1, medfilt_width))
        weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
        w = weights / weights.norm(dim=-2, keepdim=True)
        matrix = w[-6:].mean(axis=(0, 1))

        alignment = dtw(-matrix.double().numpy())

        jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
        jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN 
        jump_times += AUDIO_TIME_PER_TOKEN / AUDIO_SAMPLES_PER_TOKEN * duration_offset
        words, word_tokens = split_tokens_on_spaces(tokens) 

        # display the word-level timestamps in a table
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        begin_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]

        # we only want to display words up to the current time
        stop_ind = np.argmax(end_times[np.where(end_times < duration)[0]])


        avg_jump_diffs = np.diff(begin_times)

        data = pd.DataFrame([
            dict(word=word, begin=begin, end=end, diff=diff)
            for word, begin, end, diff in zip(words[:stop_ind], begin_times[:stop_ind], end_times[:stop_ind], avg_jump_diffs[:stop_ind])
            if not word.startswith("<|") and word.strip() not in ".,!?、。" and diff > 0.01
        ])

        display("".join(full_data.word.tolist() + data.word.tolist()))
        # display("".join(data.word.tolist()))    
        # display(data)
        buffer_fill = 0


