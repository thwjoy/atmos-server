import time
import numpy as np
import pandas as pd
import torch
import string

from scipy.io import wavfile

import webrtcvad
from dtw import dtw
from dtaidistance.dtw import warping_paths
from scipy.ndimage import median_filter
import sounddevice as sd
import webrtcvad as vad
from scipy.signal import resample

import whisper


AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

def get_transcript_from_book(book, line_start, line_end):
    return "\n".join(book.split("\n")[line_start:line_end]) + "\n"

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


class RealTimeTranscriber:
    def __init__(self,
                 book,
                 line_offset,
                 chunk_duration=1,
                 language="English",
                 model_name="base",
                 token_window=1000,
                 audio_window=30,
                 vad_level=-1,
                 book_delim="\n"):
        
        self.language = language
        whisper.model.MultiHeadAttention.use_sdpa = False
        self.log_transcript = False
        # self.n_tokens = 250 # number of tokens in audio to process at a time
        self.model = whisper.load_model(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        # install hooks on the cross attention layers to retrieve the attention weights
        self.QKs = [None] * self.model.dims.n_text_layer

        # Define the hook function explicitly
        def save_attention_weights(module, input, output, index):
            self.QKs[index] = output[-1]  # Save the attention weights for each layer

        # Register the forward hooks explicitly
        for i, block in enumerate(self.model.decoder.blocks):
            block.cross_attn.register_forward_hook(lambda module, input, output, i=i: save_attention_weights(module, input, output, i))

        self.sampling_rate = 16000 # TODO fix why this has to be hardcoded
        self.chunk_size = self.sampling_rate * chunk_duration
        self.buffer_size = self.chunk_size
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_fill = 0
        self.audio_complete = np.empty(0, dtype=np.int16)
        self.duration_offset = 0
        self.token_offset = 0
        self.token_window = token_window # this should be at least 
        self.audio_window = audio_window # seconds
        self.curr_toks = []
        self.full_data = pd.DataFrame(columns=["word", "word_token", "begin", "end", "diff"])
        self.curr_data = pd.DataFrame(columns=["word", "word_token", "begin", "end", "diff"])

        self.book = book
        self.line_offset = line_offset
        line_width = 3
        self.transcription = get_transcript_from_book(book, self.line_offset, self.line_offset + line_width)
        self.line_offset += line_width
        self.tokenizer = whisper.tokenizer.get_tokenizer(self.model.is_multilingual, language=language)
        self.transcipt_tokens = self.tokenizer.encode(self.transcription)
        self.out_transcript = ""
        self.book_delim = book_delim
        self.delim_tok = self.tokenizer.encode(self.book_delim)[0]

        self.use_vad = False
        self.vad = webrtcvad.Vad()
        self.vad_level = vad_level
        if self.vad_level > -1:
            self.vad.set_mode(self.vad_level)  # Medium aggressiveness for speech detection
        self.accumulated_time = 0.0
        self.frame_duration = 30  # ms
        self.medfilt_width = 7
        self.qk_scale = 1.0

    def split_tokens_on_unicode(self, tokens: torch.Tensor):
        words = []
        word_tokens = []
        current_tokens = []
        
        for token in tokens.tolist():
            current_tokens.append(token)
            decoded = self.tokenizer.decode_with_timestamps(current_tokens)
            if "\ufffd" not in decoded:
                words.append(decoded)
                word_tokens.append(current_tokens)
            current_tokens = []
    
        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: torch.Tensor):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []
        
        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.tokenizer.eot and subword_tokens[0] is not self.tokenizer.no_speech
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)
        
        return words, word_tokens

    def audio_callback(self, indata):
        # Convert to int16 for VAD processing
        audio_chunk = indata[:, 0].astype(np.float32)
        audio_chunk_int16 = (audio_chunk * 32767).astype(np.int16)  # Convert to int16 for VAD

        frame_size = int(self.sampling_rate * (self.frame_duration / 1000.0))  # Samples per 30ms frame

        # Loop through the 30ms frames in the chunk
        for i in range(0, len(audio_chunk_int16), frame_size):
            frame = audio_chunk_int16[i:i + frame_size]
            self.accumulated_time += self.frame_duration / 1000.0  # Update the accumulated time
            # Check if the frame contains speech
            if (self.vad_level > -1 and self.vad.is_speech(frame.tobytes(), sample_rate=self.sampling_rate)) or self.vad_level == -1:
                # Only add frames with speech to the buffer
                frames_to_fill = min(self.buffer_size - self.buffer_fill, len(frame))
                self.audio_buffer[self.buffer_fill:self.buffer_fill + frames_to_fill] = frame[:frames_to_fill] / 32767.0  # Convert back to float32
                self.buffer_fill += frames_to_fill
                
                # Stop adding if the buffer is full
                if self.buffer_fill >= self.buffer_size:
                    break

    def process_audio_chunk(self, audio_chunk):
        # self.audio_callback(audio_chunk)
        # import pdb; pdb.set_trace()
        # if self.buffer_fill >= self.buffer_size:
        #     audio_chunk = self.audio_buffer.copy()
        new_audio = audio_chunk
        self.audio_complete = np.concatenate([self.audio_complete, new_audio])
        # self.audio_full = np.concatenate([self.audio_full, new_audio])

        # save the audio
        wavfile.write(f"buffers/buffer_vad_{self.vad_level}.wav",
                        self.sampling_rate,
                        self.audio_complete)

        if len(self.audio_complete[self.duration_offset:]) > (self.sampling_rate * self.audio_window):
            # append curr_data that has a timestamp less than self.duration_offset to full_data
            self.duration_offset += len(new_audio)
            copy_df = self.curr_data[self.curr_data.begin.apply(lambda begin: begin < (self.duration_offset) // self.sampling_rate)]
            # Filter out empty or all-NA entries in `data` before concatenation
            if not copy_df.empty and not copy_df.isna().all().all():
                self.full_data = pd.concat([self.full_data, copy_df.dropna(how="all")], ignore_index=True)
            self.curr_data = self.curr_data.drop(copy_df.index).reset_index(drop=True)
            
            # update duration offset and audio_full to include overlap from last word in full_data
            # import pdb; pdb.set_trace()
            # self.duration_offset = int(self.full_data.iloc[-1].end * self.sampling_rate)                

            curr_toks = [token for tokens in copy_df.word_token for token in tokens]
            _, paths = warping_paths(self.transcipt_tokens, curr_toks, psi=0)
            min_index = int(np.argmin(paths[:, -1]))
            # self.token_offset += min_index
            self.transcipt_tokens = self.transcipt_tokens[min_index:]

        # check if we have enough tokens in the buffer
        if len(self.transcipt_tokens) - self.token_offset < self.token_window:
            # add more tokens to the buffer
            new_line = get_transcript_from_book(self.book, self.line_offset, self.line_offset + 1)
            self.line_offset += 1
            self.transcipt_tokens += self.tokenizer.encode(new_line)

        self.transcribe_chunk()
        print(f"Current trsncription: {self.get_transcript()}")
        self.buffer_fill = 0

    def transcribe_chunk(self):
        # simulate the rest of the audio being silence at time 5s
        duration = len(self.audio_complete[self.duration_offset:])
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(self.audio_complete[self.duration_offset:]))
        tokens = torch.tensor(
            [
                *self.tokenizer.sot_sequence,
                self.tokenizer.timestamp_begin,
            ] + self.transcipt_tokens[self.token_offset:self.token_offset+self.token_window] + [
                self.tokenizer.no_speech,
                self.tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN,
                self.tokenizer.eot,
            ]
        )
        with torch.no_grad():
            logits = self.model(mel.unsqueeze(0), tokens.unsqueeze(0)).squeeze(0)
            if self.log_transcript:
                print("Raw transcript: ", self.tokenizer.decode(logits.argmax(dim=-1)))
            # TODO we need a way to filter out when audio doesn't correspond to the book, maybe use whisper without transcript?            

        weights = torch.cat(self.QKs)  # layers * heads * tokens * frames    
        weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
        weights = median_filter(weights, (1, 1, 1, self.medfilt_width))
        weights = torch.tensor(weights * self.qk_scale).softmax(dim=-1)
        w = weights / weights.norm(dim=-2, keepdim=True)
        matrix = w[-6:].mean(axis=(0, 1))

        alignment = dtw(-matrix.double().numpy()) # TODO we can probably speed this up

        jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
        jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN 
        jump_times += AUDIO_TIME_PER_TOKEN / AUDIO_SAMPLES_PER_TOKEN * self.duration_offset
        words, word_tokens = self.split_tokens_on_spaces(tokens) # TODO convert this to token timings

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        begin_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]

        stop_ind = np.argmax(end_times[np.where(end_times < duration)[0]])

        avg_jump_diffs = np.diff(begin_times)

        self.curr_data = pd.DataFrame([
            dict(word=word, word_token=word_token, begin=begin, end=end, diff=diff)
            for word, word_token, begin, end, diff in zip(words[:stop_ind], word_tokens[:stop_ind], begin_times[:stop_ind], end_times[:stop_ind], avg_jump_diffs[:stop_ind])
            if not word.startswith("<|") and word.strip() not in ".,!?、。" and diff > 0.01
        ])

        # if self.curr_data.empty:

        self.out_transcript = self.get_transcript()
        
    def get_transcript(self):
        words = self.full_data.word.tolist()
        if not self.curr_data.empty:
            words += self.curr_data.word.tolist()
        return "".join(words)
    
    def get_last_word_timestamp(self):
        time_stamp = 0.0
        if not self.curr_data.empty:
            time_stamp += self.curr_data.iloc[-1].end
        elif not self.full_data.empty:
            time_stamp += self.full_data.iloc[-1].end
        return time_stamp
    
    def get_df(self):
        return pd.concat([self.full_data, self.curr_data], ignore_index=True)

    def print_and_save_df(self, file_name):
        pd.set_option('display.max_rows', None)
        df = self.get_df()
        df.to_csv(file_name)
        print(df)

    def process_audio_file(self, audio_data):
        """Loads, resamples, normalizes, and processes an audio file in chunks."""

        # Process audio in chunks and measure execution time
        start_time = time.time()
        num_chunks = len(audio_data) // self.chunk_size
        for i in range(num_chunks + 1):
            start = i * self.chunk_size
            end = start + self.chunk_size
            audio_chunk = audio_data[start:end]
            print(f"Processing chunk {i+1}/{num_chunks + 1}, Length: {len(audio_chunk)}")
            self.process_audio_chunk(audio_chunk)

# record_audio(16000, 180, "buffers/buffer_long_pause.wav")

if __name__ == "__main__":
    # Define parameters for RealTimeTranscriber
    # load in the book
    with open("example.txt", "r") as file:
        book = file.read()
    line_offset = 0
    chunk_duration = 30  # Duration for each chunk in seconds
    audio_window = 30  # Window for audio processing in seconds
    audio_file = "example.wav"  # Path to the audio file to process
    target_sample_rate = 16000  # Target sample rate for processing

    # Load audio from file
    original_sample_rate, audio_data = wavfile.read(audio_file)
     
    # Resample if the sample rate doesn't match the target
    if original_sample_rate != target_sample_rate:
        num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
        audio_data = resample(audio_data, num_samples)
        audio_data = audio_data.astype(np.int16)  # convert back to int16 after resampling

    # Normalize audio data
    audio_data = audio_data.astype(np.float32) / 32767.0

    # Instantiate RealTimeTranscriber
    transcriber = RealTimeTranscriber(
        book=book,
        line_offset=line_offset,
        chunk_duration=chunk_duration,
        audio_window=audio_window
    )

    # Process the audio file
    start_time = time.time()
    transcriber.process_audio_file(audio_data)
    end_time = time.time()

    transcriber.print_and_save_df(audio_file.replace(".wav", ".csv"))
    print(transcriber.get_transcript())
    print(f"Time to process audio: {end_time - start_time} seconds")

