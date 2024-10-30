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

from whisper import whisper


AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

transcription = """ When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.

None of them noticed a large, tawny owl flutter past the window.

At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. "Little tyke," chortled Mr. Dursley as he left the house. He got into his car and backed out of number four's drive.

It was on the corner of the street that he noticed the first sign of something peculiar -- a cat reading a map. For a second, Mr. Dursley didn't realize what he had seen -- then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive -- no, looking at the sign; cats couldn't read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day. 

But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn’t help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr Dursley couldn’t bear people who dressed in funny clothes – the get-ups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr Dursley was enraged to see that a couple of them weren’t young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr Dursley that this was probably some silly stunt – these people were obviously collecting for something … yes, that would be it. The traffic moved on, and a few minutes later, Mr Dursley arrived in the Grunnings car park, his mind back on drills.
"""
# transcription = "Unfortunately, stunning traffic flow is difficult because driver behavior cannot be predicted with 100% certainty."

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
    def __init__(self, transcription, language="English", model_name="base", sampling_rate=16000, chunk_duration=1):
        self.language = language
        whisper.model.MultiHeadAttention.use_sdpa = False
        self.log_transcript = True
        self.n_tokens = 250 # number of tokens in audio to process at a time
        self.model = whisper.load_model(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        # install hooks on the cross attention layers to retrieve the attention weights
        self.QKs = [None] * self.model.dims.n_text_layer

        # Define the hook function explicitly
        def save_attention_weights(module, input, output, index):
            self.QKs[index] = output[-1]  # Save the attention weights for each layer

        # Register the forward hooks explicitly
        for i, block in enumerate(self.model.decoder.blocks):
            block.cross_attn.register_forward_hook(lambda module, input, output, i=i: save_attention_weights(module, input, output, i))

        self.tokenizer = whisper.tokenizer.get_tokenizer(self.model.is_multilingual, language=language)
        self.transcipt_tokens = self.tokenizer.encode(transcription)

        self.sampling_rate = sampling_rate
        self.chunk_size = sampling_rate * chunk_duration
        self.buffer_size = self.chunk_size
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_fill = 0
        self.audio_full = np.empty(0, dtype=np.int16)
        self.duration_offset = 0
        self.token_offset = 0
        self.token_window = 200
        self.curr_toks = []
        self.full_data = pd.DataFrame(columns=["word", "begin", "end", "diff"])
        self.curr_data = pd.DataFrame(columns=["word", "begin", "end", "diff"])

        self.use_vad = False
        self.vad = webrtcvad.Vad()
        self.vad_level = 3
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

        frame_size = int(16000 * (self.frame_duration / 1000.0))  # Samples per 30ms frame

        # Loop through the 30ms frames in the chunk
        for i in range(0, len(audio_chunk_int16), frame_size):
            frame = audio_chunk_int16[i:i + frame_size]
            self.accumulated_time += self.frame_duration / 1000.0  # Update the accumulated time
            # Check if the frame contains speech
            if (self.vad_level > -1 and self.vad.is_speech(frame.tobytes(), sample_rate=16000)) or self.vad_level == -1:
                # Only add frames with speech to the buffer
                frames_to_fill = min(self.buffer_size - self.buffer_fill, len(frame))
                self.audio_buffer[self.buffer_fill:self.buffer_fill + frames_to_fill] = frame[:frames_to_fill] / 32767.0  # Convert back to float32
                self.buffer_fill += frames_to_fill
                
                # Stop adding if the buffer is full
                if self.buffer_fill >= self.buffer_size:
                    break

    def process_audio_chunk(self, audio_chunk):
        self.audio_callback(audio_chunk)
        if self.buffer_fill >= self.buffer_size:
            audio_chunk = self.audio_buffer.copy()
            new_audio = audio_chunk
            self.audio_full = np.concatenate([self.audio_full, new_audio])

            # save the audio
            wavfile.write(f"buffers/buffer_vad_{self.vad_level}.wav",
                          self.sampling_rate,
                          self.audio_full)

            if len(self.audio_full) // AUDIO_SAMPLES_PER_TOKEN > self.n_tokens:
                # shift the audio buffer to the right
                self.duration_offset += len(new_audio)

                # get words from data frameå
                duration_s = (self.duration_offset + self.chunk_size) / self.sampling_rate 
                if not self.curr_data.empty:
                    times = self.curr_data.end.tolist()
                    # get index where times are less than duration_s
                    index = np.argmax(np.array(times) > duration_s)

                    # insert data from Pandas data up to index index into data_full
                    self.full_data = pd.concat([self.full_data, self.curr_data.iloc[:index]], ignore_index=True)
                    
                    self.curr_toks += self.tokenizer.encode("".join(self.curr_data.iloc[:index].word.tolist()))

                    # Compute DTW warping paths
                    _, paths = warping_paths(self.transcipt_tokens,
                                                    self.curr_toks, psi=0)

                    # Find the minimum distance in the last row
                    min_distances = paths[:, -1]  # Get matching for first index
                    min_index = int(np.argmin(min_distances))
                    self.token_offset = min_index

            self.transcribe_chunk()
            self.buffer_fill = 0

    def transcribe_chunk(self):
        audio = self.audio_full[self.duration_offset:]

        # simulate the rest of the audio being silence at time 5s
        duration = len(audio)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        tokens = torch.tensor(
            [
                *self.tokenizer.sot_sequence,
                self.tokenizer.timestamp_begin,
            ] + self.transcipt_tokens[self.token_offset:self.token_offset+self.token_window] + [
                self.tokenizer.no_speech,
                self.tokenizer.timestamp_begin + (duration - self.duration_offset) // AUDIO_SAMPLES_PER_TOKEN,
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
        words, word_tokens = self.split_tokens_on_spaces(tokens) 

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        begin_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]

        stop_ind = np.argmax(end_times[np.where(end_times < duration)[0]])

        avg_jump_diffs = np.diff(begin_times)

        self.curr_data = pd.DataFrame([
            dict(word=word, begin=begin, end=end, diff=diff)
            for word, begin, end, diff in zip(words[:stop_ind], begin_times[:stop_ind], end_times[:stop_ind], avg_jump_diffs[:stop_ind])
            if not word.startswith("<|") and word.strip() not in ".,!?、。" and diff > 0.01
        ])
        
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


# record_audio(16000, 180, "buffers/buffer_long_pause.wav")

if __name__ == "__main__":
    # Load audio from a file and process in chunks
    audio_file = "buffers/buffer_long_pause.wav"
    rate, audio_data = wavfile.read(audio_file)

    # # we need to ensure that the audio data is normalised
    audio_data = audio_data.astype(np.float32) / 32767.0  

    transcriber = RealTimeTranscriber(transcription=transcription,
                                      sampling_rate=rate,
                                      chunk_duration=1)
    

    # Process the audio in chunks
    num_chunks = len(audio_data) // transcriber.chunk_size
    for i in range(num_chunks):
        start = i * transcriber.chunk_size
        end = start + transcriber.chunk_size
        audio_chunk = audio_data[start:end].reshape(-1, 1)
        # this is where we send an audio_chunk
        transcriber.process_audio_chunk(audio_chunk)
        print(f"\r\n Transcript so far: {transcriber.get_transcript()}")
        print(f"Last word timestamp: {transcriber.get_last_word_timestamp():.2f} seconds")

    pd.set_option('display.max_rows', None)
    print(transcriber.full_data)

