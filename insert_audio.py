import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import argparse
import ast

import torch
import whisper
from dtaidistance.dtw import warping_paths

model = whisper.load_model("base").to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="English")

def overlay_audio_with_wavfile(main_audio_path, words_df_path, timestamps_df_path, output_path):
    # Load the main audio file
    sample_rate, main_waveform = wavfile.read(main_audio_path)
    main_waveform = main_waveform.astype(np.int16)
    
    # Ensure stereo format (2 channels) for main audio if not already
    if main_waveform.ndim == 1:
        main_waveform = np.stack([main_waveform, main_waveform], axis=1)
    
    main_audio_length = main_waveform.shape[0]  # in samples
    
    # Load dataframes
    words_df = pd.read_csv(words_df_path)
    timestamps_df = pd.read_csv(timestamps_df_path)

    # # Ensure both dataframes have consistent whitespace and case
    # words_df['Word'] = words_df['Word'].str.strip().str.lower()
    # timestamps_df['word'] = timestamps_df['word'].str.strip().str.lower()
    
    # Convert timestamps in seconds to samples for placement
    timestamps_df['start_sample'] = (timestamps_df['begin'] * sample_rate).astype(int)
    timestamps_df['end_sample'] = (timestamps_df['end'] * sample_rate).astype(int)
    
    # Create a copy of the main waveform to overlay sounds
    overlay_waveform = main_waveform.copy()

    # tokenize each word in words_df

    # Ensure the 'word_tokens' column exists in words_df
    # if 'word_token' not in words_df.columns:
    #     words_df['word_token'] = None

    # for idx, row in words_df.iterrows():
    #     if pd.notnull(row['Word']) and row['Word'].strip() != '':
    #         tokens = tokenizer.encode(" " + row['Word']) # TODO fix this
    #         words_df.at[idx, 'word_token'] = tokens
    #         if row['Sound Description'] != 'silence':
    #             words_df.at[idx, 'audio'] = f"harry_potter_1_db_v2/sounds/audio_{idx+2}.wav"

    # # save words_df
    # words_df.to_csv('words_df.csv', index=False)
    words_df = pd.read_csv('words_df.csv')

    if 'begin' not in words_df.columns:
        words_df['begin'] = None

    if 'end' not in words_df.columns:
        words_df['end'] = None

    # loop through all rows of words_df, find where the seqnece matches in timestamps,
    # then overlay the sound effect

    def flattern_list(list):
        return [item for sublist in list if sublist is not None for item in sublist if item is not None]

    # words_toks = flattern_list(words_df['word_token'].tolist())
    # timestamps_toks = flattern_list(timestamps_df['word_token'].apply(ast.literal_eval).tolist())

    # # get the index in timestamps where this audio occurs
    # _, paths = warping_paths(words_toks, timestamps_toks, psi=0) # TODO fix this

    # # save paths
    # np.save('paths.npy', paths)
    paths = np.load('paths.npy')

    token_count = 0

    last_end_sample = 0
    for idx, row in words_df.iterrows():
        token_count += len(ast.literal_eval(row['word_token'])) if pd.notnull(row['word_token']) else 0
        if row['Sound Description'] != 'silence':
            # find where this word fits in timestamps
            # if idx > 100:
            #     break

            current_token_timestamp_idx = int(np.argmin(paths[token_count, :]))
            # we now need to get the timestamp index for this token_count in timestamps_df
            current_timestamp_idx = timestamps_df.index[timestamps_df['word_token'].apply(ast.literal_eval).apply(len).cumsum() >= current_token_timestamp_idx][0]
            start_sample = timestamps_df.iloc[current_timestamp_idx]['start_sample']

            # Load the sound effect
            if row['audio'] != row['audio']:
                continue

            sfx_sample_rate, sound_waveform = wavfile.read(row['audio'])
            
            # Check if the sample rate matches the target
            if sample_rate != sfx_sample_rate:
                # Calculate the number of samples needed for the new sample rate
                num_samples = int(len(sound_waveform) * sample_rate / sfx_sample_rate)
                sound_waveform = resample(sound_waveform, num_samples)
    
            sound_waveform = sound_waveform.astype(np.int16)
            
            # If the sound effect is mono, duplicate it for stereo
            if sound_waveform.ndim == 1:
                sound_waveform = np.stack([sound_waveform, sound_waveform], axis=1)
            
            # Calculate the overlay end sample based on the length of the sound effect
            end_sample = start_sample + sound_waveform.shape[0]

            words_df.at[idx, 'begin'] = start_sample / sample_rate
            words_df.at[idx, 'end'] = end_sample / sample_rate

            # Ensure we don't exceed the main audio length
            if end_sample > main_audio_length:
                end_sample = main_audio_length
                sound_waveform = sound_waveform[:end_sample - start_sample, :]

            # if start_sample > last_end_sample:
            overlay_waveform[start_sample:end_sample, :] += sound_waveform

            last_end_sample = end_sample

    # Save the output
    words_df.to_csv('words_df.csv', index=False)
    wavfile.write(output_path, sample_rate, overlay_waveform)

def main():
    parser = argparse.ArgumentParser(description="Overlay sound effects onto an audio file based on word timestamps.")
    parser.add_argument("--main_audio_path", type=str, help="Path to the main audio file")
    parser.add_argument("--words_df_path", type=str, help="Path to the CSV file containing words and audio file paths")
    parser.add_argument("--timestamps_df_path", type=str, help="Path to the CSV file containing word timestamps")
    parser.add_argument("--output_path", type=str, help="Path to save the output audio file with overlays")

    args = parser.parse_args()
    
    overlay_audio_with_wavfile(
        args.main_audio_path,
        args.words_df_path,
        args.timestamps_df_path,
        args.output_path
    )

if __name__ == "__main__":
    main()
