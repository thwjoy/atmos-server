import os
from pydub import AudioSegment
from pydub.effects import normalize

# Root directory containing subfolders with .wav files
INPUT_DIR = "./datasets/SFX_unnorm"
OUTPUT_DIR = "./datasets/SFX_normalized"

# Target dBFS level
TARGET_dBFS = -20.0

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def match_target_amplitude(sound, target_dBFS):
    """Normalize audio to match the target dBFS."""
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def normalize_wav(file_path, output_path):
    """Normalize the volume of a WAV file to the target dBFS."""
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Normalize the audio
        normalized_audio = match_target_amplitude(audio, TARGET_dBFS)
        
        # Export the normalized audio to the output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output subdirectory exists
        normalized_audio.export(output_path, format="wav")
        print(f"Normalized: {file_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_all_wavs_recursive(input_dir, output_dir):
    """Recursively process all WAV files in the directory tree."""
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                
                # Maintain the subfolder structure in the output directory
                relative_path = os.path.relpath(file_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                normalize_wav(file_path, output_path)

if __name__ == "__main__":
    # Process all WAV files in the input directory
    process_all_wavs_recursive(INPUT_DIR, OUTPUT_DIR)
    print("Volume normalization completed.")