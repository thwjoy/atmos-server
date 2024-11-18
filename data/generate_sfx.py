import argparse
import ast
import os
import csv
from pathlib import Path
import re
from tkinter.filedialog import Open
from openai import OpenAI
from elevenlabs import ElevenLabs

# add constants to path
import sys
sys.path.append('..')
from keys import OPENAI_API_KEY
from keys import ELEVENLABS_API_KEY


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

# Placeholder for the audio generation function
def generate_audio(description):
    print(f"Generating audio for: {description}")
    response = client.text_to_sound_effects.convert(
        text=description,
        duration_seconds=3,
        prompt_influence=1.0,
    )
    return response

def get_sfx_from_chatGPT(category, sub_category):
    response = client.text_to_sound_effects.get(
        category=category,
        sub_category=sub_category,
    )
    return response

def process_csv(input_csv, output_dir):
    # Ensure the output folder for output_csv exists
    output_folder_path = Path(output_dir)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Define the full path for output_csv file inside the specified folder
    output_csv = output_folder_path / "SFX_descriptions.csv"
    # Check if the output CSV already exists to determine if we need to write the header
    file_exists = output_csv.is_file()

    with open(input_csv, newline='', encoding='utf-8') as csvfile, \
         open(output_csv, 'a', newline='', encoding='utf-8') as output_file:
        
        reader = csv.DictReader(csvfile)
        fieldnames = ['description', 'category', 'sub_category', 'filename']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        
        # Write header only if the output file does not already exist
        if not file_exists:
            writer.writeheader()

        for row in reader:
            # Get fields from each row
            description = row['description']
            category = row['category']
            sub_category = row['sub_category']

            # Generate the file path and check if it already exists
            file_path = Path(output_dir) / category / sub_category / f"{description[:30].replace(' ', '_')}.wav"
            if file_path.is_file():
                print(f"File already exists, skipping generation: {file_path}")
                continue  # Skip to the next row if file exists

            # Generate audio based on the description only if the file does not exist
            audio_content = generate_audio(description)

            # Save audio to the specified file path
            save_audio(audio_content, file_path)

            # Write the details and file path to the output CSV
            writer.writerow({
                'description': description,
                'category': category,
                'sub_category': sub_category,
                'filename': str(file_path)
            })

# Modified save_audio function to accept file_path directly
def save_audio(audio_content, file_path):
    # Ensure the directory for file_path exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the audio content to the file
    with open(file_path, "wb") as audio_file:
        for chunk in audio_content:
            audio_file.write(chunk)
    
    print(f"Saved audio to: {file_path}")

def get_descritions(category, sub_category):
    client = OpenAI()
    chat = client.chat.completions.create(
            model="gpt-4o",
            modalities=["text"],
            messages=[
                {"role": "system", "content": f"""
                 You are a helpful assistant
                 """
                 },
                {
                    "role": "user",
                    "content":f"""I want you to generate me 20 sound description for the
                    category {category} and sub-category {sub_category}.
                    The output should be in the form an array like so:
                    
                    ["Cow mooing in a barn", "Chicken clucking in the coop", ...]"""
                }
            ]
        )

    try:
        response = chat.choices[0].message.content
        response = response.replace("“", "'").replace("”", "'")
        match = re.search(r'\[\s*(.*?)\s*]', response, re.DOTALL)
        if match:
            array_text = match.group(0)
            # Safely evaluate the array text as a Python literal
            array_literal = ast.literal_eval(array_text)
        else:
            print("No array found in the text.")
            array_literal = []
    except:
        print("No response found.")
        array_literal = []
    return array_literal


# Function to process the CSV and loop through each category/sub-category
def process_category_csv(file_path, output_csv):
    # make output_csv if it doesn't exist
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['description', 'category', 'sub_category'])
            writer.writeheader()

    with open(file_path, newline='', encoding='utf-8') as csvfile_input, \
         open(output_csv, 'a', newline='', encoding='utf-8') as csvfile_output:
        
        reader = csv.DictReader(csvfile_input)
        fieldnames = ['description', 'category', 'sub_category']
        writer = csv.DictWriter(csvfile_output, fieldnames=fieldnames)

        # Loop through each row in the CSV
        for row in reader:
            # Extract category, sub-category, and description from each row
            category = row['category']
            sub_category = row['sub_ategory']
            description = row['description']
            
            # Call the process_sound function with these values
            descriptions = get_descritions(category, sub_category)

            # insert descriptions into the output_dir
            for description in descriptions:
                # Write the category, sub-category, and description to the output CSV
                writer.writerow({
                    'category': category,
                    'sub_category': sub_category,
                    'description': description
                })

def main():
    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Generate audio for descriptions and save by category/sub-category.")
    parser.add_argument("--input_csv", help="Path to the input CSV file")
    parser.add_argument("--output_dir", help="Directory to save generated audio files")
    parser.add_argument("--output_csv", help="Path to the output CSV file")
    parser.add_argument("--category_csv", help="Category to generate sound effects for")

    args = parser.parse_args()

    # Process the CSV file and generate/save audio
    if not args.output_csv:
        process_csv(args.input_csv, args.output_dir)
    else:
        process_category_csv(args.category_csv, args.output_csv)

if __name__ == "__main__":
    main()