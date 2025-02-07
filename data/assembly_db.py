import argparse
import re
import pandas as pd
import os
from openai import OpenAI
import chromadb
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def format_to_sentence(text):
    # Replace hyphens and underscores with spaces
    sentence = text.replace("-", " ").replace("_", " ")
    # Remove extra spaces and punctuation from multiple hyphens or underscores
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence.capitalize()  # Capitalize the first letter

class SoundAssigner:
    def __init__(self, chroma_name, data_root="./"):
        self.data_root = data_root
        self.chroma_name = chroma_name
        self.client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.data_root, self.chroma_name))
        self.collection = self.chroma_client.get_or_create_collection("category_db")
        self.transcript = ""

    def get_embedding(self, word):
        """Get the embedding for a given text using the new OpenAI API."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[word]
        )
        return response.data[0].embedding

    def load_csv_to_db_ESC(self, csv_path):
        """Load a CSV, create embeddings for 'category' column, and store in Chroma DB."""
        print("Creating Chroma database...")
        data = pd.read_csv(csv_path)
        if 'category' not in data.columns or 'src_file' not in data.columns:
            raise ValueError("CSV must contain 'category' and 'src_file' columns")
        
        data = data.drop_duplicates(subset=['category']).reset_index(drop=True)

        # add path to name
        data['filename'] = data['filename'].apply(lambda x: os.path.join("ESC-50-master/audio", x))
        self.load_to_db(data)

    def load_csv_to_db(self, csv_path):
        """Load a CSV, create embeddings for 'category' column, and store in Chroma DB."""
        print("Creating Chroma database...")
        data = pd.read_csv(csv_path)
        self.load_to_db(data)        

    def load_SA_to_db(self, path):
        """Loads the files in path and gets the description from their name"""
        print("Creating Chroma database...")
        data = pd.DataFrame(columns=["category", "filename"])
        for root, dirs, files in os.walk(path):
            for file in files:
                name = format_to_sentence(file.split(".")[0])
                data = pd.concat([data, pd.DataFrame([{"category": name, "filename": os.path.join(root, file)}])], ignore_index=True)
        self.load_to_db(data)

    def load_to_db(self, data):
        
        # Generate embeddings for each category word and store them with src_file metadata
        for _, row in tqdm(data.iterrows()):
            word = row['description']
            src_file = row['filename']
            
            # Get embedding for the word in the category column
            embedding_vector = self.get_embedding(word)
            
            # Add to Chroma database
            self.collection.add(
                embeddings=[embedding_vector], 
                metadatas=[{"filename": src_file}], 
                ids=[word]
            )
        
        print("Chroma database created successfully.")

    def retrieve_src_file(self, word):
        """Retrieve the src_file associated with the closest match to the given word."""
        # Get embedding for the input word
        # TODO need to add something about the context
        embedding_vector = self.get_embedding(word)
        
        # Query Chroma database for the closest match
        results = self.collection.query(embedding_vector)
        if results['metadatas']:
            return os.path.join(self.data_root, results['metadatas'][0][0]['filename']), results['ids'][0][0], results['distances'][0][0]
        else:
            return None

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sound Assigner")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file for ESC-50")
    parser.add_argument("--sa_path", type=str, help="Path to the Sound Assigner files")
    parser.add_argument("--chroma_name", type=str, required=True, help="Path to store Chroma DB")
    args = parser.parse_args()

    assigner = SoundAssigner(chroma_name=args.chroma_name)
    # check if chroma exisits
    # if not os.path.exists(args.chroma_name):
    if args.csv_path is not None:
        assigner.load_csv_to_db(args.csv_path)
    if args.sa_path is not None:
        assigner.load_SA_to_db(args.sa_path)
    # assigner.load_csv_to_db_ESC(args.csv_path)
    # Example word retrieval
    test_word = "cow mooing"
    filename, ids, score = assigner.retrieve_src_file(test_word)
    print(f"Closest match for '{test_word}': {ids} with score {score}")