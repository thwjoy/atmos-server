import argparse
# import re
import pandas as pd
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
import chromadb

# Set your OpenAI API key
OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class SoundAssigner:
    def __init__(self, chroma_path):
        self.chroma_path = chroma_path
        self.client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("category_db")

    def get_embedding(self, text):
        """Get the embedding for a given text using the new OpenAI API."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def load_csv_and_create_chroma_db(self, csv_path):
        """Load a CSV, create embeddings for 'category' column, and store in Chroma DB."""
        print("Creating Chroma database...")
        data = pd.read_csv(csv_path)
        if 'category' not in data.columns or 'src_file' not in data.columns:
            raise ValueError("CSV must contain 'category' and 'src_file' columns")
        
        data = data.drop_duplicates(subset=['category']).reset_index(drop=True)
        
        # Generate embeddings for each category word and store them with src_file metadata
        for _, row in data.iterrows():
            word = row['category']
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
        embedding_vector = self.get_embedding(word)
        
        # Query Chroma database for the closest match
        results = self.collection.query(embedding_vector)
        if results['metadatas']:
            return results['metadatas'][0][0]['filename'], results['ids'][0][0]
        else:
            return None

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sound Assigner")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--chroma_path", type=str, required=True, help="Path to store Chroma DB")
    args = parser.parse_args()

    assigner = SoundAssigner(chroma_path=args.chroma_path)
    # check if chroma exisits
    # if not os.path.exists(args.chroma_path):
    # assigner.load_csv_and_create_chroma_db(args.csv_path)
    # Example word retrieval
    test_word = "As I went walking through the rainy forest."
    src_file = assigner.retrieve_src_file(test_word)
    print(f"Source file for '{test_word}': {src_file}")