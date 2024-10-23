import os
import argparse

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import json, csv
from openai import OpenAI

OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def save_as_csv(data, filename):
    if data:
        keys = data[0].keys()
        with open(filename, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

def read_csv_as_dicts(file_path):
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            # Create a CSV reader object that reads the file as dictionaries
            csv_reader = csv.DictReader(csvfile)
            
            # Convert the csv_reader object to a list of dictionaries
            data = [row for row in csv_reader]
        
        return data

class RAGAutoComplete:
    def __init__(self, doc_path, chroma_path, api_key):
        self.doc_path = doc_path
        self.chroma_path = chroma_path
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.db_chroma = None
        self.overlap = 20
        self.texts = []
        self.load_or_create_db()
        self.window_size = 5
        self.window_mid = np.floor(self.window_size/2)
        self.window_location = 40 - 3

    def load_txt(self):
        """Reads the txt file and returns its content as a string."""
        with open(self.doc_path, 'r', encoding='utf-8') as file:
            return file.read()

    def index_txt(self):
        """Loads and splits the txt file into chunks, then embeds and indexes them."""
        # Embed and load into the database
        content = self.load_txt()

        # Split the document into smaller chunks
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.chunks = self.text_splitter.create_documents([content])

        for i, chunk in enumerate(self.chunks):
            # Add the page number and index to each chunk's metadata
            chunk.metadata = {
                "chunk_index": i  # Assign a unique chunk index
            }

    def get_audio_prompt_from_chatGPT(self, text):
        client = OpenAI()

        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are a helpful assistant. I want you \
             to help me generate an audio prompt based on the following text.\
             The prompt should be very short and should only contain the names of sounds.\
             It should only contain one sound maximum. \
             If there is no sound use silence, but this should be rarely used.\
             Examples include, knocking on a door, a cat meowing, cars/motorbikes etc.
             I  want you to also estimate the duration of the audio prompt. \
             It should be as short as possible, use the a json format as an output.\
             ```json
             {
                "prompt": ,
                "audio_duration": 1.0
             }
             ```
             """},
            {"role": "user", "content": text}
        ]
        )


        ret = {
                "prompt": "silence",
                "audio_duration": 1.0
             }
        try:
            json_string = completion.choices[0].message.content.strip("`json")
            ret = json.loads(json_string)
        except:
            print("Error in getting audio prompt from chatGPT")
        return ret



    def add_audio_data(self):
        """Add audio data to the database."""
        # Add audio data to the database
        generator = AudioGenerator()

        self.audio_db = []

        sound_dir = os.path.join(self.chroma_path, "sounds")
        os.makedirs(sound_dir, exist_ok=True)
    
        batch_size = 10  # Set your desired batch size

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            
            print(f"Running prompt collection for batch {i}/{len(self.chunks)}")
            kwargs = []
            for j, chunk in tqdm(enumerate(batch)):
                chunk.metadata = {
                    "chunk_index": i + j  # Assign a unique chunk index across batches
                }
                kwargs.append(self.get_audio_prompt_from_chatGPT(chunk.page_content))
            
            print("Finished collecting prompts.")

            # get the audo
            filepaths = [os.path.join(sound_dir, f"audio_{i + j}.wav") for j in range(len(kwargs))]
            generator.generate_audio(output_file=filepaths,
                                     prompt=[k["prompt"] for k in kwargs],
                                     audio_duration=3.0) # hard code for now
            
            for j in range(len(kwargs)):
                self.audio_db.append({
                    "chunk": batch[j].page_content,
                    "chunk_index": i + j,
                    "prompt": kwargs[j]["prompt"],
                    "audio": filepaths[j]
                })

            save_as_csv(self.audio_db, os.path.join(self.chroma_path, f"{self.chroma_path}.csv"))

    def load_or_create_db(self):
        """Checks if the database exists and loads it, otherwise creates it."""
        self.index_txt()
        if os.path.exists(self.chroma_path):
            print(f"Loading existing database for {self.chroma_path}...")
            self.db_chroma = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
            self.audio_db = read_csv_as_dicts(os.path.join(self.chroma_path, f"{self.chroma_path}.csv"))
        else:
            print(f"Database for {self.chroma_path} not found. Creating a new one...")
            from saudio import AudioGenerator
            self.db_chroma = Chroma.from_documents(self.chunks, self.embeddings, persist_directory=self.chroma_path)
            self.db_chroma.persist()
            self.add_audio_data()

    def retrieve_next_chunks(self, input_sentence, n=5):
        """Find the input sentence and return the next two sentences."""
        # Perform a similarity search to find the chunk containing the input sentence
        search_results = self.db_chroma.similarity_search_with_score(input_sentence, k=1)
    
        ret_dict = {
            "chunks": [],
            "chunks_index": [],
            "chunks_scores": None
        }

        # here we want to add logic to see if the match is inline with where we expect it to be

        if search_results:
            chunk, score = search_results[0]  # Get the most relevant chunk
            index = chunk.metadata['chunk_index']

            # if index is beyond window then ignore
            if index > self.window_size + self.window_location:
                print(f"Index is {index} for sentence {self.chunks[index]} beyond window. Ignoring.")
                return ret_dict

            if index < self.window_location + self.window_mid:
                print(f"Index is {index} for sentence {self.chunks[index]} before window. Ignoring.")
                return ret_dict

            if index == self.window_location + self.window_mid:
                # play next chunk
                print("Index matches currnt position, playing next chunk.")
                index += 1

            self.window_location = int(index - self.window_mid)
            print(f"New window_location: {self.window_location} for sentence {self.chunks[self.window_location]}")

            if len(self.chunks) > index + 1:
                ret_dict["chunks"].append(self.chunks[index + 1].page_content)
                ret_dict["chunks_index"].append(index + 1)
                ret_dict["chunks_scores"] = score
        else:
            print("No search results found.")
        return ret_dict
        
    def get_audio(self, text):
        chunks = self.text_splitter.create_documents([text])
        if len(chunks) == 0:
            print("No chunks found. Please enter a longer sentence.")
            return None
        else:
            # Retrieve 
            data = []
            for c in chunks:
                chunk_data = self.retrieve_next_chunks(c.page_content, 5)
                # print(chunk_data)
                chunk_data["audio_db"] = [self.audio_db[i] for i in chunk_data["chunks_index"]]
                data.append(chunk_data)
            return data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='RAG System for Autocompleting Sentences.')
    parser.add_argument('--doc_path', type=str, help='Path to the text document')
    args = parser.parse_args()

    # Extract book name from file path and use it as the Chroma DB name
    book_name = os.path.splitext(os.path.basename(args.doc_path))[0]
    chroma_path = f"{book_name}_db"

    # Initialize the autocomplete system
    rag_system = RAGAutoComplete(doc_path=args.doc_path, chroma_path=chroma_path, api_key=OPENAI_API_KEY)
    
    # # # # Load or create the database
    
    input_sentence = "The Boy Who Lived. Mr. and Mrs. Dursley of No. 4 Privet Drive were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold such nonsense. Mr. Dursley was the director of a film called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large moustache. Mrs. Dursley was thin and blonde, and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbours. The Dursleys had a small son called Dudley, and in their opinion, there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years. In fact, Mrs. Dursley pretended she didn't have a sister at all, because her sister and her good-for-nothing husband were so undursleyish as it was possible to be. The Dursleys shuddered to think what their neighbours would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son too, but they never ever had seen him. This boy was another good reason for keeping the Potters away. They didn't want Dudley mixing with a child like that. When Mr. Dursley woke up on the dull, grey Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening."
    print(rag_system.get_audio(input_sentence))

if __name__ == "__main__":
    main()