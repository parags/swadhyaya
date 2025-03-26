import os
import json
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
import textwrap

# Define folders
INPUT_FOLDER = "/Users/palaniappanviswanathan/Desktop/AOL/rag_app/data/input"
PROCESSED_FOLDER = "/Users/palaniappanviswanathan/Desktop/AOL/rag_app/data/processed"
ERROR_FOLDER = "/Users/palaniappanviswanathan/Desktop/AOL/rag_app/data/error"
PROCESSED_FILE = "/Users/palaniappanviswanathan/Desktop/AOL/rag_app/processed_files.json"

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="rag_chunks")

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & effective

# Load processed files history
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = []

def chunk_text(text, chunk_size=100):
    """Splits text into chunks of approximately `chunk_size` words."""
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_file(file_path, file_name):
    """Processes a file: chunks text, converts each chunk into embeddings, and stores in ChromaDB."""
    try:
        print(f"file_path: {file_path}, file_name: {file_name}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Chunk the text
        #chunks = chunk_text(content)

        #for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(content).tolist()

            # Store each chunk in ChromaDB
        collection.add(
            ids=[f"{file_name}"], 
            embeddings=[embedding], 
            documents=[content],
            metadatas=[{"filename": file_name}]
        )

        # Move file to processed folder
        shutil.move(file_path, os.path.join(PROCESSED_FOLDER, file_name))

        # Update processed file log
        processed_files.append(file_name)
        with open(PROCESSED_FILE, "w") as f:
            json.dump(processed_files, f, indent=4)

        print(f"Successfully processed: {file_name}")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        shutil.move(file_path, os.path.join(ERROR_FOLDER, file_name))

def query_chromadb(query_text, top_k=3):
    """Searches ChromaDB for the top-k closest matches to the input query."""
    query_embedding = embedding_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )

    print(f"\n🔍 Query: {query_text}")
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        print(f"📄 {metadata['filename']} (Chunk {metadata['chunk_index']}): {doc[:200]}...")

def main():
    """Main function to process files from the input folder."""
    if not os.path.exists(INPUT_FOLDER):
        print("Input folder does not exist.")
        return

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(ERROR_FOLDER, exist_ok=True)

    for file_name in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, file_name)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        # Skip already processed files
        if file_name in processed_files:
            print(f"Skipping already processed file: {file_name}")
            continue

        process_file(file_path, file_name)

if __name__ == "__main__":
    main()
    query_word = input("\nEnter a word to search in ChromaDB: ")
    query_chromadb(query_word, top_k=5)
