import chromadb
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, Query
import requests
import json



# Initialize FastAPI app
app = FastAPI()

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="rag_chunks")

# Ollama API for the LLaMA 3.2:1b model
OLLAMA_API_URL = "http://localhost:11434/v1/completions"

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & effective

def query_chromadb(query_text: str, top_k: int = 3):
    """Searches ChromaDB for the top-k closest matches to the input query."""
    query_embedding = embedding_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )

    response = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        response.append({
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"],
            "text": doc
        })
    
    return response


# Function to get response from Ollama LLaMA model
def get_llama_response(word, text):
    # system_prompt = (
    #     "You are a helpful emotional support companion. You MUST answer ONLY based on the context provided to you."
    #     " In your response, start by saying 'Guruji says:'. Do not use any other words. Please do not include any external information, and stay within the context provided."
    # )
    system_prompt = (
        "You are a helpful emotional support companion. You MUST answer ONLY based on the context provided to you."
        "Please do not include any external information, and stay within the context provided."
    )
    
    prompt = f"{system_prompt}\n\nWord: {word}\nText: {text}\n\nPlease process this information and provide insights."

    payload = {
        "model": "llama3.2:1b",  # Model name
        "prompt": prompt,
        "max_tokens": 500  # Control response length
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(OLLAMA_API_URL, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    else:
        return "Failed to get response from LLaMA model."

@app.get("/search")
def search(word: str = Query(..., description="Word to search for"), top_k: int = Query(3, description="Number of top results to return")):
    """FastAPI endpoint to search for relevant text in ChromaDB."""
    results = query_chromadb(word, top_k)
    llama_response = get_llama_response(word, results)
    print(f"results:{results}") 
    return {"query": word, "results": llama_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
