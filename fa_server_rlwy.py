import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
import requests
import json
import os

# Initialize FastAPI app
app = FastAPI()

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="rag_chunks")

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in your environment variables
print(f"GROQ_API_KEY: {GROQ_API_KEY}")

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

# Function to get response from Groq's LLaMA model
def get_llama_response(word, context):
    """Fetch response from Groq's API using LLaMA models."""
    system_prompt = (
        "You are a helpful emotional support companion. You MUST answer ONLY based on the context provided to you."
        "Please do not include any external information, and stay within the context provided."
    )
    
    prompt = f"{system_prompt}\n\nWord: {word}\nContext: {context}\n\nPlease provide a meaningful response."

    payload = {
        "model": "llama3-8b",  # Groq's available LLaMA model (adjust based on needs)
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500  # Limit response length
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_API_URL, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Failed to get response from Groq API."

@app.get("/search")
def search(word: str = Query(..., description="Word to search for"), top_k: int = Query(3, description="Number of top results to return")):
    """FastAPI endpoint to search for relevant text in ChromaDB."""
    results = query_chromadb(word, top_k)
    context = " ".join([res["text"] for res in results])  # Merge chunks as context
    llama_response = get_llama_response(word, context)
    
    return {"query": word, "results": llama_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
