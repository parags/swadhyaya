import chromadb
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, Query



# Initialize FastAPI app
app = FastAPI()

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="rag_chunks")

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

@app.get("/search")
def search(word: str = Query(..., description="Word to search for"), top_k: int = Query(3, description="Number of top results to return")):
    """FastAPI endpoint to search for relevant text in ChromaDB."""
    results = query_chromadb(word, top_k)
    return {"query": word, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
