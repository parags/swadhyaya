import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="rag_chunks")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure this is set in your environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Set it in your environment variables.")

# Initialize the Groq client
client = Groq()

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & effective

def query_chromadb(query_text: str, top_k: int = 6):
    """Searches ChromaDB for the top-k closest matches to the input query."""
    query_embedding = embedding_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )

    response = []
    print(f"ChromaDB Query Results: {results}")
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        response.append({
            "filename": metadata["filename"],
            "text": doc
        })
    
    return response

# Function to get response from Groq's LLaMA model
def get_llama_response(word, userContext, context):
    """Fetch response from Groq's API using LLaMA models."""
    system_prompt = (
    "You are an assistant tasked with selecting and returning exactly one document from a provided list.\n\n"
    "**Instructions:**\n"
    "1. You will receive multiple documents labeled as 'Document 1', 'Document 2', etc.\n"
    "2. Your task is to select the ONE document that best aligns with the user's emotion and context.\n"
    "3. **You MUST return the complete text of the chosen document EXACTLY as provided.**\n"
    "4. **Do NOT return a document number. Do NOT summarize. Do NOT modify the document in any way.**\n\n"
    "---\n"
    "**User Query:**\n"
    f"Emotion detected: {word}\n"
    f"User explanation: {userContext}\n\n"
    "**Documents to choose from:**\n"
    f"{context}\n\n"
    "**Reply with the FULL text of the selected document. Do NOT add any explanations or labels.**"
    )    

    print(f"LLM Prompt: {system_prompt}")

    # Create the message payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Emotion: {word}\nUser Context: {userContext}\nAvailable Documents: {context}"}
    ]

    # Call the Groq API with the correct parameters
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.0,
        max_completion_tokens=2048,
        top_p=1.0,
        stream=True,
        stop=None,
    )

    # Handle the response (streaming output)
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    
    return response_text.strip()

@app.get("/search")
def search(
    word: str = Query(..., description="Word to search for"),
    userContext: str = Query(..., description="Additional voice context from the user"),
    top_k: int = Query(6, description="Number of top results to return")  # Defaulting to 6
):
    """FastAPI endpoint to search for relevant text in ChromaDB."""
    print(f"Inside /search - Emotion: {word}, User Context: {userContext}, Top K: {top_k}")

    # Concatenating word and user context for a better query
    query_text = word + " " + userContext
    results = query_chromadb(query_text, top_k)

    if not results:
        return {"error": "No relevant documents found in ChromaDB."}

    print(f"Retrieved {len(results)} results from ChromaDB.")

    # Extract the document texts
    #context = " ".join([res["text"] for res in results])  
    context = "\n\n".join([f"Document {i+1}: {res['text']}" for i, res in enumerate(results)])
    llama_response = get_llama_response(word, userContext, context)

    print(f"Selected Document by LLM: {llama_response}")

    return {
        "query": query_text,
        "results": llama_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
