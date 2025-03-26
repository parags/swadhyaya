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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in your environment variables
print(f"GROQ_API_KEY: {GROQ_API_KEY}")

# Initialize the Groq client
client = Groq()

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
    print(f"results: {results}")
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        response.append({
            "filename": metadata["filename"],
            "text": doc
        })
    
    return response

# Function to get response from Groq's LLaMA model
def get_llama_response(word,userContext, context):
    """Fetch response from Groq's API using LLaMA models."""
    # system_prompt = (
    #     "You are a helpful emotional support companion. You will be provided with the user's emotion along with an explaination of how the user feels as userContext. You will also be provided with a set of documents as context. You job is to look at the emotion and userContext and pick the most appropriate single document from the set of documents and return that document verbatim. DO NOT add any extra word of your own."
    #     You MUST answer ONLY based on the context provided to you."
    #     "Please do not include any external information, and stay within the context provided. Using the Word and Context provided to you, summarize nicely like Guruji, very politely and compassionately, advising his disciple."
    # )
    system_prompt = (
        "You are a helpful emotional support companion. You will be provided with the user's emotion in the emotion field. You will also be provided with an explaination of how the user feels in the userContext field. You will also be provided with a set of documents under the context field. You job is to look at the emotion and userContext and pick the most appropriate single document from the set of documents and return that document verbatim. DO NOT add any extra word of your own."
    )
    prompt = f"{system_prompt}\n\nEmotion detected by health metrics as emotion: {word}\n \n Additional speech to text input from the user explaining how they actually feel as userContext: {userContext}\n\nContext: {context}\n\n"
    print(f"prompt: {prompt}")
    # Create the message payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Call the Groq API with the correct parameters
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.0,
        max_completion_tokens=1024,
        top_p=0.9,
        stream=True,
        stop=None,
    )

    # Handle the response (printing the stream in chunks)
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    
    return response_text

@app.get("/search")
def search(word: str = Query(..., description="Word to search for"), userContext: str = Query(..., description="Additional voice context from the user"), top_k: int = Query(3, description="Number of top results to return")):
    """FastAPI endpoint to search for relevant text in ChromaDB."""
    print(f"inside @app.get().search, word: {word}, top_k: {top_k}")
    word = word+userContext
    results = query_chromadb(word, top_k)
    print(f"results:{results}")
    context = " ".join([res["text"] for res in results])  # Merge chunks as context
    llama_response = get_llama_response(word, userContext, context)
    print(f"context: {context}")
    print(f"***llama_response***: {llama_response}")
    return {"query": word, "results": llama_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
