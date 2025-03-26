import streamlit as st
import requests
import json

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/search"

# Ollama API for the LLaMA 3.2:1b model
OLLAMA_API_URL = "http://localhost:11434/v1/completions"

# Function to query FastAPI
def get_search_results(word, top_k):
    response = requests.get(FASTAPI_URL, params={"word": word, "top_k": top_k})
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to get search results from FastAPI.")
        return None

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
        # "model": "llama3.2:1b",  # Model name
        "model": "llama3.2",  # Model name
        "prompt": prompt,
        "max_tokens": 500  # Control response length
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(OLLAMA_API_URL, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    else:
        st.error("Failed to get response from LLaMA model." + str(response.text))
        return None

# Streamlit UI
st.title("RAG Search and LLaMA Insight Generator")

# Input fields
word = st.text_input("Enter word to search:")
top_k = st.number_input("Top K results:", min_value=1, max_value=10, value=3)

# Button to trigger the search and processing
if st.button("Search and Generate Insight"):
    if word:
        # Fetch search results from FastAPI
        search_results = get_search_results(word, top_k)

        if search_results:
            # Show the search results in the right sidebar
            with st.sidebar:
                st.write("Search Results:")
                for idx, result in enumerate(search_results['results'], start=1):
                    st.write(f"{idx}. {result['text']}")

            # Process the results with LLaMA
            all_text = "\n".join([result['text'] for result in search_results['results']])
            llama_response = get_llama_response(word, all_text)

            if llama_response:
                # Show the LLaMA model's response below the search box
                st.subheader("LLaMA Insight:")
                st.write(llama_response)

    else:
        st.error("Please enter a word to search.")
