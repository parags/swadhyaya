import streamlit as st
import openai
import os
from openai import OpenAI
import chromadb

def read_from_file(filename: str) -> str:
    return open("data/" + filename, "r").read()

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="knowledge_sheets")

count = 1
while count < 365:
    collection.add(
        documents=[
            read_from_file("ks" + str(count) + ".txt")
        ],
        ids=[str(count)])
    count = count + 1

# Load OpenAI API Key (Secure it properly)
client = OpenAI(
    api_key="sk-proj-4nLxJo0nLYr2vd6wtwhLT3BlbkFJIVFkUB9FuIjqdCW5j708"
)

# Streamlit App UI
st.title("Chat Emotion Analysis & Knowledge Selection")
st.write("Upload a chat text file to analyze dominant emotion and receive relevant knowledge.")

# File uploader
uploaded_file = st.file_uploader("Upload a chat text file", type=["txt"])

if uploaded_file & collection:
    chat_text = uploaded_file.getvalue().decode("utf-8")
    st.text_area("Chat Content", chat_text, height=200)

    # OpenAI API Call for Emotion Detection
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Analyze this chat and return only one word for the dominant emotion (e.g., 'anger', 'sadness', 'doubt', 'boredom', 'fear', 'anxiety'): {chat_text}"
            }
        ],
        model="gpt-4o"
    )

    # Extracting the dominant emotion
    detected_emotion = response.choices[0].message.content.strip().lower()
    st.write(f"**Detected Emotion:** {detected_emotion.capitalize()}")

    results = collection.query(
        query_texts=["Given a conversation text, retrieve relevant passages that closely apply to the conversation delimited by ```.\n" +
                "```" +chat_text + "```"], # Chroma will embed this for you
        n_results=3 # how many results to return
    )
    # Mapping detected emotion to a relevant knowledge sheet
    knowledge_sheets = results['documents']

    # selected_knowledge = knowledge_sheets.get(detected_emotion, "No specific match found")
    st.write(f"**Selected Knowledge Sheet:**\n\n{knowledge_sheets}")