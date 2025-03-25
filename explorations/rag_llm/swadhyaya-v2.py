import streamlit as st
import openai
import docx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Load OpenAI API Key (Secure it properly)
client = OpenAI(
    api_key="sk-proj-4nLxJo0nLYr2vd6wtwhLT3BlbkFJIVFkUB9FuIjqdCW5j708"
)

# Function to extract text from multiple Word documents
def extract_knowledge_sheets(docx_files):
    sections = {}
    section_counter = 1  # Numbering sections
    
    for docx_file in docx_files:
        doc = docx.Document(docx_file)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])  # Read all text
        
        # Split into sections arbitrarily if no headings exist
        raw_sections = full_text.split("\n\n")  # Using double newline as delimiter
        
        for section in raw_sections:
            section_key = str(section_counter)  # Numbered sections as keys
            sections[section_key] = section.strip()
            section_counter += 1
    
    return sections

# Function to generate text embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

# Streamlit App UI
st.title("Chat Emotion Analysis & Knowledge Selection")
st.write("Upload a chat text file and knowledge sheets to find the most relevant knowledge.")

# File uploader for chat text
uploaded_chat = st.file_uploader("Upload a chat text file", type=["txt"])



# Ensure processing starts only when both chat and knowledge sheets are uploaded
if uploaded_chat and uploaded_knowledge_files:
    chat_text = uploaded_chat.getvalue().decode("utf-8")
    st.text_area("Chat Content", chat_text, height=200)

    # Extract knowledge sheets
    knowledge_sheets = extract_knowledge_sheets(uploaded_knowledge_files)

    # Stop if no knowledge sheets were extracted
    if not knowledge_sheets:
        st.error("No knowledge sheets found in the uploaded files. Please check the content.")
        st.stop()

    # First LLM Call: Detect Dominant Emotion (Restricted to predefined emotions)
    valid_emotions = ['blame', 'sadness', 'doubt', 'boredom', 'fear', 'anxiety']
    detected_emotion = ""

    while detected_emotion not in valid_emotions:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this chat and return only one word for the dominant emotion. Choose only from the following list: {', '.join(valid_emotions)}. Do not return any new words: {chat_text}"
                }
            ],
            model="gpt-4o"
        )
        detected_emotion = response.choices[0].message.content.strip().lower()

    st.write(f"**Detected Emotion:** {detected_emotion.capitalize()}")

    # Generate embedding for the full chat text
    chat_embedding = get_embedding(chat_text)

    # Generate embeddings for all knowledge sheets
    knowledge_embeddings = {key: get_embedding(value) for key, value in knowledge_sheets.items()}

    # Compute cosine similarity between chat embedding and each knowledge sheet
    best_match_key = None
    highest_similarity = -1

    for key, embedding in knowledge_embeddings.items():
        similarity = cosine_similarity([chat_embedding], [embedding])[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_key = key

    selected_knowledge = knowledge_sheets.get(best_match_key, "No specific match found.")

    # **Second LLM Call: Validate and Confirm Selection, ensuring it picks the most relevant knowledge sheet for detected emotion**
    response_knowledge = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                The detected emotion is '{detected_emotion}'.
                
                Below are all available knowledge sheets:

                {knowledge_sheets}

                Your task:
                - Select the **most relevant** knowledge sheet that addresses '{detected_emotion}' directly.
                - If no section explicitly mentions this emotion, select the closest match.
                - Only return the **section number** that best corresponds to '{detected_emotion}', nothing else.
                - Do not choose a generic or unrelated section.
                - If multiple sections mention the detected emotion, return the most detailed one.
                """
            }
        ],
        model="gpt-4o"
    )

    final_match_key = response_knowledge.choices[0].message.content.strip()

    # Ensure the final selection is a valid section number; fallback to highest similarity match
    if final_match_key.isdigit() and final_match_key in knowledge_sheets:
        final_selected_knowledge = knowledge_sheets[final_match_key]
    else:
        final_selected_knowledge = selected_knowledge
        final_match_key = best_match_key

    # Display only the final selected knowledge sheet
    st.write(f"**Final Selected Knowledge Sheet (Section {final_match_key} - {detected_emotion.capitalize()}):**\n\n{final_selected_knowledge}")