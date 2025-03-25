import streamlit as st

import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import chromadb

def read_from_file(filename: str) -> str:
    return open("data/" + filename, "r").read()

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_sheets")

# Initialize OpenAI API
client = OpenAI(api_key="sk-proj-4nLxJo0nLYr2vd6wtwhLT3BlbkFJIVFkUB9FuIjqdCW5j708")  # Replace with actual API key

# Emotion Mapping
emotion_mapping = {
    "blame": ["blame", "guilt", "fault", "accuse", "responsibility"],
    "sadness": ["sadness", "grief", "loss", "hurt", "sorrow", "unhappy", "pain", "cry"],
    "doubt": ["doubt", "uncertainty", "confusion", "hesitation", "skepticism", "distrust", "suspicion"],
    "boredom": ["dull", "monotonous", "uninterested", "tedious", "apathetic"],
    "fear": ["scared", "afraid", "panic", "threat", "danger", "terror", "phobia"],
    "anxiety": ["nervous", "stressed", "worried", "pressure", "overthinking", "concern"]
}

# Load & Extract Knowledge Sheets
def extract_knowledge_sheets(collection):
    return collection.values
# LLM Emotion Detection
def detect_emotion(chat_text):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Classify the dominant emotion in this chat: {chat_text}. Return only one word from: ['blame', 'sadness', 'doubt', 'boredom', 'fear', 'anxiety']."}],
        model="gpt-4o"
    )
    return response.choices[0].message.content.strip().lower()

# Extract Tags from Knowledge Sheets
def extract_tags_and_text(knowledge_sheets):
    tagged_sections = {}

    for key, section in knowledge_sheets.items():
        match = re.match(r"\{(.*?)\}", section)  # Extract words inside { }
        section_tags = match.group(1).split(",") if match else []
        clean_section = section[len(match.group(0)):].strip() if match else section

        matched_emotions = set()
        for tag in section_tags:
            tag = tag.strip().lower()
            for emotion, synonyms in emotion_mapping.items():
                if tag in synonyms:
                    matched_emotions.add(emotion)

        tagged_sections[key] = {"tags": list(matched_emotions), "text": clean_section}

    return tagged_sections

# OpenAI Embedding Function
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

# Emotion Filtering via Embeddings
def get_closest_emotion_sections(detected_emotion, tagged_sections):
    emotion_embedding = get_embedding(detected_emotion)  
    section_embeddings = {key: get_embedding(data) for key, data in tagged_sections.items()}
    
    similarity_scores = {
        key: cosine_similarity([emotion_embedding], [embedding])[0][0]
        for key, embedding in section_embeddings.items()
    }

    filtered_sections = [key for key, score in similarity_scores.items() if score > 0.7]  # Adjust threshold if needed
    return filtered_sections

# Select Top 3 Matches (Cosine Similarity)
def select_top_matches(tagged_sections, chat_text, relevant_sections, detected_emotion):
    if not relevant_sections:
        return None, []

    chat_embedding = get_embedding(chat_text)
    emotion_embedding = get_embedding(detected_emotion)  
    knowledge_embeddings = {key: get_embedding(tagged_sections[key]) for key in relevant_sections}

    similarity_scores = {}
    for key, embedding in knowledge_embeddings.items():
        cosine_sim = cosine_similarity([chat_embedding], [embedding])[0][0]
        emotion_sim = cosine_similarity([emotion_embedding], [embedding])[0][0]

        # Weighted scoring: prioritize chat relevance but consider emotion alignment
        combined_score = (0.7 * cosine_sim) + (0.3 * emotion_sim)
        similarity_scores[key] = combined_score

    sorted_matches = sorted(similarity_scores.keys(), key=lambda k: similarity_scores[k], reverse=True)
    top_matches = sorted_matches[:3]

    return top_matches, [tagged_sections[k] for k in top_matches]

# LLM Selection Validation & Adjustment
def validate_selection(detected_emotion, selected_knowledge, relevant_sections, tagged_sections):
    print("\n🔍 Calling LLM for selection validation...")

    try:
        response_validation = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    The detected emotion is '{detected_emotion}'.
                    The selected knowledge sheet sections are:

                    {json.dumps(selected_knowledge, indent=2)}

                    Your Task:
                    - Confirm if these sections explicitly focus on '{detected_emotion}'.
                    - If any do **not** fully address '{detected_emotion}', suggest up to 3 better sections from: {json.dumps([tagged_sections[k]['text'] for k in relevant_sections], indent=2)}.
                    - Respond in **strict JSON format**:
                      {{
                        "valid_sections": ["..."],
                        "suggested_replacements": ["..."]
                      }}
                    """
                }
            ],
            model="gpt-4o"
        )

        if not response_validation or not response_validation.choices:
            print("❌ Error: OpenAI response is empty!")
            return selected_knowledge, "Validation Failed: Empty Response"

        raw_response = response_validation.choices[0].message.content.strip()
        print("🔹 Raw LLM Response:", raw_response)  # Debugging print

        # Ensure response is JSON formatted
        if not raw_response.startswith("{"):
            print("❌ Error: Response is not a JSON object. Attempting to fix...")
            raw_response = raw_response[raw_response.find("{"):]

        try:
            llm_response = json.loads(raw_response)
        except json.JSONDecodeError:
            print("❌ Error: LLM did not return valid JSON. Response was:", raw_response)
            return selected_knowledge, "Validation Failed: Invalid JSON Format"

        final_sections = llm_response.get("valid_sections", selected_knowledge)
        replacements = llm_response.get("suggested_replacements", [])

        return final_sections, f"Validation Completed - Adjustments: {len(replacements)} replacements made"

    except Exception as e:
        print(f"❌ Unexpected error in validate_selection: {str(e)}")
        return selected_knowledge, "Validation Failed: Unexpected Error"

# Streamlit UI
st.title("Chat Emotion Analysis & Knowledge Selection")
uploaded_chat = st.file_uploader("Upload a chat text file", type=["txt"])
ks = {}
for i in range(1, 364):
    ks_str = read_from_file("ks" + str(i) + ".txt")
    ks[i] = ks_str
    collection.add(
        documents=[
            ks_str
        ],
        ids=[str(i)])

if uploaded_chat and collection:
    chat_text = uploaded_chat.getvalue().decode("utf-8")
    detected_emotion = detect_emotion(chat_text)
    # tagged_sections = extract_tags_and_text(ks)
    tagged_sections = ks

    relevant_sections = get_closest_emotion_sections(detected_emotion, tagged_sections)

    top_matches_keys, top_matches_texts = select_top_matches(tagged_sections, chat_text, relevant_sections, detected_emotion)

    final_selection, validation_status = validate_selection(detected_emotion, top_matches_texts, relevant_sections, tagged_sections)

    st.write(f"**Detected Emotion:** {detected_emotion.capitalize()}")
    for i, (key, text) in enumerate(zip(top_matches_keys, final_selection), 1):
        st.write(f"#### {i}. Section {key}:\n{text}")

    st.write(f"**Validation Status:** {validation_status}")