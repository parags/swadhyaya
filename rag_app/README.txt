Folders:

1. rag_app: This folder has all the objects required to run the rag application 
2. chroma_db: This folder has all the knowledge sheets chunked and contains embeddings and text chunks.     
    This will be used by the fast api server endpoint

Scripts:

1. fa_server.py - This script runs the api endpoint. This endpoint will be called by the front end
2. emotion_search_llm_sb.py - This is the frontend script that calls the api endpoint

Required packages are in the requirements.txt file. This will be used in pip install

Deployment instructions:
————————————————

1. Download the rag_app folder to your local machine
2. Create a python virtual environment 
3. From inside the python virtual environment, 
         pip install -r requirements.txt