# Import necessary tools
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import PyPDF2
import docx
import io
import os
import pytesseract
from PIL import Image
from typing import Optional, List
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from google.generativeai.protos import Tool
import requests
import json

# ==============================================================================
# --- IMPORTANT: CONFIGURE YOUR KEYS ---
# Paste your new, secret Google AI API Key here.
API_KEY = "Insert your api key here"

# Paste your Search Engine ID from Step 2 here.
SEARCH_ENGINE_ID = "insert your search engine id here"
# ==============================================================================

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Google AI: {e}")

client = chromadb.PersistentClient(path="chroma_db")
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)
collection = client.get_or_create_collection(name="vit_study_assistant", embedding_function=google_ef)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def extract_text_from_file(file: UploadFile):
    content_type = file.content_type
    text = ""
    try:
        if content_type == 'application/pdf':
            stream = io.BytesIO(await file.read())
            reader = PyPDF2.PdfReader(stream)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            stream = io.BytesIO(await file.read())
            document = docx.Document(stream)
            text = "\n".join(para.text for para in document.paragraphs)
        elif content_type.startswith("image/"):
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image).strip()
    except Exception as e:
        print(f"Error extracting text from {file.filename}: {e}")
        return None
    return text

# --- API Endpoints ---
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    full_text = await extract_text_from_file(uploaded_file)
    if full_text:
        chunks = [chunk for chunk in full_text.split('\n\n') if len(chunk.strip()) > 50]
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not find any text chunks in the document.")
        collection.add(
            documents=chunks,
            metadatas=[{"source": uploaded_file.filename} for _ in chunks],
            ids=[f"{uploaded_file.filename}_{i}" for i in range(len(chunks))]
        )
        message = f"Successfully learned and saved {len(chunks)} chunks from: {uploaded_file.filename}"
    else:
        message = f"Could not extract text from {uploaded_file.filename}."
    all_items = collection.get()
    learned_files = list(set(meta['source'] for meta in all_items['metadatas'])) if all_items['metadatas'] else []
    return {"message": message, "learned_files": learned_files}

@app.post("/get-answer/")
async def get_answer(question_text: Optional[str] = Form(None), question_file: Optional[UploadFile] = File(None)):
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="The knowledge base is empty. Please upload documents first.")
    question = ""
    if question_text:
        question = question_text.strip()
    elif question_file:
        question = await extract_text_from_file(question_file)
    if not question:
        raise HTTPException(status_code=400, detail="Could not get a question from the provided input.")
    results = collection.query(query_texts=[question], n_results=3)
    retrieved_chunks = results['documents'][0]
    if not retrieved_chunks:
        return {"question": question, "message": "I couldn't find any relevant information in the documents you provided.", "ask_web_search": True}
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a helpful study assistant. Your goal is to answer the user's question based ONLY on the context provided below.
    Do not use any other information. If the context is not sufficient to answer the question, just say "The provided documents do not contain enough information to answer this question."
    CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"""
    try:
        generative_model = genai.GenerativeModel('gemini-1.5-flash')
        response = generative_model.generate_content(prompt)
        if "do not contain enough information" in response.text:
            return {"question": question, "message": response.text, "ask_web_search": True}
        else:
            return {"question": question, "answer": response.text, "source": "Your Uploaded Documents"}
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer from the AI model.")

# ==============================================================================
# --- NEW WEB SEARCH FUNCTION ---
# This function now uses the Custom Search API
# ==============================================================================
@app.post("/web-search/")
async def web_search(query: str = Form(...)):
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise HTTPException(status_code=400, detail="API key is not configured.")
    if not SEARCH_ENGINE_ID or SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE":
        raise HTTPException(status_code=400, detail="Search Engine ID is not configured.")
    
    try:
        # Step 1: Call the Google Custom Search API
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': API_KEY,
            'cx': SEARCH_ENGINE_ID,
            'q': query,
            'num': 5 # Get the top 5 results
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        search_results = response.json()

        # Step 2: Extract snippets from the results to create context
        snippets = [item.get('snippet', '') for item in search_results.get('items', [])]
        context = "\n".join(snippets).strip()

        if not context:
            return {"answer": "I couldn't find any relevant information on the web for that query.", "source": "Web Search"}

        # Step 3: Call the Gemini API with the search results as context
        prompt = f"""
        Based ONLY on the following web search results, provide a concise answer to the user's question.
        
        SEARCH RESULTS:
        {context}

        QUESTION:
        {query}

        CONCISE ANSWER:
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_response = model.generate_content(prompt)
        
        return {
            "answer": gemini_response.text,
            "source": "Web Search"
        }
    except Exception as e:
        print(f"Error during web search: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while searching the web.")