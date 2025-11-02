# Import necessary tools
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import core Python libraries for file processing and RAG
import PyPDF2
import docx
import io
import os
import pytesseract
from PIL import Image
from typing import Optional, List
import chromadb
from chromadb.utils import embedding_functions

# NEW: Use InferenceClient from huggingface_hub
from huggingface_hub import InferenceClient

# ==============================================================================
# HUGGING FACE CONFIGURATION
# ⚠️ REPLACE THIS WITH YOUR ACTUAL TOKEN FROM HUGGING FACE
HF_TOKEN = "PASTE URL HERE"  # <-- PASTE YOUR FULL TOKEN HERE

# Model to use
TEXT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Free on Inference Providers
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

print("=" * 70)
if not HF_TOKEN or HF_TOKEN == "PASTE_YOUR_TOKEN_HERE" or not HF_TOKEN.startswith("hf_"):
    print("❌ ERROR: Hugging Face token NOT configured!")
    print("Please edit line 23 in main.py and paste your token:")
    print('   HF_TOKEN = "hf_your_actual_token_here"')
    print(f"Current value: {HF_TOKEN}")
else:
    print(f"✅ Token configured: {HF_TOKEN[:15]}...")
print(f"✅ Using model: {TEXT_MODEL_ID}")
print("=" * 70)

# Initialize Inference Client
try:
    hf_client = InferenceClient(token=HF_TOKEN)
    print("✅ Hugging Face Inference Client initialized")
except Exception as e:
    print(f"❌ Error initializing client: {e}")
    hf_client = None

# ==============================================================================

# --- CHROMA DB & EMBEDDING SETUP ---
try:
    os.makedirs("chroma_db", exist_ok=True)
    client = chromadb.PersistentClient(path="chroma_db")
    
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID
    )
    
    collection = client.get_or_create_collection(
        name="vit_study_assistant", 
        embedding_function=hf_ef
    )
    print(f"✅ ChromaDB initialized. Documents loaded: {collection.count()}")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    raise

# ==============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- FILE EXTRACTION FUNCTION ---
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
        else:
            print(f"Unsupported content type: {content_type}")
            return None
    except Exception as e:
        print(f"Error extracting text from {file.filename}: {e}")
        return None
    return text.strip() if text else None

# --- API Endpoints ---
@app.get("/")
async def read_index():
    if not os.path.exists('static/index.html'):
        return {
            "message": "VIT Study Assistant API", 
            "status": "running",
            "endpoints": {
                "/health": "Check system status",
                "/upload-file/": "Upload documents (POST)",
                "/get-answer/": "Ask questions (POST)",
                "/clear-database/": "Clear all documents (DELETE)"
            }
        }
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    token_status = "not_configured"
    if HF_TOKEN and HF_TOKEN != "PASTE_YOUR_TOKEN_HERE" and HF_TOKEN.startswith("hf_"):
        token_status = "✅ configured"
        if hf_client:
            token_status = "✅ valid and ready"
    
    return {
        "status": "healthy",
        "documents_loaded": collection.count(),
        "hf_token_status": token_status,
        "model": TEXT_MODEL_ID,
        "inference_method": "InferenceClient with Providers"
    }

@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    try:
        print(f"Processing file: {uploaded_file.filename}")
        full_text = await extract_text_from_file(uploaded_file)
        
        if not full_text:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not extract text from {uploaded_file.filename}. Check file format."
            )
        
        # Split into chunks
        chunks = [chunk.strip() for chunk in full_text.split('\n\n') if len(chunk.strip()) > 50]
        
        if not chunks:
            chunks = [chunk.strip() for chunk in full_text.split('\n') if len(chunk.strip()) > 50]
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="Could not find any meaningful text chunks in the document."
            )
        
        # Add to collection
        collection.add(
            documents=chunks,
            metadatas=[{"source": uploaded_file.filename} for _ in chunks],
            ids=[f"{uploaded_file.filename}_{i}" for i in range(len(chunks))]
        )
        
        message = f"Successfully learned {len(chunks)} chunks from: {uploaded_file.filename}"
        print(f"✅ {message}")
        
        # Get all learned files
        all_items = collection.get()
        learned_files = list(set(meta['source'] for meta in all_items['metadatas'])) if all_items['metadatas'] else []
        
        return {
            "success": True,
            "message": message, 
            "learned_files": learned_files,
            "chunks_added": len(chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# --- RAG ANSWER GENERATION ---
@app.post("/get-answer/")
async def get_answer(
    question_text: Optional[str] = Form(None), 
    question_file: Optional[UploadFile] = File(None)
):
    try:
        # Check token and client
        if not HF_TOKEN or HF_TOKEN == "PASTE_YOUR_TOKEN_HERE" or not HF_TOKEN.startswith("hf_"):
            raise HTTPException(
                status_code=500,
                detail="❌ Hugging Face token not configured. Please edit main.py line 23."
            )
        
        if not hf_client:
            raise HTTPException(
                status_code=500,
                detail="❌ Inference client not initialized. Check your token."
            )
        
        # Check if knowledge base has documents
        if collection.count() == 0:
            raise HTTPException(
                status_code=400, 
                detail="The knowledge base is empty. Please upload documents first."
            )
        
        # Get question from text or file
        question = ""
        if question_text:
            question = question_text.strip()
        elif question_file:
            question = await extract_text_from_file(question_file)
        
        if not question:
            raise HTTPException(
                status_code=400, 
                detail="Could not get a question from the provided input."
            )
        
        print(f"Question received: {question[:100]}...")
        
        # Query the collection for relevant context
        context = ""
        retrieved_chunks = []
        
        if collection.count() > 0:
            results = collection.query(query_texts=[question], n_results=3)
            retrieved_chunks = results['documents'][0]
            
            if retrieved_chunks:
                context = "\n\n".join(retrieved_chunks[:3])
                print(f"Found {len(retrieved_chunks)} relevant chunks from documents")
            else:
                print("No relevant chunks found in documents")
        else:
            print("No documents in knowledge base, answering from general knowledge")
        
        # Build messages - with or without document context
        if context:
            # Use document context if available
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful study assistant. Answer questions clearly and accurately. If relevant context from uploaded documents is provided, use it to enhance your answer."
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}

Context from uploaded documents:
{context[:2000]}

Please provide a clear and comprehensive answer. Use the document context if it's relevant, but you can also use your general knowledge to give a complete answer."""
                }
            ]
        else:
            # No document context - answer from general knowledge
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful study assistant. Answer questions clearly, accurately, and comprehensively."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        
        print(f"Calling Hugging Face Inference API with model: {TEXT_MODEL_ID}")
        
        # Use InferenceClient's chat_completion method
        try:
            response = hf_client.chat_completion(
                messages=messages,
                model=TEXT_MODEL_ID,
                max_tokens=400,
                temperature=0.7
            )
            
            # Extract the response text
            answer = response.choices[0].message.content.strip()
            
            if not answer:
                raise HTTPException(
                    status_code=500,
                    detail="Received empty response from AI model."
                )
            
            print(f"✅ Answer generated: {answer[:100]}...")
            
            # Check if answer indicates insufficient information
            if any(phrase in answer.lower() for phrase in [
                "do not contain enough information",
                "cannot answer",
                "insufficient information",
                "not enough context",
                "don't have information"
            ]):
                return {"question": question, "message": answer}
            
            return {
                "question": question, 
                "answer": answer, 
                "source": "Your Uploaded Documents",
                "chunks_used": len(retrieved_chunks),
                "model": TEXT_MODEL_ID
            }
            
        except Exception as api_error:
            print(f"❌ API Error: {api_error}")
            error_msg = str(api_error)
            
            # Handle common errors
            if "loading" in error_msg.lower():
                return {
                    "question": question,
                    "message": "The AI model is currently loading. Please wait 20-30 seconds and try again.",
                    "status": "model_loading"
                }
            elif "rate limit" in error_msg.lower():
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait a moment and try again."
                )
            elif "404" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{TEXT_MODEL_ID}' not available. This model may require a paid plan."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error calling AI model: {error_msg}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during answer generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.delete("/clear-database/")
async def clear_database():
    """Clear all documents from the knowledge base"""
    try:
        client.delete_collection("vit_study_assistant")
        global collection
        collection = client.get_or_create_collection(
            name="vit_study_assistant", 
            embedding_function=hf_ef
        )
        print("✅ Database cleared")
        return {"message": "Knowledge base cleared successfully", "documents_remaining": 0}
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("🚀 Starting VIT Study Assistant API Server...")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)