# Import necessary tools
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import core Python libraries
import PyPDF2
import docx
import io
import os
import pytesseract
from PIL import Image
from typing import Optional, List
import requests
import json
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import traceback

# --- HUGGING FACE IMPORTS ---
from huggingface_hub import InferenceClient

# ==============================================================================
# --- HUGGING FACE CONFIGURATION ---
HF_TOKEN = "hf"  # <-- PASTE YOUR TOKEN HERE

# TEXT MODELS
TEXT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"
IMAGE_TO_TEXT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# TASK-SPECIFIC MODELS
NER_MODEL = "dslim/bert-base-NER"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
TRANSLATION_MODEL = "google-t5/t5-small"  # Multilingual translation
CLASSIFICATION_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TEXT_TO_SPEECH_MODEL = "hexgrad/Kokoro-82M"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("=" * 70)
if not HF_TOKEN or HF_TOKEN == "PASTE_YOUR_TOKEN_HERE" or not HF_TOKEN.startswith("hf_"):
    print("❌ ERROR: Hugging Face token NOT configured!")
    print("Please edit the code and paste your token")
    print(f"Current value: {HF_TOKEN}")
else:
    print(f"✅ Token configured: {HF_TOKEN[:15]}...")
print("\n🤖 Multi-Modal AI Assistant Initializing...")
print(f"📌 TEXT_MODEL: {TEXT_MODEL}")
print(f"📌 IMAGE_MODEL: {IMAGE_MODEL}")
print(f"📌 TTS_MODEL: {TEXT_TO_SPEECH_MODEL}")
print("=" * 70)

# Initialize Inference Client
try:
    hf_client = InferenceClient(token=HF_TOKEN)
    print("✅ Hugging Face Client initialized")
except Exception as e:
    print(f"❌ Error initializing client: {e}")
    hf_client = None

# ==============================================================================
# --- CHROMA DB SETUP ---
try:
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("generated_audio", exist_ok=True)

    client = chromadb.PersistentClient(path="chroma_db")
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(name="vit_study_assistant", embedding_function=hf_ef)
    print(f"✅ ChromaDB initialized. Documents loaded: {collection.count()}")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")

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
app.mount("/generated_images", StaticFiles(directory="generated_images"), name="generated_images")
app.mount("/generated_audio", StaticFiles(directory="generated_audio"), name="generated_audio")

# --- FILE EXTRACTION ---
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
        print(f"Error extracting text: {e}")
        return None
    return text.strip() if text else None

# ==============================================================================
# --- CORE ENDPOINTS ---
# ==============================================================================

@app.post("/chat")
async def chat(question_text: Optional[str] = Form(None)):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")
        if not question_text:
            raise HTTPException(status_code=400, detail="Please provide a question")

        print(f"💬 Chat: {question_text[:50]}...")

        messages = [
            {"role": "system", "content": "You are a helpful and creative AI assistant."},
            {"role": "user", "content": question_text}
        ]

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = hf_client.chat_completion(
                    messages=messages,
                    model=TEXT_MODEL,
                    max_tokens=500,
                    temperature=0.7
                )
                break
            except Exception as retry_err:
                if attempt < max_retries - 1:
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    raise retry_err

        if not response or not response.choices:
            raise ValueError("Empty response from model")

        answer = response.choices[0].message.content.strip()
        print(f"✅ General Chat generated")

        return {
            "question": question_text,
            "answer": answer,
            "source": "General AI Knowledge"
        }
    except Exception as e:
        print(f"❌ Error during chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/document-qa")
async def document_qa(question_text: str = Form(...)):
    try:
        if collection.count() == 0:
            raise HTTPException(status_code=400, detail="The knowledge base is empty. Please upload documents first.")

        question = question_text.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question.")

        print(f"📚 Q&A: {question[:50]}...")

        results = collection.query(query_texts=[question], n_results=3)
        retrieved_chunks = results['documents'][0]

        if not retrieved_chunks:
            return {"question": question, "answer": "I couldn't find any relevant information in the documents.", "source": "Documents"}

        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"Based on this context, answer the question:\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"

        messages = [
            {"role": "system", "content": "You are a helpful study assistant. Answer based only on the context."},
            {"role": "user", "content": prompt}
        ]

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = hf_client.chat_completion(
                    messages=messages,
                    model=TEXT_MODEL,
                    max_tokens=500,
                    temperature=0.7
                )
                break
            except Exception as retry_err:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise retry_err

        answer = response.choices[0].message.content.strip()
        print(f"✅ Document Q&A generated")

        return {"question": question, "answer": answer, "source": "Documents"}

    except Exception as e:
        print(f"❌ Error during document Q&A: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Q&A error: {str(e)}")

@app.post("/upload-file")
async def upload_file(uploaded_file: UploadFile = File(...)):
    try:
        print(f"📄 Processing: {uploaded_file.filename}")
        full_text = await extract_text_from_file(uploaded_file)

        if not full_text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        chunks = [chunk.strip() for chunk in full_text.split('\n\n') if len(chunk.strip()) > 50]
        if not chunks:
            chunks = [chunk.strip() for chunk in full_text.split('\n') if len(chunk.strip()) > 50]

        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful text found")

        collection.add(
            documents=chunks,
            metadatas=[{"source": uploaded_file.filename} for _ in chunks],
            ids=[f"{uploaded_file.filename}_{i}" for i in range(len(chunks))]
        )

        print(f"✅ Learned {len(chunks)} chunks")

        all_items = collection.get()
        learned_files = list(set(meta['source'] for meta in all_items['metadatas'])) if all_items['metadatas'] else []

        return {
            "success": True,
            "message": f"Learned {len(chunks)} chunks from: {uploaded_file.filename}",
            "learned_files": learned_files,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# --- IMAGE GENERATION ---
# ==============================================================================

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")

        print(f"🎨 Generating image: {prompt[:100]}...")
        image = hf_client.text_to_image(prompt, model=IMAGE_MODEL)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        filepath = os.path.join("generated_images", filename)
        image.save(filepath)

        print(f"✅ Image saved: {filename}")

        return {
            "message": "Image generated successfully!",
            "image_url": f"/generated_images/{filename}",
            "prompt": prompt
        }
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

# ==============================================================================
# --- TEXT PROCESSING ENDPOINTS ---
# ==============================================================================

@app.post("/classify-text")
async def text_classification(text: str = Form(...), labels: str = Form(...)):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")

        candidate_labels = [label.strip() for label in labels.split(',')]
        if not candidate_labels:
            raise HTTPException(status_code=400, detail="Please provide labels separated by commas")

        print(f"🏷️ Classifying text: {text[:50]}...")
        response = hf_client.zero_shot_classification(
            text,
            candidate_labels,
            model=CLASSIFICATION_MODEL,
        )

        print(f"✅ Text Classified")
        return {"text": text, "classification": response}
    except Exception as e:
        print(f"❌ Error classifying text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/ner")
async def token_classification(text: str = Form(...)):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")

        print(f"🧩 NER on text: {text[:50]}...")
        response = hf_client.token_classification(text, model=NER_MODEL)

        print(f"✅ NER Completed")
        return {"text": text, "entities": response}
    except Exception as e:
        print(f"❌ Error performing NER: {str(e)}")
        raise HTTPException(status_code=500, detail=f"NER error: {str(e)}")

@app.post("/summarize")
async def summarize(text: str = Form(...)):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")

        print(f"📝 Summarizing text: {text[:100]}...")
        summary = hf_client.summarization(text, model=SUMMARIZATION_MODEL)

        print(f"✅ Summarization Completed")
        if isinstance(summary, list) and summary and isinstance(summary[0], dict) and 'summary_text' in summary[0]:
            summary_text = summary[0]['summary_text']
        else:
            summary_text = str(summary)

        return {
            "original_text": text[:200] + "...",
            "summary": summary_text
        }
    except Exception as e:
        print(f"❌ Error during summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/translate")
async def translate(
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("es")
):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")

        print(f"🌍 Translating: {source_lang} -> {target_lang}")

        prompt = f"Translate from {source_lang} to {target_lang}. Only output the translation:\n\n{text}"

        messages = [
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": prompt}
        ]

        response = hf_client.chat_completion(
            messages=messages,
            model=TEXT_MODEL,
            max_tokens=500,
            temperature=0.3
        )

        translation = response.choices[0].message.content.strip()
        print(f"✅ Translation completed")

        return {
            "original": text,
            "translation": translation,
            "source_language": source_lang,
            "target_language": target_lang
        }
    except Exception as e:
        print(f"❌ Error translating: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

# ==============================================================================
# --- SPEECH TASKS ---
# ==============================================================================

@app.post("/text-to-speech")
async def text_to_speech(text: str = Form(...)):
    try:
        if not hf_client:
            raise HTTPException(status_code=500, detail="AI Client not initialized")

        print(f"🔊 Converting to speech: {text[:50]}...")

        try:
            audio_bytes = hf_client.text_to_speech(text, model=TEXT_TO_SPEECH_MODEL)
        except Exception as tts_err:
            print(f"⚠️ TTS model unavailable: {tts_err}")
            raise HTTPException(status_code=503, detail="Text-to-speech service temporarily unavailable")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{timestamp}.wav"
        filepath = os.path.join("generated_audio", filename)

        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        print(f"✅ Audio saved: {filename}")

        return {
            "message": "Speech generated successfully!",
            "audio_url": f"/generated_audio/{filename}",
            "text": text[:100] + "..." if len(text) > 100 else text
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation error: {str(e)}")


# ==============================================================================
# --- UTILITY ENDPOINTS ---
# ==============================================================================

@app.delete("/clear-database")
async def clear_database():
    try:
        global collection
        client.delete_collection("vit_study_assistant")
        collection = client.get_or_create_collection(name="vit_study_assistant", embedding_function=hf_ef)
        print("✅ Database cleared")
        return {"message": "Knowledge base cleared", "documents_remaining": 0}
    except Exception as e:
        print(f"❌ Clear DB error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_loaded": collection.count(),
        "token_status": "✅ configured" if HF_TOKEN.startswith("hf_") else "❌ not_configured",
        "features": {
            "core": ["chat", "document_qa", "upload_file"],
            "text": ["classify_text", "ner", "summarize", "translate"],
            "vision": ["generate_image"],
            "speech": ["text_to_speech"]
        },
        "models": {
            "text": TEXT_MODEL,
            "image_gen": IMAGE_MODEL,
            "image_to_text": IMAGE_TO_TEXT_MODEL,
            "ner": NER_MODEL,
            "summarization": SUMMARIZATION_MODEL,
            "translation": TRANSLATION_MODEL,
            "classification": CLASSIFICATION_MODEL,
            "tts": TEXT_TO_SPEECH_MODEL,
            "embedding": EMBEDDING_MODEL
        }
    }

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("🚀 Starting Multi-Modal AI Assistant...")
    print("📋 Available Features:")
    print("   ✅ Chat & Document Q&A")
    print("   ✅ Text Classification & NER")
    print("   ✅ Summarization & Translation")
    print("   ✅ Image Generation")
    print("   ✅ Text-to-Speech")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)