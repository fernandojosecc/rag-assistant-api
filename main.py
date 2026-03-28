from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv
from rag_pipeline import process_and_store_pdf, retrieve_and_answer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Document Assistant API", version="1.0.0")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for chat request
class ChatRequest(BaseModel):
    question: str

# Pydantic model for chat response
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

# Pydantic model for upload response
class UploadResponse(BaseModel):
    status: str
    chunks: int

# Pydantic model for health response
class HealthResponse(BaseModel):
    status: str

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    - Accepts a PDF file
    - Processes it with LangChain into chunks
    - Stores chunks as vectors in Pinecone using OpenAI embeddings
    - Returns success status and number of chunks
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Process and store the PDF
        chunk_count = process_and_store_pdf(file)
        
        return UploadResponse(status="success", chunks=chunk_count)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer questions using RAG pipeline.
    
    - Accepts a question in JSON format
    - Retrieves top 4 relevant chunks from Pinecone
    - Sends chunks + question to Claude Haiku
    - Returns answer in both English and Spanish with sources
    """
    try:
        # Retrieve relevant chunks and generate answer
        result = retrieve_and_answer(request.question)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Railway deployment.
    
    - Returns simple status check
    - Used by Railway to verify the service is running
    """
    return HealthResponse(status="ok")

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "RAG Document Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload PDF files",
            "chat": "POST /chat - Ask questions about uploaded documents",
            "health": "GET /health - Health check endpoint"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
