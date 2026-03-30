from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
from dotenv import load_dotenv
from config import settings, validate_environment
from rag_pipeline import process_and_store_pdf, retrieve_and_answer
import logging

# Load environment variables and validate with try-catch
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment variables safely
try:
    validate_environment()
    logger.info("Environment variables validated successfully")
except ValueError as e:
    logger.error(f"Environment validation failed: {str(e)}")
    logger.warning("Starting app without environment validation - this may cause runtime errors")

# Initialize FastAPI app
app = FastAPI(title="RAG Document Assistant API", version="1.0.0")

# Add CORS middleware
origins = [
    "https://rag-assistant-ui.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for chat request
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask about the documents")

# Pydantic model for chat response
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

# Pydantic model for upload response
class UploadResponse(BaseModel):
    status: str
    chunks: int
    message: str = None

# Pydantic model for error response
class ErrorResponse(BaseModel):
    detail: str
    error_type: str = None

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    - Accepts a PDF file (max 10MB)
    - Processes it with LangChain into chunks
    - Stores chunks as vectors in Pinecone using OpenAI embeddings
    - Returns success status and number of chunks
    """
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {settings.max_file_size // (1024*1024)}MB"
            )
        
        # Reset file pointer
        file.file.seek(0)
        
        # Process and store the PDF
        chunk_count = process_and_store_pdf(file)
        
        logger.info(f"Successfully processed {file.filename} into {chunk_count} chunks")
        return UploadResponse(
            status="success", 
            chunks=chunk_count,
            message=f"Successfully processed {file.filename}"
        )
    
    except ValueError as e:
        logger.error(f"Validation error in upload: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer questions using RAG pipeline.
    
    - Accepts a question in JSON format (max 1000 characters)
    - Retrieves top 4 relevant chunks from Pinecone
    - Sends chunks + question to Claude Haiku
    - Returns answer in both English and Spanish with sources
    """
    try:
        logger.info(f"Received chat request: {request.question[:100]}...")
        
        # Retrieve relevant chunks and generate answer
        result = retrieve_and_answer(request.question)
        
        logger.info("Successfully generated chat response")
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    
    except ValueError as e:
        logger.error(f"Validation error in chat: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/startup-test")
async def startup_test():
    """
    Simple test endpoint to verify Railway deployment is working.
    """
    return {
        "status": "working",
        "message": "Application started successfully",
        "timestamp": str(os.times()),
        "env_vars": {
            "OPENAI_API_KEY": "✓" if os.getenv("OPENAI_API_KEY") else "✗",
            "ANTHROPIC_API_KEY": "✓" if os.getenv("ANTHROPIC_API_KEY") else "✗",
            "PINECONE_API_KEY": "✓" if os.getenv("PINECONE_API_KEY") else "✗",
        }
    }

@app.get("/up")
async def railway_health_check():
    """
    Simple health check for Railway.
    Railway often checks /up endpoint.
    """
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Railway deployment.
    
    - Returns simple status check
    - Used by Railway to verify the service is running
    """
    return {"status": "ok"}

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    try:
        return {
            "message": settings.api_title,
            "version": settings.api_version,
            "endpoints": {
                "upload": "POST /upload - Upload PDF files (max 10MB)",
                "chat": "POST /chat - Ask questions about uploaded documents (max 1000 chars)",
                "health": "GET /health - Health check endpoint",
                "startup-test": "GET /startup-test - Debug endpoint"
            }
        }
    except Exception as e:
        logger.error(f"Root endpoint failed: {str(e)}")
        return {
            "message": "RAG Document Assistant API",
            "version": "1.0.0",
            "status": "limited functionality",
            "endpoints": {
                "startup-test": "GET /startup-test - Debug endpoint",
                "health": "GET /health - Health check endpoint"
            }
        }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port
    )
