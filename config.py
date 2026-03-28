import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable validation."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    
    # Model Configuration
    openai_embedding_model: str = "text-embedding-ada-002"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    
    # Pinecone Configuration
    pinecone_index_name: str = "rag-documents"
    pinecone_dimension: int = 1536
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-west-2"
    
    # Text Processing Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_retrieved_docs: int = 4
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = [".pdf"]
    
    # API Configuration
    api_title: str = "RAG Document Assistant API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # CORS Configuration
    cors_origins: list = ["*"]  # Default for development, should be restricted in production
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "PINECONE_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


# System prompt for the assistant
SYSTEM_PROMPT = """You are a helpful document assistant.
Answer questions based ONLY on the provided document context.
Always respond with exactly this format:
English: [your answer in English]
Español: [tu respuesta en español]
Source: [copy the most relevant sentence from the context]
If the answer is not in the document, say so in both languages."""
