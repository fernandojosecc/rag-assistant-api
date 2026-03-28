import logging
import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances for singleton pattern
_embeddings_instance: Optional[OpenAIEmbeddings] = None
_chat_model_instance: Optional[ChatAnthropic] = None
_pinecone_client: Optional[Pinecone] = None


def get_embeddings() -> OpenAIEmbeddings:
    """Get or create OpenAI embeddings instance (singleton pattern)."""
    global _embeddings_instance
    
    if _embeddings_instance is None:
        logger.info("Initializing OpenAI embeddings")
        _embeddings_instance = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )
    
    return _embeddings_instance


def get_chat_model() -> ChatAnthropic:
    """Get or create Anthropic chat model instance (singleton pattern)."""
    global _chat_model_instance
    
    if _chat_model_instance is None:
        logger.info("Initializing Anthropic chat model")
        _chat_model_instance = ChatAnthropic(
            model=settings.anthropic_model,
            temperature=0,
            anthropic_api_key=settings.anthropic_api_key
        )
    
    return _chat_model_instance


def get_pinecone_client() -> Pinecone:
    """Get or create Pinecone client instance (singleton pattern)."""
    global _pinecone_client
    
    if _pinecone_client is None:
        logger.info("Initializing Pinecone client")
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    
    return _pinecone_client


def validate_file_upload(filename: str, file_size: int) -> None:
    """Validate uploaded file meets requirements."""
    # Check file extension
    file_ext = os.path.splitext(filename.lower())[1]
    if file_ext not in settings.allowed_file_types:
        raise ValueError(f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_types}")
    
    # Check file size
    if file_size > settings.max_file_size:
        raise ValueError(f"File size {file_size} exceeds maximum allowed size {settings.max_file_size}")


def validate_question(question: str) -> None:
    """Validate chat question meets requirements."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if len(question) > 1000:  # Reasonable limit
        raise ValueError("Question too long. Maximum 1000 characters allowed")


def create_pinecone_index_if_not_exists(index_name: str) -> None:
    """Create Pinecone index if it doesn't exist."""
    pinecone = get_pinecone_client()
    
    if index_name not in pinecone.list_indexes().names():
        logger.info(f"Creating Pinecone index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=settings.pinecone_dimension,
            metric=settings.pinecone_metric,
            spec={
                "cloud": settings.pinecone_cloud,
                "region": settings.pinecone_region
            }
        )


def handle_errors(func):
    """Decorator for consistent error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise Exception(f"An error occurred: {str(e)}")
    
    return wrapper
