import os
import tempfile
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings, SYSTEM_PROMPT
from utils import (
    get_embeddings, 
    get_chat_model, 
    get_pinecone_client,
    validate_file_upload,
    validate_question,
    create_pinecone_index_if_not_exists,
    handle_errors,
    logger
)

@handle_errors
def process_pdf(pdf_file_path: str) -> List[str]:
    """Process PDF file and split into chunks."""
    logger.info(f"Processing PDF: {pdf_file_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    logger.info(f"Processed PDF into {len(chunk_texts)} chunks")
    return chunk_texts

@handle_errors
def store_in_pinecone(chunks: List[str], index_name: str = None) -> int:
    """Store chunks in Pinecone vector database."""
    if index_name is None:
        index_name = settings.pinecone_index_name
        
    logger.info(f"Storing {len(chunks)} chunks in Pinecone index: {index_name}")
    
    # Create index if it doesn't exist
    create_pinecone_index_if_not_exists(index_name)
    
    # Get embeddings and store chunks
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    logger.info(f"Successfully stored {len(chunks)} chunks in Pinecone")
    return len(chunks)

@handle_errors
def retrieve_and_answer(question: str, index_name: str = None) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate answer."""
    if index_name is None:
        index_name = settings.pinecone_index_name
        
    logger.info(f"Processing question: {question[:100]}...")
    
    # Validate question
    validate_question(question)
    
    # Get instances
    embeddings = get_embeddings()
    chat_model = get_chat_model()
    
    # Connect to existing index
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(question, k=settings.max_retrieved_docs)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    logger.info(f"Retrieved {len(docs)} relevant documents")
    
    # Generate answer using Claude Haiku
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    
    response = chat_model.invoke(messages)
    answer = response.content
    
    # Extract sources
    sources = [doc.page_content for doc in docs]
    
    logger.info("Generated answer successfully")
    return {
        "answer": answer,
        "sources": sources
    }

@handle_errors
def process_and_store_pdf(pdf_file) -> int:
    """Process uploaded PDF file and store in Pinecone."""
    logger.info("Processing uploaded PDF file")
    
    # Validate file
    validate_file_upload(pdf_file.filename, len(pdf_file.file.read()))
    
    # Reset file pointer after reading
    pdf_file.file.seek(0)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = pdf_file.file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Process PDF
        chunks = process_pdf(temp_file_path)
        
        # Store in Pinecone
        chunk_count = store_in_pinecone(chunks)
        
        logger.info(f"Successfully processed and stored {chunk_count} chunks")
        return chunk_count
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)
        logger.info("Cleaned up temporary file")
