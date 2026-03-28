import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone, ServerlessSpec
import tempfile

# Initialize embeddings and models (will be initialized when needed)
embeddings = None
chat_model = None

# System prompt for the assistant
SYSTEM_PROMPT = """You are a helpful document assistant.
Answer questions based ONLY on the provided document context.
Always respond with exactly this format:
English: [your answer in English]
Español: [tu respuesta en español]
Source: [copy the most relevant sentence from the context]
If the answer is not in the document, say so in both languages."""

def process_pdf(pdf_file_path: str) -> List[str]:
    """Process PDF file and split into chunks."""
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def store_in_pinecone(chunks: List[str], index_name: str = "rag-documents") -> int:
    """Store chunks in Pinecone vector database."""
    try:
        # Initialize embeddings
        global embeddings
        if embeddings is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        pinecone = Pinecone(api_key=pinecone_api_key)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        
        # Store chunks in Pinecone
        vector_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=index_name
        )
        
        return len(chunks)
    
    except Exception as e:
        raise Exception(f"Error storing in Pinecone: {str(e)}")

def retrieve_and_answer(question: str, index_name: str = "rag-documents") -> Dict[str, Any]:
    """Retrieve relevant chunks and generate answer."""
    try:
        # Initialize embeddings
        global embeddings
        if embeddings is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize chat model
        global chat_model
        if chat_model is None:
            chat_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        # Connect to existing index
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Retrieve relevant chunks
        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer using Claude Haiku
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]
        
        response = chat_model.invoke(messages)
        answer = response.content
        
        # Extract sources
        sources = [doc.page_content for doc in docs]
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        raise Exception(f"Error retrieving and answering: {str(e)}")

def process_and_store_pdf(pdf_file) -> int:
    """Process uploaded PDF file and store in Pinecone."""
    try:
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
            
            return chunk_count
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        raise Exception(f"Error processing and storing PDF: {str(e)}")
