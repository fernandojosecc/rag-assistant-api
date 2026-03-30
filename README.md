# 🤖 RAG Document Assistant API

A production-ready Retrieval-Augmented Generation (RAG) API that enables intelligent document Q&A using LangChain, Pinecone vector database, and Anthropic Claude Haiku.

## 📋 Project Overview

This RAG document assistant is part of my AI engineering portfolio, demonstrating advanced capabilities in:
- **Vector database integration** with Pinecone
- **Large Language Model orchestration** with Anthropic Claude
- **Embedding generation** with OpenAI
- **API development** with FastAPI
- **Document processing** with LangChain

## 🏗️ Architecture

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│   User  │───▶│ Next.js  │───▶│ FastAPI  │───▶│ LangChain │───▶│ Pinecone │
│ Frontend│    │ Frontend │    │ Backend  │    │ Pipeline  │    │ Vector DB│
└─────────┘    └──────────┘    └──────────┘    └───────────┘    └──────────┘
                                             │
                                             ▼
                                        ┌──────────┐
                                        │  Claude   │
                                        │  Haiku    │
                                        └──────────┘
```

## 🔄 How RAG Works (5-Step Process)

1. **📄 Document Upload**
   - User uploads PDF through Next.js frontend
   - FastAPI receives and validates file (max 10MB)

2. **🔍 Text Processing**
   - LangChain splits document into chunks (500 chars, 50 overlap)
   - Extracts text content using PyPDFLoader

3. **🧠 Vectorization**
   - OpenAI generates embeddings using text-embedding-ada-002
   - Each chunk converted to 1536-dimensional vector

4. **💾 Vector Storage**
   - Vectors stored in Pinecone vector database
   - Enables fast similarity search and retrieval

5. **🤖 Intelligent Q&A**
   - User question converted to embedding
   - Retrieves top 4 most relevant chunks
   - Claude Haiku generates bilingual response (English/Spanish)

## 🛠️ Tech Stack

### Backend
- **Runtime**: Python 3.13
- **Framework**: FastAPI (async, auto-docs)
- **Document Processing**: LangChain, PyPDF
- **Vector Database**: Pinecone (serverless)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Language Model**: Anthropic Claude Haiku
- **Deployment**: Railway (Docker)

### Frontend (Separate Repository)
- **Framework**: Next.js 14
- **Deployment**: Vercel
- **UI/UX**: Modern, responsive design

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key
- Anthropic API key  
- Pinecone API key

### Installation
```bash
# Clone the repository
git clone https://github.com/fernandojosecc/rag-assistant-api.git
cd rag-assistant-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
PINECONE_API_KEY=your-pinecone-api-key

# Optional Configuration
PINECONE_INDEX_NAME=rag-documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Running Locally
```bash
# Start development server
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation
open http://localhost:8000/docs
```

## 📚 API Endpoints

### Health Check
```http
GET /health
```
Returns API health status.

### Document Upload
```http
POST /upload
Content-Type: multipart/form-data
```
Upload and process PDF files (max 10MB).

**Response:**
```json
{
  "status": "success",
  "chunks": 12,
  "message": "Successfully processed document.pdf"
}
```

### Q&A Chat
```http
POST /chat
Content-Type: application/json
```
Ask questions about uploaded documents.

**Request:**
```json
{
  "question": "What is artificial intelligence?"
}
```

**Response:**
```json
{
  "answer": "English: Artificial Intelligence is a branch of computer science...\n\nEspañol: La Inteligencia Artificial es una rama de la informática...\n\nSource: \"Artificial Intelligence (AI) is a branch of computer science...\"",
  "sources": ["Relevant document chunk 1", "Relevant document chunk 2", "..."]
}
```

## 🔧 Configuration

### Settings Overview
All application settings are centralized in `config.py`:

- **Chunk Settings**: Size (500), Overlap (50)
- **Model Settings**: Claude Haiku, OpenAI embeddings
- **File Limits**: 10MB max, PDF only
- **CORS**: Configurable origins
- **Logging**: Structured logging throughout

### Environment Validation
The application validates required environment variables at startup and provides clear error messages for missing configuration.

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your-document.pdf"
```

### Chat Query
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## 🚀 Deployment

### Railway (Production)
```bash
# Deploy to Railway
git push origin main
```

**Environment Variables in Railway:**
- Set all three API keys in Railway dashboard
- Railway automatically assigns port via `$PORT`

### Health Checks
- `/health` endpoint for monitoring
- `/up` endpoint for Railway health checks
- Automatic error handling and logging

## 🔒 Security Features

- **Input Validation**: File size, type, and content validation
- **CORS Configuration**: Configurable origins (development vs production)
- **Error Handling**: Secure error responses, no sensitive data exposure
- **Logging**: Structured logging for monitoring and debugging

## 📊 Performance Optimizations

- **Singleton Pattern**: Models initialized once, reused across requests
- **Connection Pooling**: Pinecone client reuse
- **Lazy Loading**: Resources created only when needed
- **Chunk Optimization**: Balanced size for context and retrieval

## 🤝 Contributing

This is a portfolio project demonstrating AI engineering capabilities. For learning purposes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

**Author**: Fernando Contreras  
**Portfolio**: [Your Portfolio URL]  
**Email**: [Your Email]

## 🔗 Links

- **🚀 Live Demo**: [Your Vercel Frontend URL]
- **📱 Frontend Repo**: [Your Next.js Repository URL]  
- **🔧 Backend Repo**: https://github.com/fernandojosecc/rag-assistant-api
- **👨‍💻 Portfolio**: [Your Portfolio URL]

## 📄 License

This project is part of my AI engineering portfolio and is available for educational purposes.

---

**⭐ Built with ❤️ as part of my AI engineering journey**
