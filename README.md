# 📑 LEGAILAIMLSERVICES: Multi-Service Legal AI Platform

Welcome to the **Nexus Legal AI Platform**. This repository integrates multiple AI microservices to provide a comprehensive legal assistant, document analysis engine, and drafting suite.

---

## 🏛️ Architecture Overview

The platform is divided into three core services:

1.  **`ASK,DRAFT/Legal-Chatbot-Updated`**: The primary RAG (Retrieval-Augmented Generation) engine. Handles legal Q&A using Qdrant Cloud and Ollama.
2.  **`interact/legal-document-analysis`**: A specialized analysis service for extracting clauses, assessment of risks, and document summarization.
3.  **`authentication`**: Centralized JWT-based auth service to secure endpoints across the platform.

---

## 🚀 Getting Started

### 1. Environment Configuration
Each service requires its own `.env` file. Copy the provided `.env.example` files and fill in your credentials:
- `QDRANT_API_KEY`
- `GROQ_API_KEY`
- `OLLAMA_BASE_URL` (Points to your local GPU node)

### 2. Dependency Management
Install the necessary Python packages for each service:
```bash
# Example for Chatbot
cd ASK,DRAFT/Legal-Chatbot-Updated
pip install -r master_requirements.txt
```

### 3. Running the Services
Start the services in the following order:
1. **Auth Service**: `python auth_service.py` (Port 5000)
2. **Analysis API**: `uvicorn app.main:app --port 8001`
3. **Chatbot API**: `uvicorn api:app --port 8000`

---

## 🛠️ Technology Stack
- **Backend**: FastAPI (Python)
- **Vector DB**: Qdrant Cloud
- **LLM Orchestration**: LangChain & Ollama
- **Models**: Qwen 2.5 (14B), Llama 3.3 (70B)
- **Database**: SQLite (Local caching) & MongoDB (Metadata)

---

## 🔒 Security
- **No Proxy Protocol**: Configured to bypass corporate proxies for internal GPU node connectivity.
- **JWT Protection**: All analysis and drafting endpoints require a valid bearer token.

---
*Generated with precision by the Cognitbotz Engineering Team.*
