# CRITICAL: Must be set BEFORE any other imports to bypass proxy for Ollama
import os
os.environ['NO_PROXY'] = '192.168.0.56'
os.environ['no_proxy'] = '192.168.0.56'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="📑 Legal Document Analysis Platform API",
    description="""
    ## 🚀 Comprehensive AI-Powered Legal Document Analysis Platform
    
    **Advanced document processing and analysis capabilities for legal professionals:**
    
    ### 🔍 **Document Processing**
    - **PDF Text Extraction**: Extract text from PDFs with OCR support
    - **Document Structure Analysis**: Tag and analyze document sections
    
    ### 📋 **AI Analysis Features**
    - **Clause Analysis**: Identify and analyze legal clauses with AI insights
    - **Chronology Builder**: Extract and organize chronological events
    - **Document Chat**: Interactive AI assistant for document Q&A
    - **Document Summarization**: Generate intelligent summaries with custom instructions
    - **Risk Assessment**: Identify legal risks with actionable recommendations
    - **Document Classification**: Classify documents by type, subject, and importance
    # - **Document Comparison**: Compare two PDFs and identify differences  # COMMENTED OUT - Document comparison functionality disabled
    
    ### 🎯 **Key Benefits**
    - **AI-Powered**: Advanced language models for accurate analysis
    - **Comprehensive**: Full document lifecycle analysis
    - **Professional**: Designed for legal professionals and compliance teams
    - **Scalable**: RESTful API for integration with existing systems
    
    ### 📚 **Getting Started**
    1. Start with `/extract-text/` to process your PDF documents
    2. Use the extracted data with other analysis endpoints
    3. Explore the interactive API documentation below
    
    ### 🔗 **Integration**
    - **Streamlit Frontend**: http://localhost:8501 (if running)
    - **API Documentation**: Available in the interactive docs below
    - **Health Check**: Use `/health/` endpoint for system status
    """,
    version="1.0.0",
    contact={
        "name": "Legal Document Analysis Platform",
        "url": "http://localhost:8000/docs",
    },
    license_info={
        "name": "MIT License",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routes
app.include_router(router)