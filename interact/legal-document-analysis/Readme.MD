# Legal Document Analysis & Chronology Builder

A comprehensive AI-powered system for analyzing legal documents, extracting events, and building chronological timelines.

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Prerequisites**: Docker and Docker Compose installed
2. **Setup**: Create `.env` file with API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   WEAVIATE_API_KEY=your_weaviate_api_key_here  # Optional
   WEAVIATE_URL=your_weaviate_cluster_url_here  # Optional
   ```
3. **Start**: `docker-compose up -d`
4. **Access**: http://localhost:8501 (Frontend) | http://localhost:8000/docs (API)

### Using Python

1. **Prerequisites**: Python 3.11+, virtual environment
2. **Install**: `pip install -r dependencies.txt`
3. **Setup**: Create `.env` file (see above)
4. **Run Backend**: `python run_backend.py` (Terminal 1)
5. **Run Frontend**: `python run_frontend.py` (Terminal 2)
6. **Access**: http://localhost:8501

## 📋 Features

- **📝 Document Text Extraction**: Handles PDFs and images (PNG, JPG, JPEG) with OCR
- **📋 Clause Analysis**: AI-powered legal clause classification and analysis
- **📅 Chronology Builder**: Automatic event extraction and timeline construction
- **💬 Document Chat**: Interactive AI assistant for document Q&A
- **📄 Document Summarization**: Customizable summaries with key points
- **⚠️ Risk Assessment**: Legal risk identification with actionable recommendations
- **🏷️ Document Classification**: Classify by type, subject, and importance
- **🔍 Advanced Search**: BM25, vector, and hybrid search capabilities

## 🛠️ Installation

### System Requirements
- Python 3.11+ or Docker
- 4GB RAM minimum (8GB recommended)
- Internet connection for AI processing

### Python Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r dependencies.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Create .env file with your API keys
```

## 🖥️ Running the System

### Option 1: Docker (Recommended for Production)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build

# Check status
docker-compose ps
```

**Docker Commands:**
- `docker-compose up -d` - Start in background
- `docker-compose logs -f [service]` - View logs
- `docker-compose restart [service]` - Restart service
- `docker-compose exec [service] bash` - Access container shell
- `docker-compose down` - Stop and remove containers

### Option 2: Python Scripts

```bash
# Terminal 1 - Backend
python run_backend.py

# Terminal 2 - Frontend
python run_frontend.py
```

### Option 3: Manual Commands

```bash
# Terminal 1 - Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend
streamlit run streamlit_frontend.py --server.port 8501
```

## 📱 Usage

### Web Interface (Streamlit)

Access at http://localhost:8501

**Workflow:**
1. Upload a PDF document or image (PNG, JPG, JPEG)
2. Select analysis mode:
   - Text Extraction
   - Clause Analysis
   - Chronology Builder
   - Document Chat
   - Document Summarization
   - Risk Assessment
   - Document Classification
3. Configure options (if applicable)
4. View results and export data

### API Endpoints

Access API docs at http://localhost:8000/docs

**Main Endpoints:**
- `POST /extract-text/` - Extract and tag text from PDF or images (PNG, JPG, JPEG)
- `POST /analyze-clauses/` - Analyze legal clauses
- `POST /extract-chronology/` - Extract chronological events
- `POST /chat-with-document/` - Chat with AI about document
- `POST /summarize-documents/` - Generate summaries
- `POST /assess-risks/` - Assess legal risks
- `POST /classify-document/` - Classify documents
- `GET /health/` - System health check

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │    │   FastAPI API   │
│   (Frontend)    │◄───┤    (Backend)    │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │   Core Services     │
         │  - Extractor        │
         │  - Tagging          │
         │  - LLM Processor    │
         │  - Chronology       │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  External Services  │
         │  - Groq LLM         │
         │  - Weaviate (opt)   │
         └─────────────────────┘
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for vector search)
WEAVIATE_API_KEY=your_weaviate_api_key_here
WEAVIATE_URL=your_weaviate_cluster_url_here
```

### Docker Configuration

- **Backend Port**: 8000
- **Frontend Port**: 8501
- **Data Volume**: `./data` (mounted for persistence)
- **Health Checks**: Enabled for both services

## 🛠️ Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill
```

**Module Not Found:**
```bash
# Reinstall dependencies
pip install -r dependencies.txt
```

**Docker Issues:**
```bash
# Check logs
docker-compose logs

# Rebuild
docker-compose up -d --build

# Verify .env file
docker-compose exec backend env | grep GROQ
```

**API Key Errors:**
- Verify `.env` file exists
- Check no extra spaces/quotes around values
- Ensure file is in project root

**Connection Refused:**
- Ensure backend is running before frontend
- Check firewall settings
- Verify ports 8000 and 8501 are available

## 📚 Technology Stack

- **Backend**: FastAPI, Python 3.11
- **Frontend**: Streamlit
- **AI/ML**: LangChain, Groq LLM (Llama 3.1)
- **PDF Processing**: PyPDF2, pypdfium2, Tesseract OCR
- **NLP**: spaCy
- **Vector DB**: Weaviate (optional)
- **Containerization**: Docker, Docker Compose

## 📖 Key Components

- **Extractor Service**: PDF text extraction with OCR fallback
- **Tagging Service**: Document structure analysis
- **LLM Processor**: AI-powered analysis and classification
- **Chronology Builder**: Event extraction and timeline construction
- **Temporal Processor**: Date normalization and resolution
- **Weaviate Store**: Vector database for search (optional)

## 🎯 Use Cases

- Legal document analysis and review
- Contract clause identification
- Chronological event extraction
- Risk assessment and compliance
- Document classification and organization
- Interactive document Q&A

## 📝 Development

### Project Structure

```
.
├── app/                 # Backend application
│   ├── main.py         # FastAPI app
│   ├── routes.py       # API endpoints
│   ├── models/          # Data models
│   └── services/       # Business logic
├── streamlit_frontend.py  # Frontend UI
├── docker-compose.yml  # Docker orchestration
├── Dockerfile          # Docker image definition
├── dependencies.txt    # Python dependencies (pinned)
├── requirements.txt    # Python dependencies (loose)
└── .env               # Environment variables
```

### Making Changes

**With Docker:**
```bash
# Edit code locally
# Restart service
docker-compose restart backend
# Or rebuild
docker-compose up -d --build
```

**With Python:**
- Backend auto-reloads on changes
- Frontend auto-reloads on changes

## 📄 License

MIT License

## 🤝 Support

For issues and questions:
1. Check logs: `docker-compose logs` or console output
2. Review API docs: http://localhost:8000/docs
3. Verify environment variables
4. Check system requirements

---

**Built with ❤️ for legal professionals**
