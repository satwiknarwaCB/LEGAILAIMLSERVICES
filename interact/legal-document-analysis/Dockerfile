# Multi-stage Dockerfile for Legal Document Analysis Platform
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
# dependencies.txt contains pinned versions for reproducible builds
# requirements.txt is kept for reference/compatibility
COPY requirements.txt dependencies.txt ./

# Install Python dependencies (use dependencies.txt for exact version pinning)
RUN pip install --no-cache-dir -r dependencies.txt
RUN pip install --no-cache-dir langchain-groq

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "run_backend.py"]

