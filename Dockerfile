# MASTER DOCKERFILE - LEGAILAIMLSERVICES
# Consolidates all 6 services into a single high-performance image

FROM python:3.12-slim

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
# Includes Tesseract OCR with English, Hindi, and Marathi support
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    libtesseract-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files first for better caching
COPY requirements.txt .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install master Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install spaCy NLP model
RUN python -m spacy download en_core_web_sm

# Copy the entire project code
COPY . .

# Create log directory for supervisor
RUN mkdir -p /var/log/

# Expose all service ports
# 8000: Chatbot API
# 8001: Interact API
# 8002: Agents API
# 8080: Auth Service
# 8501: Interact Frontend
# 8502: Chatbot Frontend
EXPOSE 8000 8001 8002 8080 8501 8502

# Set environment variables for inter-service communication inside the container
ENV AUTH_SERVICE_URL=http://localhost:8080
ENV CHATBOT_API_URL=http://localhost:8000
ENV INTERACT_API_URL=http://localhost:8001
ENV AGENTS_API_URL=http://localhost:8002
ENV APP_ENV=production

# Start all services using Supervisor
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
