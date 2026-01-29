# Dockerfile for Chatbot API Service
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY master_requirements.txt .
RUN pip install --no-cache-dir -r master_requirements.txt
RUN pip install --no-cache-dir langchain-groq

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
