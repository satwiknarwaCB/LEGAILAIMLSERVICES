# Legal Document Analysis API Documentation

## Overview
The Legal Document Analysis platform provides RESTful APIs for processing legal documents, extracting chronological events, and building timelines. All endpoints are accessible without authentication.

## Base URL
```
http://localhost:8001
```

## API Endpoints

### Document Text Extraction
**POST /extract-text/**
Extract text content from PDF documents or images with OCR support
- Request: Multipart form data with file upload
- Response: Extracted text with document structure tags

### Legal Clause Analysis
**POST /analyze-clauses/**
Identify and classify legal clauses within documents
- Request: Document text content
- Response: Classified clauses with risk assessments

### Chronological Event Extraction
**POST /extract-chronology/**
Extract chronological events and build timeline sequences
- Request: Document text content
- Response: Structured timeline with dated events

### Document Chat
**POST /chat-with-document/**
Interactive Q&A about document content
- Request: User question and document context
- Response: AI-generated answers with source references

### Document Summarization
**POST /summarize-documents/**
Generate customizable document summaries
- Request: Document text with summary preferences
- Response: Concise summary with key points

### Legal Risk Assessment
**POST /assess-risks/**
Identify potential legal risks in documents
- Request: Document text content
- Response: Risk categories with mitigation recommendations

### Document Classification
**POST /classify-document/**
Categorize documents by type and subject matter
- Request: Document text content
- Response: Classification tags with confidence scores

### System Health Check
**GET /health/**
Monitor system status and service availability
- Response: Health status with component details

## Error Handling
- 200: Success
- 400: Bad Request
- 422: Unprocessable Entity
- 500: Internal Server Error

## Rate Limits
- 100 requests per minute per client
- 10MB maximum file size for uploads

## Data Formats
- Request/Response: JSON
- File Uploads: multipart/form-data
- Supported Files: PDF, PNG, JPG, JPEG

## Integration Notes
1. All endpoints are CORS-enabled for localhost development
2. File uploads require proper MIME type headers
3. Large document processing may take 5-30 seconds
4. Responses include processing timestamps for performance monitoring

The API provides comprehensive legal document analysis capabilities without requiring authentication, making it suitable for integration into various legal tech workflows.