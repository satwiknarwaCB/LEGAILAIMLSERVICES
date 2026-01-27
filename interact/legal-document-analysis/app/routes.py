from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from datetime import date, datetime
from typing import Optional, List
# Only using first page extraction functions - commenting out unused full PDF functions
from .services.extractor import extract_from_pdf_first_page_only, extract_from_scanned_pdf_first_page_only, extract_from_image
# from .services.extractor import extract_from_pdf, extract_from_scanned_pdf  # Unused - full PDF processing not needed
from .services.tagging import tag_documents
from .services.llm_processor import LLMProcessor
from .services.weaviate_store import WeaviateStore
# Only using specific models - commenting out unused ones
from .models.chronology import (ClauseAnalysisRequest, FlexibleClauseAnalysisRequest, ChronologyFromSectionsRequest, TaggedSection, 
                               DocumentChatRequest, DocumentChatResponse, ChatMessage,
                               DocumentSummarizationRequest, DocumentSummarizationResponse, DocumentSummary,
                               RiskAssessmentRequest, RiskAssessmentResponse, RiskItem,
                               DocumentClassificationRequest, DocumentClassificationResponse, DocumentClassification)
# COMMENTED OUT - Document comparison functionality disabled
# from .models.chronology import (DocumentComparisonRequest, DocumentComparisonResponse, DocumentComparison,
#                                ClauseDifference, ObligationDifference, TimelineDifference)
# from .models.chronology import ChronologyRequest, ChronologyResponse  # Unused - not needed for current endpoints

router = APIRouter()

# Initialize services
llm_processor = LLMProcessor()
weaviate_store = WeaviateStore()

# ===== SYSTEM ENDPOINTS =====

@router.get("/")
async def root():
    """
    🏠 API Root
    
    Welcome to the Legal Document Analysis Platform API.
    
    **Available Features:**
    - PDF Text Extraction & Analysis
    - AI Clause Analysis
    - AI Chronology Builder
    - AI Document Chat Assistant
    - AI Document Summarization
    - AI Legal Risk Assessment
    - AI Document Classification
    # - AI Document Comparison  # COMMENTED OUT - Document comparison functionality disabled
    
    **Next Steps:**
    - Visit `/docs` for interactive API documentation
    - Use `/health` to check system status
    - Start with `/extract-text/` to process documents
    """
    return {
        "message": "Welcome to the Legal Document Analysis Platform API",
        "version": "1.0.0",
        "features": [
            "PDF Text Extraction & Analysis",
            "AI Clause Analysis", 
            "AI Chronology Builder",
            "AI Document Chat Assistant",
            "AI Document Summarization",
            "AI Legal Risk Assessment",
            "AI Document Classification"
            # "AI Document Comparison"  # COMMENTED OUT - Document comparison functionality disabled
        ],
        "documentation": "/docs",
        "health_check": "/health",
        "streamlit_frontend": "http://localhost:8501 (if running)"
    }

@router.get("/health")
async def health_check():
    """
    🏥 System Health Check
    
    Check the health status of the API and its dependencies.
    
    **Returns:**
    - API status
    - Service availability
    - System information
    """
    try:
        # Check if services are initialized
        llm_status = "healthy" if llm_processor else "unhealthy"
        weaviate_status = "healthy" if weaviate_store else "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "llm_processor": llm_status,
                "weaviate_store": weaviate_status
            },
            "endpoints": {
                "total": 8,
                "document_processing": 2,
                "ai_analysis": 6
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# ===== SHARED TEXT EXTRACTION FUNCTION =====

async def extract_text_from_file(file: UploadFile) -> str:
    """Shared text extraction logic - handles PDF and image files"""
    file_bytes = await file.read()
    filename = file.filename.lower() if file.filename else ""
    
    # Check if it's an image file
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        return extract_from_image(file_bytes)
    
    # Otherwise, treat as PDF
    try:
        text = extract_from_pdf_first_page_only(file_bytes)
    except Exception:
        text = ""
    
    if not text.strip():
        text = extract_from_scanned_pdf_first_page_only(file_bytes)
    
    return text

# ===== TEXT EXTRACTION ENDPOINT =====

@router.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    """
    📝 Document Text Extraction & Analysis
    
    Extract text from PDF documents or images (PNG, JPG, JPEG) and analyze document structure.
    
    **Features:**
    - Extracts text from PDF documents (first page) or images
    - Handles both regular and scanned PDFs with OCR
    - Supports PNG, JPG, JPEG images with OCR
    - Analyzes document structure and tags sections
    - Provides text length and processing metadata
    
    **Use Case:** First step for all document analysis workflows
    """
    try:
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        
        # Structure document into tagged sections
        tagged_sections = tag_documents(text)
        
        return JSONResponse(content={
            "tagged_sections": tagged_sections,
            "total_sections": len(tagged_sections),
            "raw_text_length": len(text),
            "processing_metadata": {
                "document_type": "text_extraction_and_tagging",
                "timestamp": datetime.now().isoformat(),
                "pages_processed": "first_page_only",
                "extraction_method": "pdf_ocr_fallback",
                "processing_steps": ["text_extraction", "document_tagging"]
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting and tagging text: {str(e)}")

# ===== CLAUSE CLASSIFICATION/SUMMARIZATION ENDPOINT =====

@router.post("/analyze-clauses/")
async def analyze_clauses(
    file: Optional[UploadFile] = File(None),
    document_type: Optional[str] = Query(None, description="Type of document being analyzed")
):
    """
    📋 AI Clause Analysis
    
    Identify and analyze legal clauses with AI insights from PDF documents or images.
    
    **Features:**
    - Analyzes document structure and identifies legal clauses
    - Tags sections with headings and content
    - Provides clause analysis with AI-powered insights
    - Extracts key legal elements and provisions
    
    **Input:** PDF or image file upload (multipart/form-data) - supports PDF, PNG, JPG, JPEG
    
    **Use Case:** Deep analysis of legal document structure and clauses
    """
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="PDF file is required")
        
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG, JPEG)")
        
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Tag the document
        tagged_sections = tag_documents(text)
        doc_type = document_type or "legal_document"
        
        # Analyze clauses using LLM
        clause_results = llm_processor.process_contract(tagged_sections)
        
        # Store in Weaviate for search
        weaviate_store.create_class_obj("LegalClauses")
        weaviate_store.add_documents("LegalClauses", tagged_sections)
        
        return JSONResponse(content={
            "clause_analysis": clause_results,
            "total_sections": len(tagged_sections),
            "processing_metadata": {
                "document_type": "clause_analysis",
                "timestamp": datetime.now().isoformat(),
                "input_type": "pdf_upload",
                "document_type_provided": doc_type,
                "input_format": "pdf_upload"
            }
        })
        
    except HTTPException:
        # Re-raise HTTPException as-is (for validation errors)
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for specific Groq API errors
        if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
            )
        elif "API key" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error analyzing clauses: {error_msg}")

# ===== CLAUSE ANALYSIS (JSON) ENDPOINT =====

@router.post("/analyze-clauses-json/")
async def analyze_clauses_json(request: FlexibleClauseAnalysisRequest):
    """
    📋 AI Clause Analysis (JSON)
    
    Identify and analyze legal clauses with AI insights from tagged sections.
    
    **Features:**
    - Analyzes document structure and identifies legal clauses
    - Tags sections with headings and content
    - Provides clause analysis with AI-powered insights
    - Extracts key legal elements and provisions
    
    **Input:** JSON request with tagged sections
    
    **Use Case:** Deep analysis of legal document structure and clauses (for existing integrations)
    """
    try:
        # Convert TaggedSection models to the format expected by LLM processor
        tagged_sections = []
        for section in request.tagged_sections:
            tagged_sections.append({
                "heading": section.heading,
                "body": section.body,
                "documents": section.documents
            })
        
        # Determine input format and document type
        document_type = request.document_type or "legal_document"
        input_format = "extract_text_output" if request.total_sections is not None else "standard"
        
        # Analyze clauses using LLM
        clause_results = llm_processor.process_contract(tagged_sections)
        
        # Store in Weaviate for search
        weaviate_store.create_class_obj("LegalClauses")
        weaviate_store.add_documents("LegalClauses", tagged_sections)
        
        return JSONResponse(content={
            "clause_analysis": clause_results,
            "total_sections": len(tagged_sections),
            "processing_metadata": {
                "document_type": "clause_analysis",
                "timestamp": datetime.now().isoformat(),
                "input_type": "tagged_sections",
                "document_type_provided": document_type,
                "input_format": input_format
            }
        })
        
    except HTTPException:
        # Re-raise HTTPException as-is (for validation errors)
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for specific Groq API errors
        if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
            )
        elif "API key" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error analyzing clauses: {error_msg}")

# ===== CHRONOLOGY EXTRACTION ENDPOINT =====

@router.post("/extract-chronology/")
async def extract_chronology(
    file: UploadFile = File(...),
    document_date: Optional[date] = Query(None, description="Reference date for relative date resolution")
):
    """
    📅 AI Chronology Builder
    
    Extract and organize chronological events from PDF documents or images.
    
    **Features:**
    - Extracts chronological events from document content
    - Organizes events by date and timeline
    - Provides event descriptions and confidence scores
    - Identifies key dates, deadlines, and milestones
    
    **Input:** PDF or image file upload (multipart/form-data) - supports PDF, PNG, JPG, JPEG with optional document_date query parameter
    
    **Use Case:** Timeline analysis and event tracking in legal documents
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG, JPEG)")
        
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Tag the document
        tagged_sections = tag_documents(text)
        doc_date = document_date  # Use query parameter if provided
        document_type = "legal_document"
        
        # Combine all section bodies into a single text for chronology extraction
        combined_text = ""
        for section in tagged_sections:
            if isinstance(section, dict):
                combined_text += f"\n{section['heading']}\n{section['body']}\n"
            else:
                combined_text += f"\n{section.heading}\n{section.body}\n"
        
        # Build chronology using LLM
        document_id = f"chron_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timeline = llm_processor.build_chronology(
            text=combined_text,
            document_id=document_id,
            document_date=doc_date
        )
        
        # Check if chronology features are available
        if timeline is None:
            raise HTTPException(
                status_code=503, 
                detail="Chronology features not available. Please check your environment setup and dependencies."
            )
        
        # Store chronology events in Weaviate
        weaviate_store.create_chronology_class("ChronologyEvents")
        events_to_store = []
        for event in timeline.events:
            events_to_store.append({
                "description": event.description,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "normalized_date": event.normalized_date.isoformat() if event.normalized_date else None,
                "confidence_score": event.confidence_score,
                "source_text": event.source_text,
                "document_section": event.document_section,
                "temporal_expressions": [expr.text for expr in event.temporal_expressions],
                "metadata": event.metadata
            })
        
        weaviate_store.add_chronology_events("ChronologyEvents", events_to_store, document_id)
        
        # Prepare response with properly serialized timeline
        response_data = {
            "timeline": {
                "document_id": timeline.document_id,
                "events": [
                    {
                        "id": event.id,
                        "description": event.description,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                        "normalized_date": event.normalized_date.isoformat() if event.normalized_date else None,
                        "confidence_score": event.confidence_score,
                        "source_text": event.source_text,
                        "document_section": event.document_section,
                        "temporal_expressions": [
                            {
                                "text": expr.text,
                                "normalized_date": expr.normalized_date.isoformat() if expr.normalized_date else None,
                                "confidence": expr.confidence,
                                "is_relative": expr.is_relative,
                                "reference_point": expr.reference_point,
                                "position": expr.position
                            } for expr in event.temporal_expressions
                        ],
                        "metadata": event.metadata
                    } for event in timeline.events
                ],
                "created_at": timeline.created_at.isoformat(),
                "total_events": timeline.total_events,
                "date_range": {
                    "start": timeline.date_range.get("start").isoformat() if timeline.date_range.get("start") else None,
                    "end": timeline.date_range.get("end").isoformat() if timeline.date_range.get("end") else None
                },
                "confidence_summary": timeline.confidence_summary,
                "temporal_conflicts": timeline.temporal_conflicts
            },
            "processing_time": 0.0,  # You can add timing if needed
            "extraction_summary": {
                "total_events": timeline.total_events,
                "date_range": {
                    "start": timeline.date_range.get("start").isoformat() if timeline.date_range.get("start") else None,
                    "end": timeline.date_range.get("end").isoformat() if timeline.date_range.get("end") else None
                },
                "confidence_avg": timeline.confidence_summary.get("average", 0.0),
                "temporal_conflicts": len(timeline.temporal_conflicts)
            },
            "processing_metadata": {
                "document_type": "chronology_extraction",
                "timestamp": datetime.now().isoformat(),
                "input_type": "tagged_sections",
                "total_sections": len(tagged_sections),
                "combined_text_length": len(combined_text),
                "document_type_provided": document_type
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting chronology: {str(e)}")

# ===== DOCUMENT CHAT ENDPOINT =====

@router.post("/chat-with-document/")
async def chat_with_document(
    file: UploadFile = File(...),
    user_message: str = Query(..., description="User's question or message"),
    document_type: Optional[str] = Query(None, description="Type of document being analyzed")
):
    """
    💬 AI Document Chat Assistant
    
    Interactive Q&A with your document (PDF or image) using advanced AI.
    
    **Features:**
    - Interactive chat interface with document content
    - AI-powered question answering about document details
    - Context-aware responses based on document sections
    - Maintains conversation history and context
    
    **Input:** PDF or image file upload (multipart/form-data) - supports PDF, PNG, JPG, JPEG with user_message and document_type query parameters
    
    **Use Case:** Interactive document exploration and Q&A sessions
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG, JPEG)")
        
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Tag the document
        tagged_sections = tag_documents(text)
        user_msg = user_message
        doc_type = document_type or "legal_document"
        chat_history = []  # Start with empty chat history for PDF uploads
        
        # Chat with document using LLM
        chat_result = llm_processor.chat_with_document(
            tagged_sections=tagged_sections,
            user_message=user_msg,
            chat_history=chat_history,
            document_type=doc_type
        )
        
        # Convert chat history back to ChatMessage models
        updated_chat_history = []
        for msg in chat_result["chat_history"]:
            # Handle timestamp conversion
            timestamp = msg["timestamp"]
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
                
            updated_chat_history.append(ChatMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=timestamp
            ))
        
        return JSONResponse(content={
            "assistant_message": chat_result["assistant_message"],
            "chat_history": [msg.model_dump(mode='json') for msg in updated_chat_history],
            "processing_metadata": chat_result["processing_metadata"]
        })
        
    except Exception as e:
        error_msg = str(e)
        # Check for specific Groq API errors
        if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
            )
        elif "API key" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error in document chat: {error_msg}")

# ===== DOCUMENT SUMMARIZATION ENDPOINT =====

@router.post("/summarize-documents/")
async def summarize_documents(
    file: UploadFile = File(...),
    summary_instructions: str = Query(..., description="Custom instructions for how to summarize the documents"),
    summary_type: Optional[str] = Query("comprehensive", description="Type of summary: comprehensive, executive, bullet_points, or custom"),
    max_length: Optional[int] = Query(500, description="Maximum length of each summary in words"),
    include_key_points: Optional[bool] = Query(True, description="Whether to include key points extraction"),
    compare_documents: Optional[bool] = Query(False, description="Whether to provide comparative analysis across documents")
):
    """
    📄 AI Document Summarization
    
    Generate intelligent summaries with customizable instructions from PDF documents or images.
    
    **Features:**
    - Generates comprehensive document summaries
    - Customizable summary types (executive, bullet points, custom)
    - AI-powered content analysis and key point extraction
    - Supports multiple documents and comparison summaries
    
    **Input:** PDF or image file upload (multipart/form-data) - supports PDF, PNG, JPG, JPEG with query parameters for summarization options
    
    **Use Case:** Quick document overview and key information extraction
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG, JPEG)")
        
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Tag the document
        tagged_sections = tag_documents(text)
        
        # Create document structure
        documents_to_summarize = [{
            "document_id": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": file.filename or "Uploaded Document",
            "document_type": "legal_document",
            "tagged_sections": tagged_sections
        }]
        
        summary_instructions_text = summary_instructions
        summary_type_val = summary_type or "comprehensive"
        max_length_val = max_length or 500
        include_key_points_val = include_key_points if include_key_points is not None else True
        compare_documents_val = compare_documents if compare_documents is not None else False
        
        # Generate summaries using LLM
        summarization_result = llm_processor.summarize_documents(
            documents=documents_to_summarize,
            summary_instructions=summary_instructions_text,
            summary_type=summary_type_val,
            max_length=max_length_val,
            include_key_points=include_key_points_val,
            compare_documents=compare_documents_val
        )
        
        # Convert summaries to DocumentSummary models
        document_summaries = []
        for summary_data in summarization_result["summaries"]:
            document_summaries.append(DocumentSummary(
                document_id=summary_data["document_id"],
                title=summary_data["title"],
                summary=summary_data["summary"],
                key_points=summary_data["key_points"],
                document_type=summary_data["document_type"],
                word_count=summary_data["word_count"],
                confidence_score=summary_data["confidence_score"]
            ))
        
        return JSONResponse(content={
            "summaries": [summary.model_dump(mode='json') for summary in document_summaries],
            "comparative_analysis": summarization_result["comparative_analysis"],
            "processing_metadata": summarization_result["processing_metadata"],
            "total_documents": summarization_result["total_documents"],
            "total_processing_time": summarization_result["total_processing_time"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for specific Groq API errors
        if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
            )
        elif "API key" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error in document summarization: {error_msg}")

# ===== RISK ASSESSMENT ENDPOINT =====

@router.post("/assess-risks/")
async def assess_document_risks(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query("contract", description="Type of document being assessed"),
    assessment_focus: Optional[str] = Query("comprehensive", description="Focus area: comprehensive, financial, legal, compliance, operational"),
    include_recommendations: Optional[bool] = Query(True, description="Whether to include suggested actions"),
    risk_categories: Optional[str] = Query("financial,legal,compliance,operational", description="Comma-separated risk categories"),
    custom_instructions: Optional[str] = Query(None, description="Custom instructions for risk assessment")
):
    """
    ⚠️ AI Legal Risk Assessment
    
    Identify and analyze legal risks with actionable recommendations from PDF documents or images.
    
    **Features:**
    - Identifies potential legal risks and compliance issues
    - Provides detailed risk analysis with severity levels
    - Offers actionable recommendations for risk mitigation
    - Covers various risk categories (legal, financial, operational)
    
    **Input:** PDF or image file upload (multipart/form-data) - supports PDF, PNG, JPG, JPEG with query parameters for assessment options
    
    **Use Case:** Comprehensive risk analysis and compliance checking
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG, JPEG)")
        
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Tag the document
        tagged_sections = tag_documents(text)
        
        # Parse query parameters
        doc_type = document_type or "contract"
        assessment_focus_val = assessment_focus or "comprehensive"
        include_recommendations_val = include_recommendations if include_recommendations is not None else True
        risk_categories_list = [cat.strip() for cat in risk_categories.split(",")] if risk_categories else ["financial", "legal", "compliance", "operational"]
        custom_instructions_text = custom_instructions
        
        # Generate risk assessment using LLM
        risk_assessment_result = llm_processor.assess_document_risks(
            tagged_sections=tagged_sections,
            document_type=doc_type,
            assessment_focus=assessment_focus_val,
            risk_categories=risk_categories_list,
            custom_instructions=custom_instructions_text,
            include_recommendations=include_recommendations_val
        )
        
        # Convert risks to RiskItem models
        risk_items = []
        for risk_data in risk_assessment_result["risks"]:
            risk_items.append(RiskItem(
                clause_type=risk_data["clause_type"],
                risk_description=risk_data["risk_description"],
                severity=risk_data["severity"],
                severity_score=risk_data["severity_score"],
                suggested_action=risk_data["suggested_action"],
                legal_basis=risk_data["legal_basis"],
                impact_area=risk_data["impact_area"],
                confidence_score=risk_data["confidence_score"]
            ))
        
        return JSONResponse(content={
            "risks": [risk.model_dump(mode='json') for risk in risk_items],
            "risk_summary": risk_assessment_result["risk_summary"],
            "overall_risk_level": risk_assessment_result["overall_risk_level"],
            "processing_metadata": risk_assessment_result["processing_metadata"],
            "assessment_timestamp": risk_assessment_result["assessment_timestamp"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for specific Groq API errors
        if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
            )
        elif "API key" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error in risk assessment: {error_msg}")

# ===== DOCUMENT CLASSIFICATION ENDPOINT =====

@router.post("/classify-document/")
async def classify_document(
    file: UploadFile = File(...),
    document_type_hint: Optional[str] = Query(None, description="Optional hint about the document type"),
    classification_focus: Optional[str] = Query(None, description="Optional focus area for classification")
):
    """
    🏷️ AI Document Classification
    
    Classify documents by type, subject, and importance level from PDF documents or images.
    
    **Features:**
    - Classifies documents by type (Contract, NDA, Employment Agreement, etc.)
    - Identifies subject/focus areas (Employment Terms, Confidentiality, etc.)
    - Assesses importance/priority levels (High, Medium, Low)
    - Provides confidence scores and reasoning for classifications
    
    **Input:** PDF or image file upload (multipart/form-data) - supports PDF, PNG, JPG, JPEG with optional query parameters
    
    **Use Case:** Document categorization and priority management
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be a PDF or image (PNG, JPG, JPEG)")
        
        # Extract text from file (PDF or image)
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Tag the document
        tagged_sections = tag_documents(text)
        doc_type_hint = document_type_hint
        classification_focus_val = classification_focus
        
        # Generate document classification using LLM
        classification_result = llm_processor.classify_document(
            tagged_sections=tagged_sections,
            document_type_hint=doc_type_hint,
            classification_focus=classification_focus_val
        )
        
        # Create DocumentClassification model
        document_classification = DocumentClassification(
            document_type=classification_result["document_type"],
            subject=classification_result["subject"],
            importance=classification_result["importance"],
            reasoning=classification_result["reasoning"],
            confidence_scores=classification_result["confidence_scores"],
            processing_metadata=classification_result["processing_metadata"]
        )
        
        return JSONResponse(content={
            "classification": document_classification.model_dump(mode='json'),
            "processing_metadata": classification_result["processing_metadata"],
            "classification_timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for specific Groq API errors
        if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
            )
        elif "API key" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error in document classification: {error_msg}")

# ===== DOCUMENT COMPARISON ENDPOINT =====

# @router.post("/compare-documents/")
# async def compare_documents(
#     file_a: UploadFile = File(..., description="First PDF document to compare"),
#     file_b: UploadFile = File(..., description="Second PDF document to compare"),
#     comparison_focus: str = Query(default="comprehensive", description="Focus area: comprehensive, clauses_only, obligations_only, timelines_only"),
#     include_word_level: bool = Query(default=False, description="Include word-level changes for critical clauses"),
#     risk_threshold: str = Query(default="Medium", description="Minimum risk level to highlight: Low, Medium, High, Critical"),
#     document_a_title: str = Query(default=None, description="Title for the first document"),
#     document_b_title: str = Query(default=None, description="Title for the second document")
# ):
#     """
#     🔄 AI Document Comparison
    
#     Compare two PDF documents and identify differences in clauses, obligations, and timelines.
    
#     **Features:**
#     - Compares two PDF documents side-by-side
#     - Identifies differences in clauses, obligations, and timelines
#     - Provides risk assessment for changes
#     - Automatically detects identical documents
#     - Supports word-level change analysis for critical clauses
    
#     **Use Case:** Document version comparison and change analysis
#     """
#     try:
#         # Validate file types
#         if not file_a.filename.lower().endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="First file must be a PDF")
#         if not file_b.filename.lower().endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="Second file must be a PDF")
        
#         # Extract text from both PDFs
#         text_a = await extract_text_from_pdf_first_page(file_a)
#         if not text_a.strip():
#             raise HTTPException(status_code=400, detail="Could not extract text from first PDF")
        
#         text_b = await extract_text_from_pdf_first_page(file_b)
#         if not text_b.strip():
#             raise HTTPException(status_code=400, detail="Could not extract text from second PDF")
        
#         # Check if documents are identical
#         if text_a.strip() == text_b.strip():
#             # Documents are identical - return no differences result
#             return JSONResponse(content={
#                 "comparison": {
#                     "comparison_summary": "The two documents are identical with no differences found.",
#                     "clause_differences": [],
#                     "obligation_differences": [],
#                     "timeline_differences": [],
#                     "high_risk_changes": [],
#                     "overall_risk_assessment": "No Risk - Documents are identical",
#                     "processing_metadata": {
#                         "comparison_focus": comparison_focus,
#                         "include_word_level": include_word_level,
#                         "risk_threshold": risk_threshold,
#                         "total_documents_compared": 2,
#                         "total_differences": 0,
#                         "high_risk_count": 0,
#                         "processing_timestamp": datetime.now().isoformat(),
#                         "processing_time": 0.0,
#                         "documents_identical": True
#                     }
#                 },
#                 "processing_metadata": {
#                     "comparison_focus": comparison_focus,
#                     "include_word_level": include_word_level,
#                     "risk_threshold": risk_threshold,
#                     "total_documents_compared": 2,
#                     "total_differences": 0,
#                     "high_risk_count": 0,
#                     "processing_timestamp": datetime.now().isoformat(),
#                     "processing_time": 0.0,
#                     "documents_identical": True
#                 },
#                 "comparison_timestamp": datetime.now().isoformat(),
#                 "document_info": {
#                     "document_a": {
#                         "filename": file_a.filename,
#                         "title": document_a_title or file_a.filename or "Document A",
#                         "sections_count": 0
#                     },
#                     "document_b": {
#                         "filename": file_b.filename,
#                         "title": document_b_title or file_b.filename or "Document B", 
#                         "sections_count": 0
#                     }
#                 }
#             })
        
#         # Tag documents
#         tagged_sections_a = tag_documents(text_a)
#         tagged_sections_b = tag_documents(text_b)
        
#         # Prepare documents for comparison
#         documents_to_compare = [
#             {
#                 "document_id": "document_a",
#                 "title": document_a_title or file_a.filename or "Document A",
#                 "document_type": "legal_document",
#                 "tagged_sections": tagged_sections_a
#             },
#             {
#                 "document_id": "document_b", 
#                 "title": document_b_title or file_b.filename or "Document B",
#                 "document_type": "legal_document",
#                 "tagged_sections": tagged_sections_b
#             }
#         ]
        
#         # Generate document comparison using LLM
#         comparison_result = llm_processor.compare_documents(
#             documents=documents_to_compare,
#             comparison_focus=comparison_focus,
#             include_word_level=include_word_level,
#             risk_threshold=risk_threshold
#         )
        
#         # Convert differences to Pydantic models
#         clause_differences = []
#         for diff in comparison_result.get("clause_differences", []):
#             clause_differences.append(ClauseDifference(
#                 clause_title=diff["clause_title"],
#                 document_a_content=diff["document_a_content"],
#                 document_b_content=diff["document_b_content"],
#                 change_type=diff["change_type"],
#                 risk_level=diff["risk_level"],
#                 reasoning=diff["reasoning"]
#             ))
        
#         obligation_differences = []
#         for diff in comparison_result.get("obligation_differences", []):
#             obligation_differences.append(ObligationDifference(
#                 party=diff["party"],
#                 document_a_obligation=diff["document_a_obligation"],
#                 document_b_obligation=diff["document_b_obligation"],
#                 change_type=diff["change_type"],
#                 risk_level=diff["risk_level"],
#                 reasoning=diff["reasoning"]
#             ))
        
#         timeline_differences = []
#         for diff in comparison_result.get("timeline_differences", []):
#             timeline_differences.append(TimelineDifference(
#                 event=diff["event"],
#                 document_a_timeline=diff["document_a_timeline"],
#                 document_b_timeline=diff["document_b_timeline"],
#                 change_type=diff["change_type"],
#                 risk_level=diff["risk_level"],
#                 reasoning=diff["reasoning"]
#             ))
        
#         # Create DocumentComparison model
#         document_comparison = DocumentComparison(
#             comparison_summary=comparison_result["comparison_summary"],
#             clause_differences=clause_differences,
#             obligation_differences=obligation_differences,
#             timeline_differences=timeline_differences,
#             high_risk_changes=comparison_result.get("high_risk_changes", []),
#             overall_risk_assessment=comparison_result["overall_risk_assessment"],
#             processing_metadata=comparison_result["processing_metadata"]
#         )
        
#         return JSONResponse(content={
#             "comparison": document_comparison.model_dump(mode='json'),
#             "processing_metadata": comparison_result["processing_metadata"],
#             "comparison_timestamp": datetime.now().isoformat(),
#             "document_info": {
#                 "document_a": {
#                     "filename": file_a.filename,
#                     "title": document_a_title or file_a.filename or "Document A",
#                     "sections_count": len(tagged_sections_a)
#                 },
#                 "document_b": {
#                     "filename": file_b.filename,
#                     "title": document_b_title or file_b.filename or "Document B", 
#                     "sections_count": len(tagged_sections_b)
#                 }
#             }
#         })
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         error_msg = str(e)
#         # Check for specific Groq API errors
#         if "Rate limit reached" in error_msg or "rate_limit_exceeded" in error_msg:
#             raise HTTPException(
#                 status_code=429, 
#                 detail="API rate limit exceeded. Please wait a few minutes and try again, or upgrade your Groq API plan."
#             )
#         elif "API key" in error_msg or "authentication" in error_msg.lower():
#             raise HTTPException(
#                 status_code=401,
#                 detail="API authentication failed. Please check your GROQ_API_KEY in the .env file."
#             )
#         else:
#             raise HTTPException(status_code=500, detail=f"Error in document comparison: {error_msg}")