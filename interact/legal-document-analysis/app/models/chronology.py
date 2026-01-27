from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum

class EventType(str, Enum):
    """Enumeration of event types for classification"""
    LEGAL_ACTION = "legal_action"
    CONTRACT_EVENT = "contract_event"
    PAYMENT = "payment"
    DEADLINE = "deadline"
    MEETING = "meeting"
    COMMUNICATION = "communication"
    DOCUMENT_CREATION = "document_creation"
    DECISION = "decision"
    NOTIFICATION = "notification"
    OTHER = "other"

class TemporalExpression(BaseModel):
    """Represents a temporal expression found in text"""
    text: str = Field(..., description="Original text of the temporal expression")
    normalized_date: Optional[date] = Field(None, description="Normalized date in ISO format")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for the date extraction")
    is_relative: bool = Field(False, description="Whether this is a relative date expression")
    reference_point: Optional[str] = Field(None, description="Reference point for relative dates")
    position: int = Field(..., description="Character position in the original text")

class Event(BaseModel):
    """Represents an extracted event with temporal information"""
    id: str = Field(..., description="Unique identifier for the event")
    description: str = Field(..., description="Description of the event")
    event_type: EventType = Field(..., description="Type/category of the event")
    timestamp: Optional[datetime] = Field(None, description="Precise timestamp if available")
    normalized_date: Optional[date] = Field(None, description="Normalized date")
    temporal_expressions: List[TemporalExpression] = Field(default_factory=list, description="Temporal expressions found in the event text")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence score for the event extraction")
    source_text: str = Field(..., description="Original text from which the event was extracted")
    document_section: str = Field(..., description="Section of the document where the event was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the event")

class Timeline(BaseModel):
    """Represents a complete timeline of events"""
    document_id: str = Field(..., description="Identifier of the source document")
    events: List[Event] = Field(default_factory=list, description="List of events in chronological order")
    created_at: datetime = Field(default_factory=datetime.now, description="When the timeline was created")
    total_events: int = Field(0, description="Total number of events in the timeline")
    date_range: Dict[str, Optional[date]] = Field(default_factory=dict, description="Start and end dates of the timeline")
    confidence_summary: Dict[str, float] = Field(default_factory=dict, description="Summary statistics of confidence scores")
    temporal_conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="List of detected temporal conflicts")

# ===== COMMENTED OUT - UNUSED MODELS =====
# These models were defined but are not actually used in the current implementation
# The system uses ChronologyFromSectionsRequest instead of ChronologyRequest

# class ChronologyRequest(BaseModel):
#     """Request model for chronology extraction"""
#     document_text: str = Field(..., description="Text content to analyze")
#     document_date: Optional[date] = Field(None, description="Reference date for relative date resolution")
#     document_type: Optional[str] = Field(None, description="Type of document being analyzed")
#     include_metadata: bool = Field(True, description="Whether to include detailed metadata")

# class ChronologyResponse(BaseModel):
#     """Response model for chronology extraction"""
#     timeline: Timeline = Field(..., description="Generated timeline")
#     processing_time: float = Field(..., description="Time taken to process the document")
#     extraction_summary: Dict[str, Any] = Field(..., description="Summary of the extraction process")

class TaggedSection(BaseModel):
    """Model for a tagged document section"""
    heading: str = Field(..., description="Section heading")
    body: str = Field(..., description="Section body content")
    documents: List[str] = Field(default_factory=list, description="Chunked documents within the section")

class ClauseAnalysisRequest(BaseModel):
    """Request model for clause analysis from tagged sections"""
    tagged_sections: List[TaggedSection] = Field(..., description="Tagged sections from text extraction")
    document_type: Optional[str] = Field(None, description="Type of document being analyzed")

class FlexibleClauseAnalysisRequest(BaseModel):
    """Flexible request model that accepts both extract-text output and standard format"""
    tagged_sections: List[TaggedSection] = Field(..., description="Tagged sections from text extraction")
    document_type: Optional[str] = Field(None, description="Type of document being analyzed")
    # Optional fields from extract-text output
    total_sections: Optional[int] = Field(None, description="Total number of sections (from extract-text output)")
    raw_text_length: Optional[int] = Field(None, description="Raw text length (from extract-text output)")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata (from extract-text output)")

class ChronologyFromSectionsRequest(BaseModel):
    """Request model for chronology extraction from tagged sections"""
    tagged_sections: List[TaggedSection] = Field(..., description="Tagged sections from text extraction")
    document_date: Optional[date] = Field(None, description="Reference date for relative date resolution")
    document_type: Optional[str] = Field(None, description="Type of document being analyzed")

# ===== CHAT MODELS =====

class ChatMessage(BaseModel):
    """Model for a chat message"""
    role: str = Field(..., description="Role of the message sender (user, assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")

class DocumentChatRequest(BaseModel):
    """Request model for document chat"""
    tagged_sections: List[TaggedSection] = Field(..., description="Tagged sections from text extraction")
    user_message: str = Field(..., description="User's question or message")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Previous chat messages for context")
    document_type: Optional[str] = Field(None, description="Type of document being analyzed")

class DocumentChatResponse(BaseModel):
    """Response model for document chat"""
    assistant_message: str = Field(..., description="Assistant's response")
    chat_history: List[ChatMessage] = Field(..., description="Updated chat history including the new messages")
    processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the chat processing")

# ===== DOCUMENT SUMMARIZATION MODELS =====

class DocumentSummary(BaseModel):
    """Model for a single document summary"""
    document_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title or identifier")
    summary: str = Field(..., description="Generated summary")
    key_points: List[str] = Field(default_factory=list, description="Key points extracted from the document")
    document_type: str = Field(..., description="Type of document")
    word_count: int = Field(0, description="Approximate word count of the document")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for the summary")

class DocumentSummarizationRequest(BaseModel):
    """Request model for document summarization"""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to summarize (each with tagged_sections)")
    summary_instructions: str = Field(..., description="Custom instructions for how to summarize the documents")
    summary_type: str = Field(default="comprehensive", description="Type of summary: comprehensive, executive, bullet_points, or custom")
    max_length: int = Field(default=500, description="Maximum length of each summary in words")
    include_key_points: bool = Field(default=True, description="Whether to include key points extraction")
    compare_documents: bool = Field(default=False, description="Whether to provide comparative analysis across documents")

class DocumentSummarizationResponse(BaseModel):
    """Response model for document summarization"""
    summaries: List[DocumentSummary] = Field(..., description="List of document summaries")
    comparative_analysis: Optional[str] = Field(None, description="Comparative analysis if compare_documents is True")
    processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the summarization processing")
    total_documents: int = Field(..., description="Total number of documents processed")
    total_processing_time: float = Field(..., description="Total time taken to process all documents")

# ===== RISK ASSESSMENT MODELS =====

class RiskItem(BaseModel):
    """Model for a single risk item"""
    clause_type: str = Field(..., description="Type of clause or section where risk was identified")
    risk_description: str = Field(..., description="Description of the identified risk")
    severity: str = Field(..., description="Risk severity level: Low, Medium, High, Critical")
    severity_score: int = Field(..., ge=1, le=5, description="Numeric severity score (1=Low, 5=Critical)")
    suggested_action: str = Field(..., description="Recommended action to mitigate the risk")
    legal_basis: str = Field(..., description="Legal basis or reasoning for the risk assessment")
    impact_area: str = Field(..., description="Area of impact: Financial, Legal, Operational, Compliance, etc.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the risk assessment")

class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment"""
    tagged_sections: List[TaggedSection] = Field(..., description="Tagged sections from text extraction")
    document_type: str = Field(default="contract", description="Type of document being assessed")
    assessment_focus: str = Field(default="comprehensive", description="Focus area: comprehensive, financial, legal, compliance, operational")
    include_recommendations: bool = Field(default=True, description="Whether to include suggested actions")
    risk_categories: List[str] = Field(default_factory=lambda: ["financial", "legal", "compliance", "operational"], description="Risk categories to assess")
    custom_instructions: Optional[str] = Field(None, description="Custom instructions for risk assessment")

class RiskAssessmentResponse(BaseModel):
    """Response model for risk assessment"""
    risks: List[RiskItem] = Field(..., description="List of identified risks")
    risk_summary: Dict[str, Any] = Field(..., description="Summary statistics of identified risks")
    overall_risk_level: str = Field(..., description="Overall risk level of the document")
    processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the risk assessment processing")
    assessment_timestamp: datetime = Field(default_factory=datetime.now, description="When the assessment was performed")

# ===== DOCUMENT CLASSIFICATION MODELS =====

class DocumentClassification(BaseModel):
    """Model for document classification results"""
    document_type: str = Field(..., description="Type of document (e.g., Contract, NDA, Employment Agreement)")
    subject: str = Field(..., description="Subject/Focus area (e.g., Employment Terms, Confidentiality, Payment Terms)")
    importance: str = Field(..., description="Importance level: High, Medium, or Low")
    reasoning: Dict[str, str] = Field(..., description="Reasoning for each classification category")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each classification")
    processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the classification process")

class DocumentClassificationRequest(BaseModel):
    """Request model for document classification"""
    tagged_sections: List[TaggedSection] = Field(..., description="Tagged sections from text extraction")
    document_type_hint: Optional[str] = Field(None, description="Optional hint about the document type")
    classification_focus: Optional[str] = Field(None, description="Optional focus area for classification")

class DocumentClassificationResponse(BaseModel):
    """Response model for document classification"""
    classification: DocumentClassification = Field(..., description="Document classification results")
    processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the classification processing")
    classification_timestamp: datetime = Field(default_factory=datetime.now, description="When the classification was performed")

# ===== DOCUMENT COMPARISON MODELS =====
# COMMENTED OUT - Document comparison functionality disabled

# class ClauseDifference(BaseModel):
#     """Model for clause differences between documents"""
#     clause_title: str = Field(..., description="Title or identifier of the clause")
#     document_a_content: str = Field(..., description="Content of the clause in document A")
#     document_b_content: str = Field(..., description="Content of the clause in document B")
#     change_type: str = Field(..., description="Type of change: Added, Deleted, Modified, or Unchanged")
#     risk_level: str = Field(default="Low", description="Risk level of the change: Low, Medium, High, Critical")
#     reasoning: str = Field(..., description="Explanation of the difference and its implications")

# class ObligationDifference(BaseModel):
#     """Model for obligation differences between documents"""
#     party: str = Field(..., description="Party whose obligation changed")
#     document_a_obligation: str = Field(..., description="Obligation in document A")
#     document_b_obligation: str = Field(..., description="Obligation in document B")
#     change_type: str = Field(..., description="Type of change: Added, Deleted, Modified, or Unchanged")
#     risk_level: str = Field(default="Low", description="Risk level of the change: Low, Medium, High, Critical")
#     reasoning: str = Field(..., description="Explanation of the obligation difference")

# class TimelineDifference(BaseModel):
#     """Model for timeline differences between documents"""
#     event: str = Field(..., description="Event or milestone name")
#     document_a_timeline: str = Field(..., description="Timeline in document A")
#     document_b_timeline: str = Field(..., description="Timeline in document B")
#     change_type: str = Field(..., description="Type of change: Added, Deleted, Modified, or Unchanged")
#     risk_level: str = Field(default="Low", description="Risk level of the change: Low, Medium, High, Critical")
#     reasoning: str = Field(..., description="Explanation of the timeline difference")

# class DocumentComparison(BaseModel):
#     """Model for document comparison results"""
#     comparison_summary: str = Field(..., description="Summary of key differences between documents")
#     clause_differences: List[ClauseDifference] = Field(default_factory=list, description="List of clause differences")
#     obligation_differences: List[ObligationDifference] = Field(default_factory=list, description="List of obligation differences")
#     timeline_differences: List[TimelineDifference] = Field(default_factory=list, description="List of timeline differences")
#     high_risk_changes: List[Dict[str, Any]] = Field(default_factory=list, description="High-risk changes that need attention")
#     overall_risk_assessment: str = Field(..., description="Overall risk assessment of the changes")
#     processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the comparison process")

# class DocumentComparisonRequest(BaseModel):
#     """Request model for document comparison"""
#     documents: List[Dict[str, Any]] = Field(..., description="List of documents to compare (each with tagged_sections)")
#     comparison_focus: str = Field(default="comprehensive", description="Focus area: comprehensive, clauses_only, obligations_only, timelines_only")
#     include_word_level: bool = Field(default=False, description="Include word-level changes for critical clauses")
#     risk_threshold: str = Field(default="Medium", description="Minimum risk level to highlight: Low, Medium, High, Critical")

# class DocumentComparisonResponse(BaseModel):
#     """Response model for document comparison"""
#     comparison: DocumentComparison = Field(..., description="Document comparison results")
#     processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the comparison processing")
#     comparison_timestamp: datetime = Field(default_factory=datetime.now, description="When the comparison was performed")