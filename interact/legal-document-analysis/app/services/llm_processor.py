import re
import os

# CRITICAL: Set NO_PROXY before importing langchain_ollama (which uses httpx)
os.environ['NO_PROXY'] = '192.168.0.56'

from datetime import datetime, date
from typing import List, Dict, Optional, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from utils.env_util import EnvironmentVariables
try:
    from app.models.chronology import Event, EventType, Timeline
    from app.services.temporal_processor import TemporalProcessor
    from app.services.event_extractor import EventExtractor
    from app.services.chronology_builder import ChronologyBuilder
    CHRONOLOGY_AVAILABLE = True
except ImportError as e:
    print(f"Chronology components not available: {e}")
    CHRONOLOGY_AVAILABLE = False


class LLMProcessor:
    def __init__(self) -> None:
        # Load env vars
        EnvironmentVariables()
        
        # Environment-aware LLM initialization
        app_env = EnvironmentVariables.APP_ENV
        
        if app_env == "production":
            print(f"🚀 [Interact] Initializing Groq LLM (Production)")
            groq_api_key = EnvironmentVariables.GROQ_API_KEY
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY missing for production environment")
            
            self.llm = ChatGroq(
                model_name=EnvironmentVariables.GROQ_MODEL,
                groq_api_key=groq_api_key,
                temperature=0
            )
        else:
            base_url = EnvironmentVariables.OLLAMA_BASE_URL
            model = EnvironmentVariables.OLLAMA_MODEL
            print(f"🏠 [Interact] Connecting to Ollama at: {base_url}")
            
            self.llm = ChatOllama(
                model=model,
                base_url=base_url,
                temperature=0,
                timeout=120
            )
        
        # Initialize chronology services if available
        if CHRONOLOGY_AVAILABLE:
            self.temporal_processor = TemporalProcessor()
            self.event_extractor = EventExtractor()
            self.chronology_builder = ChronologyBuilder()
        else:
            self.temporal_processor = None
            self.event_extractor = None
            self.chronology_builder = None

    def get_classification_chain(self) -> LLMChain:
        """Classify text into predefined clause categories."""
        CLASSIFY_PROMPT = """
        You are a legal clause classification expert. 
        Classify the following text into one of these categories only:
        - Confidentiality
        - Termination
        - Indemnity
        - Arbitration
        - Payment Terms
        - Governing Law
        - Liability
        - Force Majeure
        - Intellectual Property
        - Data Protection
        - Compliance
        - Dispute Resolution
        - Performance Obligations
        - Other (if none of the above match)

        Text: {context}

        Answer with only the category name.
        """
        prompt = PromptTemplate(template=CLASSIFY_PROMPT, input_variables=["context"])
        return LLMChain(llm=self.llm, prompt=prompt)

    def get_summarization_chain(self) -> LLMChain:
        """Summarize text in plain-English, focusing on the detected clause."""
        SUMMARIZE_PROMPT = """
        You are a legal document summarization expert.
        Summarize the following {clause} clause into a short, clear paragraph 
        that a non-lawyer can easily understand.

        Focus on the key points, obligations, rights, and important details 
        mentioned in the text. Be concise but comprehensive.

        Context: {context}
        Clause Type: {clause}

        Provide a clear summary of what this clause means and its implications:"""
        prompt = PromptTemplate(
            template=SUMMARIZE_PROMPT, input_variables=["context", "clause"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def classify_and_summarize(self, text: str):
        """End-to-end: classify the clause type, then summarize it."""
        try:
            # Step 1: Classify
            classify_chain = self.get_classification_chain()
            clause = classify_chain.run({"context": text}).strip()
            
            # Clean up clause classification
            if not clause or clause.lower() in ["none", "n/a", "not found", "clause not found"]:
                clause = "Other"

            # Step 2: Summarize
            summarize_chain = self.get_summarization_chain()
            summary = summarize_chain.run({"context": text, "clause": clause}).strip()

            # Step 3: Clean up summary
            summary = re.sub(r"<.*?>", "", summary).replace("\n", " ")
            
            # Ensure we have a meaningful summary
            if not summary or summary.lower() in ["none", "n/a", "not found", "clause not found"]:
                summary = f"This section contains {clause.lower()} related content that requires legal review."

            return {"clause": clause, "summary": summary}
            
        except Exception as e:
            print(f"Error in classify_and_summarize: {e}")
            return {
                "clause": "Other", 
                "summary": f"Error processing this section: {str(e)}"
            }

    def process_contract(self, tagged_sections: list):
        """
        Takes structured sections (from tagging.py),
        classifies + summarizes each chunk,
        and returns a structured report.
        """
        results = []

        for section in tagged_sections:
            section_result = {
                "heading": section["heading"],
                "clauses": []
            }

            # Process documents if they exist
            if section.get("documents") and len(section["documents"]) > 0:
                for doc in section["documents"]:
                    if doc.strip():  # Only process non-empty documents
                        output = self.classify_and_summarize(doc)
                        section_result["clauses"].append(output)
            
            # If no documents or empty documents, try to process the body content
            if not section_result["clauses"] and section.get("body") and section["body"].strip():
                output = self.classify_and_summarize(section["body"])
                section_result["clauses"].append(output)
            
            # If still no clauses found, add a default message
            if not section_result["clauses"]:
                section_result["clauses"].append({
                    "clause": "No content found",
                    "summary": "This section appears to be empty or contains no analyzable content."
                })

            results.append(section_result)

        return results

    # ===== DOCUMENT CHAT METHODS =====

    def get_document_chat_chain(self) -> LLMChain:
        """Create a chain for document-based chat interactions."""
        DOCUMENT_CHAT_PROMPT = """
        You are an expert legal document assistant. You have access to a legal document that has been processed and structured into sections.
        
        Document Context:
        {document_context}
        
        Chat History:
        {chat_history}
        
        User Question: {user_question}
        
        Instructions:
        1. Answer the user's question based on the document content provided
        2. If the question is not related to the document, politely redirect to document-related topics
        3. Be specific and cite relevant sections when possible
        4. If you're unsure about something, say so rather than guessing
        5. Keep responses concise but informative
        6. Use plain English to explain legal concepts
        
        Response:"""
        
        prompt = PromptTemplate(
            template=DOCUMENT_CHAT_PROMPT,
            input_variables=["document_context", "chat_history", "user_question"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def chat_with_document(self, tagged_sections: List[Dict[str, Any]], 
                          user_message: str, chat_history: List[Dict[str, Any]] = None,
                          document_type: str = "legal_document") -> Dict[str, Any]:
        """
        Chat with the document using the provided context.
        
        Args:
            tagged_sections: Processed document sections
            user_message: User's question or message
            chat_history: Previous chat messages for context
            document_type: Type of document being analyzed
            
        Returns:
            Dictionary containing the assistant's response and updated chat history
        """
        if chat_history is None:
            chat_history = []
        
        # Prepare document context
        document_context = self._prepare_document_context(tagged_sections)
        
        # Prepare chat history context
        chat_history_context = self._prepare_chat_history_context(chat_history)
        
        # Get chat chain and generate response
        chat_chain = self.get_document_chat_chain()
        
        try:
            response = chat_chain.invoke({
                "document_context": document_context,
                "chat_history": chat_history_context,
                "user_question": user_message
            })
            
            # Handle different response formats
            if hasattr(response, 'content'):
                assistant_message = response.content.strip()
            elif isinstance(response, dict) and 'text' in response:
                assistant_message = response['text'].strip()
            else:
                assistant_message = str(response).strip()
            
            # Create updated chat history
            updated_history = chat_history.copy()
            updated_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            updated_history.append({
                "role": "assistant", 
                "content": assistant_message,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "assistant_message": assistant_message,
                "chat_history": updated_history,
                "processing_metadata": {
                    "document_type": document_type,
                    "total_sections": len(tagged_sections),
                    "document_context_length": len(document_context),
                    "chat_history_length": len(chat_history),
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            return {
                "assistant_message": error_message,
                "chat_history": chat_history,
                "processing_metadata": {
                    "document_type": document_type,
                    "error": str(e),
                    "processing_timestamp": datetime.now().isoformat()
                }
            }

    def _prepare_document_context(self, tagged_sections: List[Dict[str, Any]]) -> str:
        """Prepare document context for chat."""
        context_parts = []
        
        for i, section in enumerate(tagged_sections, 1):
            heading = section.get("heading", f"Section {i}")
            body = section.get("body", "")
            documents = section.get("documents", [])
            
            context_parts.append(f"Section {i}: {heading}")
            if body:
                context_parts.append(f"Content: {body}")
            
            if documents:
                context_parts.append("Key Points:")
                for j, doc in enumerate(documents[:3], 1):  # Limit to first 3 chunks
                    context_parts.append(f"  {j}. {doc}")
            
            context_parts.append("")  # Add spacing between sections
        
        return "\n".join(context_parts)

    def _prepare_chat_history_context(self, chat_history: List[Dict[str, Any]]) -> str:
        """Prepare chat history context."""
        if not chat_history:
            return "No previous conversation."
        
        history_parts = []
        for msg in chat_history[-5:]:  # Only include last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_parts.append(f"{role.title()}: {content}")
        
        return "\n".join(history_parts)

    # ===== DOCUMENT SUMMARIZATION METHODS =====

    def get_document_summarization_chain(self) -> LLMChain:
        """Create a chain for document summarization."""
        SUMMARIZATION_PROMPT = """
        You are an expert document summarization assistant. You will create high-quality summaries based on the provided instructions.
        
        Document Content:
        {document_content}
        
        Summary Instructions: {summary_instructions}
        Summary Type: {summary_type}
        Maximum Length: {max_length} words
        Include Key Points: {include_key_points}
        
        Please create a summary that:
        1. Follows the specific instructions provided
        2. Captures the most important information
        3. Is clear and concise
        4. Uses plain English when explaining complex concepts
        5. Maintains accuracy to the original content
        
        Format your response as JSON:
        {{
            "summary": "Your comprehensive summary here",
            "key_points": ["Key point 1", "Key point 2", "Key point 3"],
            "confidence_score": 0.9,
            "word_count": 150
        }}
        """
        
        prompt = PromptTemplate(
            template=SUMMARIZATION_PROMPT,
            input_variables=["document_content", "summary_instructions", "summary_type", "max_length", "include_key_points"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def get_comparative_analysis_chain(self) -> LLMChain:
        """Create a chain for comparative analysis of multiple documents."""
        COMPARATIVE_PROMPT = """
        You are an expert at analyzing and comparing multiple documents. You will provide a comprehensive comparative analysis.
        
        Document Summaries:
        {document_summaries}
        
        Analysis Instructions: {analysis_instructions}
        
        Please provide a comparative analysis that:
        1. Identifies similarities and differences between documents
        2. Highlights key themes and patterns
        3. Points out potential conflicts or contradictions
        4. Provides insights about the overall document set
        5. Suggests areas that may need attention
        
        Response:"""
        
        prompt = PromptTemplate(
            template=COMPARATIVE_PROMPT,
            input_variables=["document_summaries", "analysis_instructions"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def summarize_documents(self, documents: List[Dict[str, Any]], 
                           summary_instructions: str,
                           summary_type: str = "comprehensive",
                           max_length: int = 500,
                           include_key_points: bool = True,
                           compare_documents: bool = False) -> Dict[str, Any]:
        """
        Summarize multiple documents with customizable instructions.
        
        Args:
            documents: List of documents with tagged_sections
            summary_instructions: Custom instructions for summarization
            summary_type: Type of summary (comprehensive, executive, bullet_points, custom)
            max_length: Maximum length of each summary in words
            include_key_points: Whether to include key points extraction
            compare_documents: Whether to provide comparative analysis
            
        Returns:
            Dictionary containing summaries and metadata
        """
        import time
        start_time = time.time()
        
        summaries = []
        summarization_chain = self.get_document_summarization_chain()
        
        for i, doc in enumerate(documents):
            try:
                # Prepare document content
                document_content = self._prepare_document_context(doc.get("tagged_sections", []))
                document_id = doc.get("document_id", f"doc_{i+1}")
                document_title = doc.get("title", f"Document {i+1}")
                document_type = doc.get("document_type", "legal_document")
                
                # Generate summary
                response = summarization_chain.invoke({
                    "document_content": document_content,
                    "summary_instructions": summary_instructions,
                    "summary_type": summary_type,
                    "max_length": str(max_length),
                    "include_key_points": str(include_key_points)
                })
                
                # Parse response
                if hasattr(response, 'content'):
                    response_text = response.content.strip()
                elif isinstance(response, dict) and 'text' in response:
                    response_text = response['text'].strip()
                else:
                    response_text = str(response).strip()
                
                # Try to parse JSON response
                try:
                    import json
                    import re
                    # Find JSON in response
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        summary_data = json.loads(json_match.group())
                    else:
                        # Fallback if no JSON found
                        summary_data = {
                            "summary": response_text,
                            "key_points": [],
                            "confidence_score": 0.8,
                            "word_count": len(response_text.split())
                        }
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    summary_data = {
                        "summary": response_text,
                        "key_points": [],
                        "confidence_score": 0.7,
                        "word_count": len(response_text.split())
                    }
                
                # Create document summary
                doc_summary = {
                    "document_id": document_id,
                    "title": document_title,
                    "summary": summary_data.get("summary", ""),
                    "key_points": summary_data.get("key_points", []),
                    "document_type": document_type,
                    "word_count": summary_data.get("word_count", 0),
                    "confidence_score": summary_data.get("confidence_score", 0.8)
                }
                
                summaries.append(doc_summary)
                
            except Exception as e:
                print(f"Error summarizing document {i+1}: {e}")
                # Create error summary
                summaries.append({
                    "document_id": f"doc_{i+1}",
                    "title": f"Document {i+1}",
                    "summary": f"Error generating summary: {str(e)}",
                    "key_points": [],
                    "document_type": "error",
                    "word_count": 0,
                    "confidence_score": 0.0
                })
        
        # Generate comparative analysis if requested
        comparative_analysis = None
        if compare_documents and len(summaries) > 1:
            try:
                comparative_chain = self.get_comparative_analysis_chain()
                
                # Prepare summaries for comparison
                summary_texts = []
                for summary in summaries:
                    summary_texts.append(f"Document: {summary['title']}\nSummary: {summary['summary']}")
                
                comparative_response = comparative_chain.invoke({
                    "document_summaries": "\n\n".join(summary_texts),
                    "analysis_instructions": summary_instructions
                })
                
                if hasattr(comparative_response, 'content'):
                    comparative_analysis = comparative_response.content.strip()
                elif isinstance(comparative_response, dict) and 'text' in comparative_response:
                    comparative_analysis = comparative_response['text'].strip()
                else:
                    comparative_analysis = str(comparative_response).strip()
                    
            except Exception as e:
                print(f"Error generating comparative analysis: {e}")
                comparative_analysis = f"Error generating comparative analysis: {str(e)}"
        
        processing_time = time.time() - start_time
        
        return {
            "summaries": summaries,
            "comparative_analysis": comparative_analysis,
            "processing_metadata": {
                "summary_type": summary_type,
                "max_length": max_length,
                "include_key_points": include_key_points,
                "compare_documents": compare_documents,
                "processing_timestamp": datetime.now().isoformat(),
                "total_documents": len(documents)
            },
            "total_documents": len(documents),
            "total_processing_time": processing_time
        }

    # ===== RISK ASSESSMENT METHODS =====

    def get_risk_assessment_chain(self) -> LLMChain:
        """Create a chain for risk assessment analysis."""
        RISK_ASSESSMENT_PROMPT = """
        You are an expert legal risk assessment analyst. You will identify, analyze, and provide recommendations for legal risks in documents.
        
        Document Content:
        {document_content}
        
        Assessment Focus: {assessment_focus}
        Document Type: {document_type}
        Risk Categories: {risk_categories}
        Custom Instructions: {custom_instructions}
        
        Please analyze the document and identify risks. For each risk, provide:
        1. The clause type where the risk was found
        2. A clear description of the risk
        3. Severity level (Low, Medium, High, Critical)
        4. A numeric severity score (1-5)
        5. Suggested action to mitigate the risk
        6. Legal basis for the assessment
        7. Impact area (Financial, Legal, Operational, Compliance, etc.)
        8. Confidence score (0.0-1.0)
        
        Focus on these risk categories: {risk_categories}
        
        Format your response as JSON:
        {{
            "risks": [
                {{
                    "clause_type": "Indemnity",
                    "risk_description": "Contractor must indemnify client for all losses without limit",
                    "severity": "High",
                    "severity_score": 4,
                    "suggested_action": "Negotiate a financial cap (e.g., limited to contract value)",
                    "legal_basis": "Unlimited indemnity creates excessive liability exposure",
                    "impact_area": "Financial",
                    "confidence_score": 0.9
                }}
            ],
            "risk_summary": {{
                "total_risks": 5,
                "high_risks": 2,
                "medium_risks": 2,
                "low_risks": 1,
                "critical_risks": 0
            }},
            "overall_risk_level": "High"
        }}
        """
        
        prompt = PromptTemplate(
            template=RISK_ASSESSMENT_PROMPT,
            input_variables=["document_content", "assessment_focus", "document_type", "risk_categories", "custom_instructions"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def assess_document_risks(self, tagged_sections: List[Dict[str, Any]], 
                             document_type: str = "contract",
                             assessment_focus: str = "comprehensive",
                             risk_categories: List[str] = None,
                             custom_instructions: str = None,
                             include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Assess risks in a document using AI analysis.
        
        Args:
            tagged_sections: List of tagged sections from document
            document_type: Type of document being assessed
            assessment_focus: Focus area for assessment
            risk_categories: Categories of risks to assess
            custom_instructions: Custom instructions for assessment
            include_recommendations: Whether to include suggested actions
            
        Returns:
            Dictionary containing risk assessment results
        """
        import time
        start_time = time.time()
        
        if risk_categories is None:
            risk_categories = ["financial", "legal", "compliance", "operational"]
        
        if custom_instructions is None:
            custom_instructions = "Provide a comprehensive risk assessment focusing on potential legal, financial, and operational risks."
        
        try:
            # Prepare document content
            document_content = self._prepare_document_context(tagged_sections)
            
            # Create risk assessment chain
            risk_chain = self.get_risk_assessment_chain()
            
            # Generate risk assessment
            response = risk_chain.invoke({
                "document_content": document_content,
                "assessment_focus": assessment_focus,
                "document_type": document_type,
                "risk_categories": ", ".join(risk_categories),
                "custom_instructions": custom_instructions
            })
            
            # Parse response
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif isinstance(response, dict) and 'text' in response:
                response_text = response['text'].strip()
            else:
                response_text = str(response).strip()
            
            # Try to parse JSON response
            try:
                import json
                import re
                # Find JSON in response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    risk_data = json.loads(json_match.group())
                else:
                    # Fallback if no JSON found
                    risk_data = {
                        "risks": [],
                        "risk_summary": {"total_risks": 0, "high_risks": 0, "medium_risks": 0, "low_risks": 0, "critical_risks": 0},
                        "overall_risk_level": "Unknown"
                    }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                risk_data = {
                    "risks": [],
                    "risk_summary": {"total_risks": 0, "high_risks": 0, "medium_risks": 0, "low_risks": 0, "critical_risks": 0},
                    "overall_risk_level": "Unknown"
                }
            
            # Ensure all required fields are present
            if "risks" not in risk_data:
                risk_data["risks"] = []
            if "risk_summary" not in risk_data:
                risk_data["risk_summary"] = {"total_risks": 0, "high_risks": 0, "medium_risks": 0, "low_risks": 0, "critical_risks": 0}
            if "overall_risk_level" not in risk_data:
                risk_data["overall_risk_level"] = "Unknown"
            
            # Validate and clean risk items
            validated_risks = []
            for risk in risk_data.get("risks", []):
                if isinstance(risk, dict):
                    # Ensure all required fields are present with defaults
                    validated_risk = {
                        "clause_type": risk.get("clause_type", "Unknown"),
                        "risk_description": risk.get("risk_description", "Risk identified"),
                        "severity": risk.get("severity", "Medium"),
                        "severity_score": max(1, min(5, risk.get("severity_score", 3))),
                        "suggested_action": risk.get("suggested_action", "Review and assess this clause") if include_recommendations else "No recommendation provided",
                        "legal_basis": risk.get("legal_basis", "Standard legal analysis"),
                        "impact_area": risk.get("impact_area", "General"),
                        "confidence_score": max(0.0, min(1.0, risk.get("confidence_score", 0.8)))
                    }
                    validated_risks.append(validated_risk)
            
            risk_data["risks"] = validated_risks
            
            # Update risk summary if needed
            if not risk_data["risk_summary"].get("total_risks"):
                risk_data["risk_summary"]["total_risks"] = len(validated_risks)
            
            processing_time = time.time() - start_time
            
            return {
                "risks": validated_risks,
                "risk_summary": risk_data["risk_summary"],
                "overall_risk_level": risk_data["overall_risk_level"],
                "processing_metadata": {
                    "document_type": document_type,
                    "assessment_focus": assessment_focus,
                    "risk_categories": risk_categories,
                    "include_recommendations": include_recommendations,
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_sections_analyzed": len(tagged_sections)
                },
                "assessment_timestamp": datetime.now().isoformat(),
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"Error in risk assessment: {e}")
            # Return error response
            return {
                "risks": [],
                "risk_summary": {"total_risks": 0, "high_risks": 0, "medium_risks": 0, "low_risks": 0, "critical_risks": 0},
                "overall_risk_level": "Error",
                "processing_metadata": {
                    "error": str(e),
                    "document_type": document_type,
                    "assessment_focus": assessment_focus,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "assessment_timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            }

    # ===== CHRONOLOGY-SPECIFIC METHODS =====

    def get_event_extraction_chain(self) -> LLMChain:
        """Extract events and temporal expressions from text using LLM."""
        EVENT_EXTRACTION_PROMPT = """
        You are an expert at extracting events and temporal information from legal documents.
        
        Analyze the following text and identify:
        1. Specific events that occurred
        2. When they occurred (dates, times, temporal expressions)
        3. The type of each event
        
        Event types to consider:
        - Legal Action (lawsuits, court proceedings, legal filings)
        - Contract Event (signing, amendments, terminations)
        - Payment (payments, invoices, financial transactions)
        - Deadline (due dates, expiration dates, timelines)
        - Meeting (meetings, conferences, appointments)
        - Communication (emails, letters, notifications)
        - Document Creation (reports, certificates, licenses)
        - Decision (approvals, rejections, determinations)
        - Notification (alerts, warnings, announcements)
        - Other (any other significant event)
        
        For each event, provide:
        - Event description (what happened)
        - Temporal expression (when it happened)
        - Event type
        - Confidence level (0.0 to 1.0)
        
        Text: {context}
        
        Format your response as JSON with this structure:
        {{
            "events": [
                {{
                    "description": "Brief description of the event",
                    "temporal_expression": "Date or time reference found in text",
                    "event_type": "One of the event types listed above",
                    "confidence": 0.8
                }}
            ]
        }}
        """
        prompt = PromptTemplate(template=EVENT_EXTRACTION_PROMPT, input_variables=["context"])
        return LLMChain(llm=self.llm, prompt=prompt)

    def get_temporal_resolution_chain(self) -> LLMChain:
        """Resolve ambiguous temporal expressions using LLM."""
        TEMPORAL_RESOLUTION_PROMPT = """
        You are an expert at resolving temporal expressions and normalizing dates.
        
        Given a temporal expression and a reference date, determine the exact date.
        
        Reference Date: {reference_date}
        Temporal Expression: {temporal_expression}
        Context: {context}
        
        Rules:
        1. If the expression is absolute (e.g., "January 15, 2024"), use that date
        2. If the expression is relative (e.g., "two weeks later"), calculate from reference date
        3. If ambiguous, provide the most likely interpretation
        4. Consider the context to resolve ambiguities
        
        Respond with a JSON object:
        {{
            "normalized_date": "YYYY-MM-DD",
            "confidence": 0.9,
            "explanation": "Brief explanation of how the date was determined"
        }}
        """
        prompt = PromptTemplate(
            template=TEMPORAL_RESOLUTION_PROMPT, 
            input_variables=["reference_date", "temporal_expression", "context"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def extract_events_and_timestamps(self, text: str, document_section: str = "unknown") -> List[Event]:
        """Extract events and their temporal information from text."""
        if not CHRONOLOGY_AVAILABLE or not self.event_extractor:
            print("Chronology features not available")
            return []
            
        # Use LLM for event extraction
        event_chain = self.get_event_extraction_chain()
        
        try:
            # Get LLM response using invoke instead of deprecated run
            llm_response = event_chain.invoke({"context": text})
            
            # Handle different response formats
            if hasattr(llm_response, 'content'):
                response_text = llm_response.content
            elif isinstance(llm_response, dict) and 'text' in llm_response:
                response_text = llm_response['text']
            else:
                response_text = str(llm_response)
            
            # Clean and validate JSON response
            response_text = response_text.strip()
            if not response_text:
                print("Empty response from LLM")
                return self.event_extractor.identify_events(text, document_section)
            
            # Try to find JSON in the response (in case there's extra text)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            # Parse JSON response
            import json
            events_data = json.loads(response_text)
            
            # Convert to Event objects
            events = []
            for event_data in events_data.get("events", []):
                # Fallback to rule-based extraction if temporal processor unavailable
                if not self.temporal_processor:
                    temporal_expressions = []
                else:
                    temporal_expressions = self.temporal_processor.extract_temporal_expressions(
                        event_data.get("temporal_expression", "")
                    )
                
                # Create Event object with all required parameters
                event = Event(
                    id=f"llm_{len(events)}",
                    description=event_data.get("description", ""),
                    event_type=EventType(event_data.get("event_type", "other").lower().replace(" ", "_")),
                    timestamp=None,  # Add missing timestamp parameter
                    normalized_date=None,  # Add missing normalized_date parameter
                    source_text=text,
                    document_section=document_section,
                    temporal_expressions=temporal_expressions,
                    confidence_score=event_data.get("confidence", 0.5),
                    metadata={
                        "extraction_method": "llm",
                        "llm_response": event_data
                    }
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            print(f"Error in LLM event extraction: {e}")
            # Fallback to rule-based extraction
            return self.event_extractor.identify_events(text, document_section)

    # ===== COMMENTED OUT - UNUSED METHODS =====
    # These methods are defined but not actually called in the current implementation
    
    # def normalize_temporal_expressions(self, text: str, document_date: Optional[date] = None) -> List[Dict[str, Any]]:
    #     """Normalize temporal expressions in text using LLM."""
    #     if document_date is None:
    #         document_date = date.today()
    #     
    #     # Simple fallback when temporal processor is not available
    #     if not self.temporal_processor:
    #         return []
    #     
    #     # Extract temporal expressions
    #     temporal_expressions = self.temporal_processor.extract_temporal_expressions(text)
    #     
    #     # Use LLM for complex resolution
    #     resolution_chain = self.get_temporal_resolution_chain()
    #     normalized_expressions = []
    #     
    #     for expr in temporal_expressions:
    #         try:
    #             llm_response = resolution_chain.invoke({
    #                 "reference_date": document_date.isoformat(),
    #                 "temporal_expression": expr.text,
    #                 "context": text
    #             })
    #             
    #             import json
    #             resolution_data = json.loads(llm_response)
    #             
    #             normalized_expressions.append({
    #                 "original_text": expr.text,
    #                 "normalized_date": resolution_data.get("normalized_date"),
    #                 "confidence": resolution_data.get("confidence", 0.5),
    #                 "explanation": resolution_data.get("explanation", ""),
    #                 "position": expr.position
    #             })
    #             
    #         except Exception as e:
    #             print(f"Error resolving temporal expression '{expr.text}': {e}")
    #             # Fallback to rule-based resolution
    #             normalized_date = self.temporal_processor._resolve_relative_date(
    #                 expr.text, document_date
    #             ) if expr.is_relative else self.temporal_processor._parse_absolute_date(expr.text)
    #             
    #             normalized_expressions.append({
    #                 "original_text": expr.text,
    #                 "normalized_date": normalized_date.isoformat() if normalized_date else None,
    #                 "confidence": expr.confidence,
    #                 "explanation": "Rule-based resolution",
    #                 "position": expr.position
    #             })
    #     
    #     return normalized_expressions

    # def resolve_relative_dates(self, events: List[Event], reference_date: date) -> List[Event]:
    #     """Resolve relative dates in events using LLM assistance."""
    #     return self.temporal_processor.resolve_relative_dates(events, reference_date)

    def build_chronology(self, text: str, document_id: str, 
                        document_date: Optional[date] = None) -> Timeline:
        """Build a complete chronology from text using LLM-enhanced processing."""
        if not CHRONOLOGY_AVAILABLE or not self.chronology_builder:
            print("Chronology features not available")
            return None
            
        if document_date is None:
            document_date = date.today()
        
        # Extract events using LLM
        events = self.extract_events_and_timestamps(text, "main_document")
        
        # Build timeline
        timeline = self.chronology_builder.build_timeline(events, document_id, document_date)
        
        return timeline

    # ===== DOCUMENT CLASSIFICATION METHODS =====

    def get_document_classification_chain(self) -> LLMChain:
        """Create a chain for document classification."""
        DOCUMENT_CLASSIFICATION_PROMPT = """
        You are an expert legal document classifier. You will analyze legal and corporate documents and classify them across three key dimensions.
        
        Document Content:
        {document_content}
        
        Document Type Hint: {document_type_hint}
        Classification Focus: {classification_focus}
        
        Please classify this document based on the following criteria:
        
        1. **Document Type**: Choose from these categories:
           - Contract (general agreements, service contracts, purchase agreements)
           - NDA (Non-Disclosure Agreement, confidentiality agreements)
           - Employment Agreement (employment contracts, offer letters, HR documents)
           - Licensing Agreement (software licenses, IP licenses, franchise agreements)
           - Court Filing (legal pleadings, motions, court documents)
           - Policy Document (company policies, procedures, guidelines)
           - Lease Agreement (property leases, equipment leases)
           - Partnership Agreement (joint ventures, partnerships)
           - Terms of Service (website terms, user agreements)
           - Other (if none of the above match)
        
        2. **Subject/Focus Area**: Choose from these categories:
           - Employment Terms (hiring, termination, benefits, compensation)
           - Confidentiality (NDAs, trade secrets, proprietary information)
           - Payment Terms (billing, invoicing, financial obligations)
           - Partnership (joint ventures, collaborations, alliances)
           - Compliance (regulatory requirements, legal compliance)
           - Regulatory (government regulations, industry standards)
           - Litigation (legal disputes, court proceedings, settlements)
           - Intellectual Property (patents, trademarks, copyrights)
           - Real Estate (property transactions, leases, zoning)
           - Technology (software, IT services, data protection)
           - Other (if none of the above match)
        
        3. **Importance/Priority**: Choose from these levels:
           - High (critical for legal/financial risk, requires immediate attention)
           - Medium (relevant but routine, standard business operations)
           - Low (informational/reference only, minimal legal impact)
        
        For each classification, provide:
        - The classification choice
        - A brief reasoning (1-2 sentences) explaining why you classified it this way
        - A confidence score (0.0 to 1.0)
        
        Format your response as JSON:
        {{
            "document_type": "Contract",
            "subject": "Payment Terms",
            "importance": "High",
            "reasoning": {{
                "document_type": "This document contains mutual obligations and terms between parties, indicating it's a contract.",
                "subject": "The document primarily focuses on payment schedules, billing terms, and financial obligations.",
                "importance": "This contract involves significant financial commitments and legal obligations that could impact business operations."
            }},
            "confidence_scores": {{
                "document_type": 0.9,
                "subject": 0.85,
                "importance": 0.8
            }}
        }}
        """
        
        prompt = PromptTemplate(
            template=DOCUMENT_CLASSIFICATION_PROMPT,
            input_variables=["document_content", "document_type_hint", "classification_focus"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def classify_document(self, tagged_sections: List[Dict[str, Any]], 
                         document_type_hint: str = None,
                         classification_focus: str = None) -> Dict[str, Any]:
        """
        Classify a document based on its content and structure.
        
        Args:
            tagged_sections: List of tagged sections from document
            document_type_hint: Optional hint about the document type
            classification_focus: Optional focus area for classification
            
        Returns:
            Dictionary containing classification results
        """
        import time
        start_time = time.time()
        
        try:
            # Prepare document content
            document_content = self._prepare_document_context(tagged_sections)
            
            # Create classification chain
            classification_chain = self.get_document_classification_chain()
            
            # Generate classification
            response = classification_chain.invoke({
                "document_content": document_content,
                "document_type_hint": document_type_hint or "Not specified",
                "classification_focus": classification_focus or "General classification"
            })
            
            # Parse response
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif isinstance(response, dict) and 'text' in response:
                response_text = response['text'].strip()
            else:
                response_text = str(response).strip()
            
            # Try to parse JSON response
            try:
                import json
                import re
                # Find JSON in response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    classification_data = json.loads(json_match.group())
                else:
                    # Fallback if no JSON found
                    classification_data = {
                        "document_type": "Other",
                        "subject": "Other",
                        "importance": "Medium",
                        "reasoning": {
                            "document_type": "Unable to determine document type from content.",
                            "subject": "Unable to determine subject focus from content.",
                            "importance": "Unable to assess importance level from content."
                        },
                        "confidence_scores": {
                            "document_type": 0.3,
                            "subject": 0.3,
                            "importance": 0.3
                        }
                    }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                classification_data = {
                    "document_type": "Other",
                    "subject": "Other",
                    "importance": "Medium",
                    "reasoning": {
                        "document_type": "Error parsing classification response.",
                        "subject": "Error parsing classification response.",
                        "importance": "Error parsing classification response."
                    },
                    "confidence_scores": {
                        "document_type": 0.1,
                        "subject": 0.1,
                        "importance": 0.1
                    }
                }
            
            # Ensure all required fields are present
            if "document_type" not in classification_data:
                classification_data["document_type"] = "Other"
            if "subject" not in classification_data:
                classification_data["subject"] = "Other"
            if "importance" not in classification_data:
                classification_data["importance"] = "Medium"
            if "reasoning" not in classification_data:
                classification_data["reasoning"] = {
                    "document_type": "No reasoning provided",
                    "subject": "No reasoning provided",
                    "importance": "No reasoning provided"
                }
            if "confidence_scores" not in classification_data:
                classification_data["confidence_scores"] = {
                    "document_type": 0.5,
                    "subject": 0.5,
                    "importance": 0.5
                }
            
            # Validate confidence scores
            for key in ["document_type", "subject", "importance"]:
                if key in classification_data["confidence_scores"]:
                    classification_data["confidence_scores"][key] = max(0.0, min(1.0, classification_data["confidence_scores"][key]))
                else:
                    classification_data["confidence_scores"][key] = 0.5
            
            processing_time = time.time() - start_time
            
            return {
                "document_type": classification_data["document_type"],
                "subject": classification_data["subject"],
                "importance": classification_data["importance"],
                "reasoning": classification_data["reasoning"],
                "confidence_scores": classification_data["confidence_scores"],
                "processing_metadata": {
                    "document_type_hint": document_type_hint,
                    "classification_focus": classification_focus,
                    "total_sections_analyzed": len(tagged_sections),
                    "document_content_length": len(document_content),
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_time": processing_time
                }
            }
            
        except Exception as e:
            print(f"Error in document classification: {e}")
            # Return error response
            return {
                "document_type": "Other",
                "subject": "Other",
                "importance": "Medium",
                "reasoning": {
                    "document_type": f"Error during classification: {str(e)}",
                    "subject": f"Error during classification: {str(e)}",
                    "importance": f"Error during classification: {str(e)}"
                },
                "confidence_scores": {
                    "document_type": 0.0,
                    "subject": 0.0,
                    "importance": 0.0
                },
                "processing_metadata": {
                    "error": str(e),
                    "document_type_hint": document_type_hint,
                    "classification_focus": classification_focus,
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time
                }
            }

    # ===== DOCUMENT COMPARISON METHODS =====
    # COMMENTED OUT - Document comparison functionality disabled

    # def get_document_comparison_chain(self) -> LLMChain:
    #     """Create a chain for document comparison analysis."""
    #     DOCUMENT_COMPARISON_PROMPT = """
    #     You are an expert legal document comparison analyst. You will compare two or more legal documents and identify all differences, including changes in clauses, obligations, and timelines.
    #     
    #     Documents to Compare:
    #     {documents_content}
    #     
    #     Comparison Focus: {comparison_focus}
    #     Include Word-Level Changes: {include_word_level}
    #     Risk Threshold: {risk_threshold}
    #     
    #     Please perform a comprehensive comparison and identify:
    #     
    #     1. **Clause Differences**: Compare clauses between documents and identify:
    #        - Added clauses (new clauses in later documents)
    #        - Deleted clauses (removed clauses from earlier documents)
    #        - Modified clauses (changed content in existing clauses)
    #        - Unchanged clauses (for reference)
    #     
    #     2. **Obligation Differences**: Compare obligations of each party and identify:
    #        - Changes in party responsibilities
    #        - New obligations added
    #        - Obligations removed or modified
    #        - Changes in performance requirements
    #     
    #     3. **Timeline Differences**: Compare dates, deadlines, and timelines:
    #        - Contract start/end dates
    #        - Payment schedules
    #        - Delivery deadlines
    #        - Termination notice periods
    #        - Renewal dates
    #     
    #     4. **Risk Assessment**: For each difference, assess the risk level:
    #        - Low: Minor changes with minimal impact
    #        - Medium: Moderate changes that may affect operations
    #        - High: Significant changes with potential legal/financial impact
    #        - Critical: Major changes that could have severe consequences
    #     
    #     For each difference, provide:
    #     - Clear description of what changed
    #     - Risk level assessment
    #     - Reasoning for the risk assessment
    #     - Legal or business implications
    #     
    #     Format your response as JSON:
    #     {{
    #         "comparison_summary": "Brief summary of key differences and overall impact",
    #         "clause_differences": [
    #             {{
    #                 "clause_title": "Payment Terms",
    #                 "document_a_content": "Net 30 payment terms",
    #                 "document_b_content": "Net 45 payment terms",
    #                 "change_type": "Modified",
    #                 "risk_level": "Medium",
    #                 "reasoning": "Extended payment terms may impact cash flow but provide more flexibility"
    #             }}
    #         ],
    #         "obligation_differences": [
    #             {{
    #                 "party": "Vendor",
    #                 "document_a_obligation": "Deliver goods within 30 days",
    #                 "document_b_obligation": "Deliver goods within 45 days",
    #                 "change_type": "Modified",
    #                 "risk_level": "Low",
    #                 "reasoning": "Extended delivery time reduces pressure but may delay project timelines"
    #             }}
    #         ],
    #         "timeline_differences": [
    #             {{
    #                 "event": "Contract Termination Date",
    #                 "document_a_timeline": "31-Dec-2025",
    #                 "document_b_timeline": "31-Mar-2026",
    #                 "change_type": "Modified",
    #                 "risk_level": "High",
    #                 "reasoning": "Extended contract term increases long-term commitment and potential liability exposure"
    #             }}
    #         ],
    #         "high_risk_changes": [
    #             {{
    #                 "change_type": "clause",
    #                 "title": "Payment Terms",
    #                 "risk_level": "High",
    #                 "description": "Extended payment terms from 30 to 45 days",
    #                 "implications": "May impact cash flow and working capital requirements"
    #             }}
    #         ],
    #         "overall_risk_assessment": "Medium - Several moderate changes that require attention but no critical risks identified"
    #     }}
    #     """
    #     
    #     prompt = PromptTemplate(
    #         template=DOCUMENT_COMPARISON_PROMPT,
    #         input_variables=["documents_content", "comparison_focus", "include_word_level", "risk_threshold"]
    #     )
    #     return LLMChain(llm=self.llm, prompt=prompt)

    # def compare_documents(self, documents: List[Dict[str, Any]], 
    #                      comparison_focus: str = "comprehensive",
    #                      include_word_level: bool = False,
    #                      risk_threshold: str = "Medium") -> Dict[str, Any]:
    #     """
    #     Compare multiple documents and identify differences.
    #     
    #     Args:
    #         documents: List of documents with tagged_sections
    #         comparison_focus: Focus area for comparison
    #         include_word_level: Whether to include word-level changes
    #         risk_threshold: Minimum risk level to highlight
    #         
    #     Returns:
    #         Dictionary containing comparison results
    #     """
    #     import time
    #     start_time = time.time()
    #     
    #     try:
    #         # Prepare documents content for comparison
    #         documents_content = self._prepare_documents_for_comparison(documents)
    #         
    #         # Create comparison chain
    #         comparison_chain = self.get_document_comparison_chain()
    #         
    #         # Generate comparison
    #         response = comparison_chain.invoke({
    #             "documents_content": documents_content,
    #             "comparison_focus": comparison_focus,
    #             "include_word_level": str(include_word_level),
    #             "risk_threshold": risk_threshold
    #         })
    #         
    #         # Parse response
    #         if hasattr(response, 'content'):
    #             response_text = response.content.strip()
    #         elif isinstance(response, dict) and 'text' in response:
    #             response_text = response['text'].strip()
    #         else:
    #             response_text = str(response).strip()
    #         
    #         # Try to parse JSON response
    #         try:
    #             import json
    #             import re
    #             # Find JSON in response
    #             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    #             if json_match:
    #                 comparison_data = json.loads(json_match.group())
    #             else:
    #                 # Fallback if no JSON found
    #                 comparison_data = {
    #                     "comparison_summary": "Unable to parse comparison results from AI response.",
    #                     "clause_differences": [],
    #                     "obligation_differences": [],
    #                     "timeline_differences": [],
    #                     "high_risk_changes": [],
    #                     "overall_risk_assessment": "Unknown"
    #                 }
    #         except json.JSONDecodeError:
    #             # Fallback if JSON parsing fails
    #             comparison_data = {
    #                 "comparison_summary": "Error parsing comparison response from AI.",
    #                 "clause_differences": [],
    #                 "obligation_differences": [],
    #                 "timeline_differences": [],
    #                 "high_risk_changes": [],
    #                 "overall_risk_assessment": "Error"
    #             }
    #         
    #         # Ensure all required fields are present
    #         if "comparison_summary" not in comparison_data:
    #             comparison_data["comparison_summary"] = "Comparison completed with limited results."
    #         if "clause_differences" not in comparison_data:
    #             comparison_data["clause_differences"] = []
    #         if "obligation_differences" not in comparison_data:
    #             comparison_data["obligation_differences"] = []
    #         if "timeline_differences" not in comparison_data:
    #             comparison_data["timeline_differences"] = []
    #         if "high_risk_changes" not in comparison_data:
    #             comparison_data["high_risk_changes"] = []
    #         if "overall_risk_assessment" not in comparison_data:
    #             comparison_data["overall_risk_assessment"] = "Medium"
    #         
    #         # Validate and clean differences
    #         validated_clause_differences = []
    #         for diff in comparison_data.get("clause_differences", []):
    #             if isinstance(diff, dict):
    #                 validated_clause_differences.append({
    #                     "clause_title": diff.get("clause_title", "Unknown Clause"),
    #                     "document_a_content": diff.get("document_a_content", "Not specified"),
    #                     "document_b_content": diff.get("document_b_content", "Not specified"),
    #                     "change_type": diff.get("change_type", "Unknown"),
    #                     "risk_level": diff.get("risk_level", "Low"),
    #                     "reasoning": diff.get("reasoning", "No reasoning provided")
    #                 })
    #         
    #         validated_obligation_differences = []
    #         for diff in comparison_data.get("obligation_differences", []):
    #             if isinstance(diff, dict):
    #                 validated_obligation_differences.append({
    #                     "party": diff.get("party", "Unknown Party"),
    #                     "document_a_obligation": diff.get("document_a_obligation", "Not specified"),
    #                     "document_b_obligation": diff.get("document_b_obligation", "Not specified"),
    #                     "change_type": diff.get("change_type", "Unknown"),
    #                     "risk_level": diff.get("risk_level", "Low"),
    #                     "reasoning": diff.get("reasoning", "No reasoning provided")
    #                 })
    #         
    #         validated_timeline_differences = []
    #         for diff in comparison_data.get("timeline_differences", []):
    #             if isinstance(diff, dict):
    #                 validated_timeline_differences.append({
    #                     "event": diff.get("event", "Unknown Event"),
    #                     "document_a_timeline": diff.get("document_a_timeline", "Not specified"),
    #                     "document_b_timeline": diff.get("document_b_timeline", "Not specified"),
    #                     "change_type": diff.get("change_type", "Unknown"),
    #                     "risk_level": diff.get("risk_level", "Low"),
    #                     "reasoning": diff.get("reasoning", "No reasoning provided")
    #                 })
    #         
    #         # Filter high-risk changes based on threshold
    #         risk_levels = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    #         threshold_level = risk_levels.get(risk_threshold, 2)
    #         
    #         filtered_high_risk_changes = []
    #         for change in comparison_data.get("high_risk_changes", []):
    #             if isinstance(change, dict):
    #                 change_risk = risk_levels.get(change.get("risk_level", "Low"), 1)
    #                 if change_risk >= threshold_level:
    #                     filtered_high_risk_changes.append(change)
    #         
    #         processing_time = time.time() - start_time
    #         
    #         return {
    #             "comparison_summary": comparison_data["comparison_summary"],
    #             "clause_differences": validated_clause_differences,
    #             "obligation_differences": validated_obligation_differences,
    #             "timeline_differences": validated_timeline_differences,
    #             "high_risk_changes": filtered_high_risk_changes,
    #             "overall_risk_assessment": comparison_data["overall_risk_assessment"],
    #             "processing_metadata": {
    #                 "comparison_focus": comparison_focus,
    #                 "include_word_level": include_word_level,
    #                 "risk_threshold": risk_threshold,
    #                 "total_documents_compared": len(documents),
    #                 "total_differences": len(validated_clause_differences) + len(validated_obligation_differences) + len(validated_timeline_differences),
    #                 "high_risk_count": len(filtered_high_risk_changes),
    #                 "processing_timestamp": datetime.now().isoformat(),
    #                 "processing_time": processing_time
    #             }
    #         }
    #         
    #     except Exception as e:
    #         print(f"Error in document comparison: {e}")
    #         # Return error response
    #         return {
    #             "comparison_summary": f"Error during document comparison: {str(e)}",
    #             "clause_differences": [],
    #             "obligation_differences": [],
    #             "timeline_differences": [],
    #             "high_risk_changes": [],
    #             "overall_risk_assessment": "Error",
    #             "processing_metadata": {
    #                 "error": str(e),
    #                 "comparison_focus": comparison_focus,
    #                 "include_word_level": include_word_level,
    #                 "risk_threshold": risk_threshold,
    #                 "processing_timestamp": datetime.now().isoformat(),
    #                 "processing_time": time.time() - start_time
    #             }
    #         }

    # def _prepare_documents_for_comparison(self, documents: List[Dict[str, Any]]) -> str:
    #     """Prepare documents content for comparison analysis."""
    #     documents_content = []
    #     
    #     for i, doc in enumerate(documents):
    #         doc_id = doc.get("document_id", f"Document_{i+1}")
    #         doc_title = doc.get("title", f"Document {i+1}")
    #         doc_type = doc.get("document_type", "legal_document")
    #         
    #         # Prepare document content
    #         tagged_sections = doc.get("tagged_sections", [])
    #         document_content = self._prepare_document_context(tagged_sections)
    #         
    #         documents_content.append(f"""
    # Document {i+1}: {doc_title}
    # Document ID: {doc_id}
    # Document Type: {doc_type}
    # Content:
    # {document_content}
    # ---END OF DOCUMENT {i+1}---
    #         """)
    #     
    #     return "\n\n".join(documents_content)

    # ===== COMMENTED OUT - UNUSED METHOD =====
    # This method combines clause analysis and chronology but is not used in current implementation
    # The system uses separate endpoints for clause analysis and chronology extraction
    
    # def process_contract_with_chronology(self, tagged_sections: List[Dict[str, Any]], 
    #                                    document_id: str, document_date: Optional[date] = None) -> Dict[str, Any]:
    #     """
    #     Enhanced contract processing that includes chronology building.
    #     Combines the original clause analysis with event extraction and timeline building.
    #     """
    #     if not CHRONOLOGY_AVAILABLE or not self.chronology_builder:
    #         print("Chronology features not available - returning clause analysis only")
    #         clause_results = self.process_contract(tagged_sections)
    #         return {
    #             "clause_analysis": clause_results,
    #             "chronology": None,
    #             "processing_metadata": {
    #                 "total_sections": len(tagged_sections),
    #                 "total_events": 0,
    #                 "document_date": document_date.isoformat() if document_date else None,
    #                 "processing_timestamp": datetime.now().isoformat(),
    #                 "note": "Chronology features not available"
    #             }
    #         }
    #         
    #     if document_date is None:
    #         document_date = date.today()
    #     
    #     # Original clause processing
    #     clause_results = self.process_contract(tagged_sections)
    #     
    #     # Extract events from all sections
    #     all_events = []
    #     for section in tagged_sections:
    #         section_text = section.get("body", "")
    #         section_events = self.extract_events_and_timestamps(section_text, section.get("heading", "unknown"))
    #         all_events.extend(section_events)
    #     
    #     # Build chronology
    #     timeline = self.chronology_builder.build_timeline(all_events, document_id, document_date)
    #     
    #     # Generate comprehensive report
    #     chronology_report = self.chronology_builder.generate_chronology_report(timeline)
    #     
    #     return {
    #         "clause_analysis": clause_results,
    #         "chronology": {
    #             "timeline": timeline,
    #             "report": chronology_report,
    #             "export_json": self.chronology_builder.export_timeline(timeline, "json"),
    #             "export_summary": self.chronology_builder.export_timeline(timeline, "summary")
    #         },
    #         "processing_metadata": {
    #             "total_sections": len(tagged_sections),
    #             "total_events": len(all_events),
    #             "document_date": document_date.isoformat(),
    #             "processing_timestamp": datetime.now().isoformat()
    #         }
    #     }
