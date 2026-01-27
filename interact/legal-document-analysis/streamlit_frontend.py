import streamlit as st
import requests
import json
from datetime import datetime, date, timedelta
import io

# Try to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly/Pandas not available - visualizations will be limited")
    PLOTLY_AVAILABLE = False

# Configuration
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Legal Document Analysis Platform", layout="wide")
st.title("📑 Legal Document Analysis Platform")

# Main description
st.markdown("""
**Comprehensive AI-powered legal document analysis platform with advanced features:**

🔍 **Document Processing**: Extract text from PDFs or images (PNG, JPG, JPEG) and analyze document structure  
📋 **Clause Analysis**: Identify and analyze legal clauses with AI insights  
📅 **Chronology Builder**: Extract and organize chronological events from documents  
💬 **Document Chat**: Interactive AI assistant for document Q&A  
📄 **Document Summarization**: Generate customizable summaries with specific instructions  
⚠️ **Risk Assessment**: Identify legal risks with detailed analysis and recommendations  
🏷️ **Document Classification**: Classify documents by type, subject, and importance level  

**🚀 New Simplified Workflow**: Upload a PDF or image directly to any analysis mode - no need to extract text first!
""")

# Sidebar for mode selection
st.sidebar.title("🔧 Analysis Features")
st.sidebar.markdown("**Choose your analysis type:**")
analysis_mode = st.sidebar.radio(
    "Available Features:",
    # ["📝 Text Extraction", "📋 Clause Analysis", "📅 Chronology Builder", "💬 Document Chat", "📄 Document Summarization", "⚠️ Risk Assessment", "🏷️ Document Classification", "🔄 Document Comparison"],
    ["📝 Text Extraction", "📋 Clause Analysis", "📅 Chronology Builder", "💬 Document Chat", "📄 Document Summarization", "⚠️ Risk Assessment", "🏷️ Document Classification"],
    label_visibility="collapsed"
)

# Document date input for chronology
document_date = None
if analysis_mode == "📅 Chronology Builder":
    st.sidebar.subheader("📅 Document Reference Date")
    document_date_input = st.sidebar.date_input(
        "Document date (for relative date resolution):",
        value=date.today(),
        help="This date will be used to resolve relative dates like 'two weeks later'"
    )
    document_date = document_date_input

def call_backend_api(endpoint, files=None, json_data=None, params=None):
    """Helper function to call backend API"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        if files:
            response = requests.post(url, files=files, params=params)
        elif json_data:
            response = requests.post(url, json=json_data, params=params)
        else:
            response = requests.post(url, params=params)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please make sure the FastAPI backend is running on http://localhost:8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def display_tagged_sections(tagged_sections):
    """Display tagged sections in a nice format"""
    st.subheader("📋 Tagged Sections")
    
    for i, section in enumerate(tagged_sections):
        with st.expander(f"Section {i+1}: {section['heading']}"):
            st.write(f"**Heading:** {section['heading']}")
            st.write(f"**Body:** {section['body']}")
            if section.get('documents'):
                st.write(f"**Chunks:** {len(section['documents'])}")
                for j, doc in enumerate(section['documents'][:2]):  # Show first 2 chunks
                    st.write(f"  Chunk {j+1}: {doc[:100]}...")

def display_clause_analysis(clause_analysis):
    """Display clause analysis results"""
    st.subheader("📋 Legal Clause Analysis")
    
    for section in clause_analysis:
        st.write(f"**{section['heading']}**")
        for clause in section['clauses']:
            st.write(f"- **{clause['clause']}**: {clause['summary']}")
        st.write("---")

def display_chronology_timeline(timeline_data):
    """Display chronology timeline with visualizations"""
    timeline = timeline_data.get('timeline', {})
    events = timeline.get('events', [])
    
    if not events:
        st.warning("No events found in the timeline")
        return
    
    # Timeline summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(events))
    with col2:
        date_range = timeline.get('date_range', {})
        start_date = date_range.get('start')
        end_date = date_range.get('end')
        st.metric("Date Range", f"{start_date} to {end_date}" if start_date and end_date else "N/A")
    with col3:
        confidence_summary = timeline.get('confidence_summary', {})
        avg_conf = confidence_summary.get('average', 0)
        st.metric("Avg Confidence", f"{avg_conf:.2f}")
    with col4:
        conflicts = timeline.get('temporal_conflicts', [])
        st.metric("Conflicts", len(conflicts))
    
    # Event type distribution
    if PLOTLY_AVAILABLE and events:
        event_types = [event.get('event_type', 'other') for event in events]
        type_counts = pd.Series(event_types).value_counts()
        
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Event Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Events table
    st.subheader("📋 Detailed Events")
    events_data = []
    for event in events:
        events_data.append({
            'Date': event.get('normalized_date', 'N/A'),
            'Type': event.get('event_type', 'N/A'),
            'Description': event.get('description', 'N/A'),
            'Confidence': f"{event.get('confidence_score', 0):.2f}",
            'Section': event.get('document_section', 'N/A')
        })
    
    if PLOTLY_AVAILABLE:
        df_events = pd.DataFrame(events_data)
        st.dataframe(df_events, use_container_width=True)
    else:
        for i, event_data in enumerate(events_data):
            st.write(f"**Event {i+1}**")
            st.write(f"Date: {event_data['Date']}")
            st.write(f"Type: {event_data['Type']}")
            st.write(f"Description: {event_data['Description']}")
            st.write(f"Confidence: {event_data['Confidence']}")
            st.write(f"Section: {event_data['Section']}")
            st.write("---")

def display_chat_interface(tagged_sections):
    """Display chat interface for document interaction"""
    st.subheader("💬 AI Document Chat Assistant")
    st.markdown("**Interactive Q&A with your document using advanced AI**")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for i, msg in enumerate(st.session_state.chat_history):
            if msg['role'] == 'user':
                st.write(f"**You:** {msg['content']}")
            else:
                st.write(f"**Assistant:** {msg['content']}")
            st.write("---")
    
    # Chat input
    user_input = st.text_input("Ask a question about the document:", placeholder="e.g., What are the key terms of this contract?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", type="primary")
    with col2:
        clear_button = st.button("Clear Chat")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if send_button and user_input:
        with st.spinner("Thinking..."):
            # Prepare chat history for API
            chat_history_for_api = []
            for msg in st.session_state.chat_history:
                chat_history_for_api.append({
                    "role": msg['role'],
                    "content": msg['content'],
                    "timestamp": msg['timestamp']
                })
            
            # Call chat API
            result = call_backend_api("/chat-with-document/", json_data={
                "tagged_sections": tagged_sections,
                "user_message": user_input,
                "chat_history": chat_history_for_api,
                "document_type": "legal_document"
            })
        
        if result:
            # Update chat history
            st.session_state.chat_history = result.get('chat_history', [])
            st.rerun()
        else:
            st.error("Failed to get response from the chat service")
    
    # Show metadata
    if st.session_state.chat_history:
        st.subheader("📊 Chat Metadata")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", len(st.session_state.chat_history))
        with col2:
            user_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
            st.metric("Your Questions", user_messages)

def display_summarization_interface(tagged_sections):
    """Display document summarization interface"""
    st.subheader("📄 AI Document Summarization")
    st.markdown("**Generate intelligent summaries with customizable instructions**")
    
    # Summarization options
    col1, col2 = st.columns(2)
    
    with col1:
        summary_type = st.selectbox(
            "Summary Type:",
            ["comprehensive", "executive", "bullet_points", "custom"],
            help="Choose the type of summary to generate"
        )
        
        max_length = st.slider(
            "Maximum Length (words):",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum number of words for each summary"
        )
    
    with col2:
        include_key_points = st.checkbox(
            "Include Key Points",
            value=True,
            help="Extract and display key points from each document"
        )
        
        # COMMENTED OUT - Document comparison functionality disabled
        # compare_documents = st.checkbox(
        #     "Compare Documents",
        #     value=False,
        #     help="Provide comparative analysis across documents"
        # )
        compare_documents = False  # Always False since comparison is disabled
    
    # Custom instructions
    st.subheader("📝 Summary Instructions")
    default_instructions = {
        "comprehensive": "Provide a comprehensive summary covering all major sections and key information.",
        "executive": "Create an executive summary focusing on the most important points for decision-makers.",
        "bullet_points": "Summarize using bullet points and key takeaways.",
        "custom": "Customize your summary instructions below."
    }
    
    if summary_type == "custom":
        summary_instructions = st.text_area(
            "Custom Summary Instructions:",
            placeholder="e.g., Focus on legal risks, payment terms, and termination clauses...",
            height=100
        )
    else:
        summary_instructions = st.text_area(
            "Summary Instructions:",
            value=default_instructions[summary_type],
            height=100
        )
    
    # Generate summary button
    if st.button("Generate Summary", type="primary"):
        if not summary_instructions.strip():
            st.error("Please provide summary instructions")
        else:
            with st.spinner("Generating summaries..."):
                # Prepare document for summarization
                document_data = {
                    "document_id": "current_document",
                    "title": "Current Document",
                    "document_type": "legal_document",
                    "tagged_sections": tagged_sections
                }
                
                # Call summarization API
                result = call_backend_api("/summarize-documents/", json_data={
                    "documents": [document_data],
                    "summary_instructions": summary_instructions,
                    "summary_type": summary_type,
                    "max_length": max_length,
                    "include_key_points": include_key_points,
                    "compare_documents": compare_documents
                })
            
            if result:
                st.success("✅ Summary generated successfully!")
                
                # Display summaries
                summaries = result.get('summaries', [])
                for summary in summaries:
                    st.subheader(f"📄 {summary['title']}")
                    
                    # Summary content
                    st.write("**Summary:**")
                    st.write(summary['summary'])
                    
                    # Key points
                    if summary.get('key_points') and include_key_points:
                        st.write("**Key Points:**")
                        for point in summary['key_points']:
                            st.write(f"• {point}")
                    
                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", summary.get('word_count', 0))
                    with col2:
                        st.metric("Confidence", f"{summary.get('confidence_score', 0):.2f}")
                    with col3:
                        st.metric("Document Type", summary.get('document_type', 'Unknown'))
                    
                    st.write("---")
                
                # Comparative analysis - COMMENTED OUT - Document comparison functionality disabled
                # if result.get('comparative_analysis') and compare_documents:
                #     st.subheader("🔍 Comparative Analysis")
                #     st.write(result['comparative_analysis'])
                
                # Processing metadata
                st.subheader("📊 Processing Information")
                metadata = result.get('processing_metadata', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", result.get('total_documents', 0))
                with col2:
                    st.metric("Processing Time", f"{result.get('total_processing_time', 0):.2f}s")
                with col3:
                    st.metric("Summary Type", metadata.get('summary_type', 'Unknown'))
            else:
                st.error("Failed to generate summary")
    
    # Instructions and tips
    st.subheader("💡 Tips for Better Summaries")
    st.info("""
    **For better results:**
    - Be specific about what you want to focus on
    - Mention key areas like "legal risks", "payment terms", "obligations"
    - Use clear, actionable language in your instructions
    # - For multiple documents, enable "Compare Documents" for insights  # COMMENTED OUT - Document comparison functionality disabled
    """)

def display_risk_assessment_interface(tagged_sections):
    """Display risk assessment interface"""
    st.subheader("⚠️ AI Legal Risk Assessment")
    st.markdown("**Identify and analyze legal risks with actionable recommendations**")
    
    # Risk assessment options
    col1, col2 = st.columns(2)
    
    with col1:
        document_type = st.selectbox(
            "Document Type:",
            ["contract", "agreement", "policy", "terms_of_service", "privacy_policy", "other"],
            help="Type of document being assessed"
        )
        
        assessment_focus = st.selectbox(
            "Assessment Focus:",
            ["comprehensive", "financial", "legal", "compliance", "operational"],
            help="Focus area for risk assessment"
        )
    
    with col2:
        include_recommendations = st.checkbox(
            "Include Recommendations",
            value=True,
            help="Include suggested actions for each identified risk"
        )
        
        # Risk categories
        risk_categories = st.multiselect(
            "Risk Categories:",
            ["financial", "legal", "compliance", "operational", "reputational", "technical"],
            default=["financial", "legal", "compliance", "operational"],
            help="Categories of risks to assess"
        )
    
    # Custom instructions
    st.subheader("📝 Assessment Instructions")
    default_instructions = {
        "comprehensive": "Provide a comprehensive risk assessment covering all major risk areas including financial, legal, compliance, and operational risks.",
        "financial": "Focus on financial risks including payment terms, liability exposure, cost implications, and financial obligations.",
        "legal": "Concentrate on legal risks including liability, indemnification, termination rights, and legal compliance issues.",
        "compliance": "Assess compliance risks including regulatory requirements, data protection, industry standards, and legal obligations.",
        "operational": "Evaluate operational risks including service delivery, performance obligations, resource requirements, and operational dependencies."
    }
    
    custom_instructions = st.text_area(
        "Custom Assessment Instructions:",
        value=default_instructions.get(assessment_focus, default_instructions["comprehensive"]),
        height=100,
        help="Customize the risk assessment focus and criteria"
    )
    
    # Generate risk assessment button
    if st.button("Assess Risks", type="primary"):
        if not custom_instructions.strip():
            st.error("Please provide assessment instructions")
        elif not risk_categories:
            st.error("Please select at least one risk category")
        else:
            with st.spinner("Analyzing risks..."):
                # Call risk assessment API
                # Convert tagged sections to the format expected by the API
                sections_data = []
                for section in tagged_sections:
                    if isinstance(section, dict):
                        sections_data.append(section)
                    else:
                        sections_data.append(section.model_dump())
                
                result = call_backend_api("/assess-risks/", json_data={
                    "tagged_sections": sections_data,
                    "document_type": document_type,
                    "assessment_focus": assessment_focus,
                    "risk_categories": risk_categories,
                    "include_recommendations": include_recommendations,
                    "custom_instructions": custom_instructions
                })
            
            if result:
                st.success("✅ Risk assessment completed!")
                
                # Display overall risk level
                overall_risk = result.get('overall_risk_level', 'Unknown')
                risk_color = {
                    'Low': '🟢',
                    'Medium': '🟡', 
                    'High': '🟠',
                    'Critical': '🔴'
                }.get(overall_risk, '⚪')
                
                st.subheader(f"{risk_color} Overall Risk Level: {overall_risk}")
                
                # Display risk summary
                risk_summary = result.get('risk_summary', {})
                if risk_summary:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Risks", risk_summary.get('total_risks', 0))
                    with col2:
                        st.metric("Critical", risk_summary.get('critical_risks', 0))
                    with col3:
                        st.metric("High", risk_summary.get('high_risks', 0))
                    with col4:
                        st.metric("Medium", risk_summary.get('medium_risks', 0))
                    with col5:
                        st.metric("Low", risk_summary.get('low_risks', 0))
                
                # Display individual risks
                risks = result.get('risks', [])
                if risks:
                    st.subheader("📋 Risk Assessment Table")
                    
                    # Create risk table
                    for i, risk in enumerate(risks):
                        with st.expander(f"Risk {i+1}: {risk['clause_type']} - {risk['severity']}", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Risk Description:**")
                                st.write(risk['risk_description'])
                                
                                if include_recommendations:
                                    st.write("**Suggested Action:**")
                                    st.write(risk['suggested_action'])
                                
                                st.write("**Legal Basis:**")
                                st.write(risk['legal_basis'])
                            
                            with col2:
                                # Severity indicator
                                severity_emoji = {
                                    'Low': '🟢',
                                    'Medium': '🟡',
                                    'High': '🟠', 
                                    'Critical': '🔴'
                                }.get(risk['severity'], '⚪')
                                
                                st.metric("Severity", f"{severity_emoji} {risk['severity']}")
                                st.metric("Score", f"{risk['severity_score']}/5")
                                st.metric("Impact Area", risk['impact_area'])
                                st.metric("Confidence", f"{risk['confidence_score']:.2f}")
                else:
                    st.info("No risks identified in the document.")
                
                # Processing metadata
                st.subheader("📊 Assessment Information")
                metadata = result.get('processing_metadata', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Type", metadata.get('document_type', 'Unknown'))
                with col2:
                    st.metric("Assessment Focus", metadata.get('assessment_focus', 'Unknown'))
                with col3:
                    st.metric("Sections Analyzed", metadata.get('total_sections_analyzed', 0))
            else:
                st.error("Failed to assess risks")
    
    # Instructions and tips
    st.subheader("💡 Tips for Better Risk Assessment")
    st.info("""
    **For comprehensive risk analysis:**
    - Be specific about your risk tolerance and focus areas
    - Mention key concerns like "liability exposure", "payment terms", "termination rights"
    - Include industry-specific risks if applicable
    - Consider both immediate and long-term risks
    - Review suggested actions carefully for implementation
    """)

def display_classification_interface(tagged_sections):
    """Display document classification interface"""
    st.subheader("🏷️ AI Document Classification")
    st.markdown("**Classify documents by type, subject, and importance level**")
    
    # Classification options
    col1, col2 = st.columns(2)
    
    with col1:
        document_type_hint = st.text_input(
            "Document Type Hint (Optional):",
            placeholder="e.g., Software License Agreement, Employment Contract, NDA",
            help="Provide a hint about what type of document this might be"
        )
    
    with col2:
        classification_focus = st.selectbox(
            "Classification Focus:",
            ["General classification", "Document type identification", "Subject area focus", "Risk assessment", "Compliance check"],
            help="Focus area for the classification process"
        )
    
    # Additional options
    st.subheader("📝 Classification Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True,
            help="Display confidence scores for each classification"
        )
        
        show_reasoning = st.checkbox(
            "Show Detailed Reasoning",
            value=True,
            help="Display detailed reasoning for each classification decision"
        )
    
    with col2:
        show_metadata = st.checkbox(
            "Show Processing Metadata",
            value=False,
            help="Display technical processing information"
        )
    
    # Generate classification button
    if st.button("Classify Document", type="primary"):
        with st.spinner("Analyzing document..."):
            # Convert tagged sections to the format expected by the API
            sections_data = []
            for section in tagged_sections:
                if isinstance(section, dict):
                    sections_data.append(section)
                else:
                    sections_data.append(section.model_dump())
            
            # Call classification API
            result = call_backend_api("/classify-document/", json_data={
                "tagged_sections": sections_data,
                "document_type_hint": document_type_hint if document_type_hint.strip() else None,
                "classification_focus": classification_focus
            })
        
        if result:
            st.success("✅ Document classification completed!")
            
            classification = result.get('classification', {})
            
            # Display main classification results
            st.subheader("📋 Classification Results")
            
            # Main classification cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                doc_type = classification.get('document_type', 'Unknown')
                st.metric("Document Type", doc_type)
            
            with col2:
                subject = classification.get('subject', 'Unknown')
                st.metric("Subject/Focus", subject)
            
            with col3:
                importance = classification.get('importance', 'Unknown')
                # Add color coding for importance
                importance_emoji = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(importance, '⚪')
                st.metric("Importance", f"{importance_emoji} {importance}")
            
            # Display reasoning if requested
            if show_reasoning:
                st.subheader("💭 Classification Reasoning")
                reasoning = classification.get('reasoning', {})
                
                for key, value in reasoning.items():
                    with st.expander(f"Reasoning for {key.replace('_', ' ').title()}", expanded=True):
                        st.write(value)
            
            # Display confidence scores if requested
            if show_confidence:
                st.subheader("🎯 Confidence Scores")
                confidence_scores = classification.get('confidence_scores', {})
                
                if confidence_scores:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        doc_type_conf = confidence_scores.get('document_type', 0)
                        st.metric("Document Type Confidence", f"{doc_type_conf:.2f}")
                    
                    with col2:
                        subject_conf = confidence_scores.get('subject', 0)
                        st.metric("Subject Confidence", f"{subject_conf:.2f}")
                    
                    with col3:
                        importance_conf = confidence_scores.get('importance', 0)
                        st.metric("Importance Confidence", f"{importance_conf:.2f}")
                    
                    # Overall confidence
                    overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                    st.metric("Overall Confidence", f"{overall_confidence:.2f}")
            
            # Display processing metadata if requested
            if show_metadata:
                st.subheader("📊 Processing Information")
                processing_metadata = result.get('processing_metadata', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sections Analyzed", processing_metadata.get('total_sections_analyzed', 0))
                
                with col2:
                    st.metric("Content Length", processing_metadata.get('document_content_length', 0))
                
                with col3:
                    st.metric("Processing Time", f"{processing_metadata.get('processing_time', 0):.2f}s")
                
                with col4:
                    st.metric("Classification Timestamp", result.get('classification_timestamp', 'N/A')[:19])
            
            # Display full classification details in an expandable section
            with st.expander("📄 Full Classification Details", expanded=False):
                st.json(result)
            
            # Classification insights and recommendations
            st.subheader("💡 Classification Insights")
            
            # Generate insights based on classification
            doc_type = classification.get('document_type', '')
            subject = classification.get('subject', '')
            importance = classification.get('importance', '')
            
            insights = []
            
            if importance == 'High':
                insights.append("🔴 **High Priority**: This document requires immediate attention and careful review.")
            elif importance == 'Medium':
                insights.append("🟡 **Medium Priority**: This document is important but can be handled as part of routine operations.")
            else:
                insights.append("🟢 **Low Priority**: This document is informational and can be processed as reference material.")
            
            if doc_type in ['Contract', 'Employment Agreement', 'Licensing Agreement']:
                insights.append("📋 **Contract Document**: Consider legal review and ensure all terms are clearly understood.")
            
            if subject in ['Payment Terms', 'Financial']:
                insights.append("💰 **Financial Impact**: Pay special attention to payment schedules, amounts, and financial obligations.")
            
            if subject in ['Confidentiality', 'Intellectual Property']:
                insights.append("🔒 **Confidentiality**: Ensure proper handling of sensitive information and IP rights.")
            
            if subject in ['Compliance', 'Regulatory']:
                insights.append("⚖️ **Compliance**: Verify adherence to relevant regulations and industry standards.")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("No specific insights available for this classification.")
            
        else:
            st.error("Failed to classify document")
    
    # Instructions and tips
    st.subheader("💡 Tips for Better Classification")
    st.info("""
    **For accurate classification:**
    - Provide a document type hint if you know what type of document it is
    - Choose the appropriate classification focus based on your needs
    - Review the reasoning to understand how the classification was made
    - Use confidence scores to assess the reliability of the classification
    - High importance documents should be prioritized for review
    """)

# def display_comparison_interface():
#     """Display document comparison interface"""
#     st.subheader("🔄 AI Document Comparison")
#     st.markdown("**Compare two PDF documents and identify differences in clauses, obligations, and timelines**")
    
#     # PDF file uploaders
#     st.subheader("📁 Upload Documents for Comparison")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("**First Document (Document A)**")
#         file_a = st.file_uploader(
#             "Upload first PDF:",
#             type=["pdf"],
#             key="comparison_file_a",
#             help="Upload the first PDF document to compare"
#         )
        
#         if file_a:
#             st.success(f"✅ {file_a.name} uploaded")
    
#     with col2:
#         st.write("**Second Document (Document B)**")
#         file_b = st.file_uploader(
#             "Upload second PDF:",
#             type=["pdf"],
#             key="comparison_file_b",
#             help="Upload the second PDF document to compare"
#         )
        
#         if file_b:
#             st.success(f"✅ {file_b.name} uploaded")
    
#     # Document titles (optional)
#     if file_a or file_b:
#         st.subheader("📝 Document Information")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             doc_a_title = st.text_input(
#                 "Document A Title (Optional):",
#                 value=file_a.name if file_a else "",
#                 help="Give the first document a descriptive title"
#             )
        
#         with col2:
#             doc_b_title = st.text_input(
#                 "Document B Title (Optional):",
#                 value=file_b.name if file_b else "",
#                 help="Give the second document a descriptive title"
#             )
    
#     # Comparison options
#     if file_a and file_b:
#         st.subheader("⚙️ Comparison Options")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             comparison_focus = st.selectbox(
#                 "Comparison Focus:",
#                 ["comprehensive", "clauses_only", "obligations_only", "timelines_only"],
#                 help="Choose what aspects to focus on during comparison"
#             )
            
#             risk_threshold = st.selectbox(
#                 "Risk Threshold:",
#                 ["Low", "Medium", "High", "Critical"],
#                 index=1,  # Default to Medium
#                 help="Minimum risk level to highlight in results"
#             )
        
#         with col2:
#             include_word_level = st.checkbox(
#                 "Include Word-Level Changes",
#                 value=False,
#                 help="Include detailed word-level changes for critical clauses"
#             )
            
#             show_metadata = st.checkbox(
#                 "Show Processing Metadata",
#                 value=False,
#                 help="Display technical processing information"
#             )
        
#         # Generate comparison button
#         if st.button("Compare Documents", type="primary"):
#             with st.spinner("Comparing documents..."):
#                 # Prepare files for API call
#                 files = {
#                     "file_a": ("document_a.pdf", file_a.getvalue(), "application/pdf"),
#                     "file_b": ("document_b.pdf", file_b.getvalue(), "application/pdf")
#                 }
                
#                 # Prepare query parameters
#                 params = {
#                     "comparison_focus": comparison_focus,
#                     "include_word_level": include_word_level,
#                     "risk_threshold": risk_threshold
#                 }
                
#                 # Add document titles if provided
#                 if doc_a_title.strip():
#                     params["document_a_title"] = doc_a_title.strip()
#                 if doc_b_title.strip():
#                     params["document_b_title"] = doc_b_title.strip()
                
#                 # Call comparison API
#                 try:
#                     response = requests.post(
#                         f"{BACKEND_URL}/compare-documents/",
#                         files=files,
#                         params=params,
#                         timeout=120
#                     )
                    
#                     if response.status_code == 200:
#                         result = response.json()
#                         st.success("✅ Document comparison completed!")
                        
#                         comparison = result.get('comparison', {})
                        
#                         # Display document info
#                         doc_info = result.get('document_info', {})
#                         if doc_info:
#                             st.subheader("📄 Documents Compared")
#                             col1, col2 = st.columns(2)
                            
#                             with col1:
#                                 doc_a_info = doc_info.get('document_a', {})
#                                 st.write(f"**Document A:** {doc_a_info.get('title', 'Unknown')}")
#                                 st.write(f"Filename: {doc_a_info.get('filename', 'Unknown')}")
#                                 st.write(f"Sections: {doc_a_info.get('sections_count', 0)}")
                            
#                             with col2:
#                                 doc_b_info = doc_info.get('document_b', {})
#                                 st.write(f"**Document B:** {doc_b_info.get('title', 'Unknown')}")
#                                 st.write(f"Filename: {doc_b_info.get('filename', 'Unknown')}")
#                                 st.write(f"Sections: {doc_b_info.get('sections_count', 0)}")
                        
#                         # Check if documents are identical
#                         processing_metadata = result.get('processing_metadata', {})
#                         documents_identical = processing_metadata.get('documents_identical', False)
                        
#                         if documents_identical:
#                             st.success("✅ Documents are identical - No differences found!")
#                             st.info("📄 The two uploaded documents contain exactly the same content.")
                            
#                             # Display comparison summary
#                             st.subheader("📋 Comparison Summary")
#                             st.write(comparison.get('comparison_summary', 'No summary available'))
                            
#                             # Overall risk assessment
#                             overall_risk = comparison.get('overall_risk_assessment', 'Unknown')
#                             st.subheader(f"🟢 Overall Risk Assessment: {overall_risk}")
                            
#                             # Show processing info
#                             st.subheader("📊 Processing Information")
#                             col1, col2, col3 = st.columns(3)
                            
#                             with col1:
#                                 st.metric("Documents Compared", processing_metadata.get('total_documents_compared', 0))
                            
#                             with col2:
#                                 st.metric("Total Differences", processing_metadata.get('total_differences', 0))
                            
#                             with col3:
#                                 st.metric("Processing Time", f"{processing_metadata.get('processing_time', 0):.2f}s")
                            
#                             st.info("💡 Since the documents are identical, no detailed comparison was performed.")
                            
#                         else:
#                             # Display comparison summary
#                             st.subheader("📋 Comparison Summary")
#                             st.write(comparison.get('comparison_summary', 'No summary available'))
                            
#                             # Overall risk assessment
#                             overall_risk = comparison.get('overall_risk_assessment', 'Unknown')
#                             risk_color = {
#                                 'Low': '🟢',
#                                 'Medium': '🟡',
#                                 'High': '🟠',
#                                 'Critical': '🔴'
#                             }.get(overall_risk.split()[0], '⚪')
                            
#                             st.subheader(f"{risk_color} Overall Risk Assessment: {overall_risk}")
                            
#                             # High-risk changes
#                             high_risk_changes = comparison.get('high_risk_changes', [])
#                             if high_risk_changes:
#                                 st.subheader("🚨 High-Risk Changes")
#                                 for i, change in enumerate(high_risk_changes):
#                                     with st.expander(f"High-Risk Change {i+1}: {change.get('title', 'Unknown')}", expanded=True):
#                                         col1, col2 = st.columns([3, 1])
                                        
#                                         with col1:
#                                             st.write("**Description:**")
#                                             st.write(change.get('description', 'No description'))
#                                             st.write("**Implications:**")
#                                             st.write(change.get('implications', 'No implications provided'))
                                        
#                                         with col2:
#                                             risk_level = change.get('risk_level', 'Unknown')
#                                             risk_emoji = {
#                                                 'Low': '🟢',
#                                                 'Medium': '🟡',
#                                                 'High': '🟠',
#                                                 'Critical': '🔴'
#                                             }.get(risk_level, '⚪')
                                            
#                                             st.metric("Risk Level", f"{risk_emoji} {risk_level}")
#                                             st.metric("Change Type", change.get('change_type', 'Unknown'))
                            
#                             # Clause differences
#                             clause_differences = comparison.get('clause_differences', [])
#                             if clause_differences:
#                                 st.subheader("📋 Clause Differences")
                                
#                                 for i, diff in enumerate(clause_differences):
#                                     change_type = diff.get('change_type', 'Unknown')
#                                     risk_level = diff.get('risk_level', 'Low')
                                    
#                                     # Color coding for change type
#                                     change_emoji = {
#                                         'Added': '➕',
#                                         'Deleted': '➖',
#                                         'Modified': '🔄',
#                                         'Unchanged': '➡️'
#                                     }.get(change_type, '❓')
                                    
#                                     risk_emoji = {
#                                         'Low': '🟢',
#                                         'Medium': '🟡',
#                                         'High': '🟠',
#                                         'Critical': '🔴'
#                                     }.get(risk_level, '⚪')
                                    
#                                     with st.expander(f"{change_emoji} {diff.get('clause_title', 'Unknown Clause')} - {risk_emoji} {risk_level}", expanded=False):
#                                         col1, col2 = st.columns(2)
                                        
#                                         with col1:
#                                             st.write("**Document A:**")
#                                             st.write(diff.get('document_a_content', 'Not specified'))
                                        
#                                         with col2:
#                                             st.write("**Document B:**")
#                                             st.write(diff.get('document_b_content', 'Not specified'))
                                        
#                                         st.write("**Reasoning:**")
#                                         st.write(diff.get('reasoning', 'No reasoning provided'))
                            
#                             # Obligation differences
#                             obligation_differences = comparison.get('obligation_differences', [])
#                             if obligation_differences:
#                                 st.subheader("👥 Obligation Differences")
                                
#                                 for i, diff in enumerate(obligation_differences):
#                                     change_type = diff.get('change_type', 'Unknown')
#                                     risk_level = diff.get('risk_level', 'Low')
#                                     party = diff.get('party', 'Unknown Party')
                                    
#                                     change_emoji = {
#                                         'Added': '➕',
#                                         'Deleted': '➖',
#                                         'Modified': '🔄',
#                                         'Unchanged': '➡️'
#                                     }.get(change_type, '❓')
                                    
#                                     risk_emoji = {
#                                         'Low': '🟢',
#                                         'Medium': '🟡',
#                                         'High': '🟠',
#                                         'Critical': '🔴'
#                                     }.get(risk_level, '⚪')
                                    
#                                     with st.expander(f"{change_emoji} {party} - {risk_emoji} {risk_level}", expanded=False):
#                                         col1, col2 = st.columns(2)
                                        
#                                         with col1:
#                                             st.write("**Document A Obligation:**")
#                                             st.write(diff.get('document_a_obligation', 'Not specified'))
                                        
#                                         with col2:
#                                             st.write("**Document B Obligation:**")
#                                             st.write(diff.get('document_b_obligation', 'Not specified'))
                                        
#                                         st.write("**Reasoning:**")
#                                         st.write(diff.get('reasoning', 'No reasoning provided'))
                            
#                             # Timeline differences
#                             timeline_differences = comparison.get('timeline_differences', [])
#                             if timeline_differences:
#                                 st.subheader("📅 Timeline Differences")
                                
#                                 for i, diff in enumerate(timeline_differences):
#                                     change_type = diff.get('change_type', 'Unknown')
#                                     risk_level = diff.get('risk_level', 'Low')
#                                     event = diff.get('event', 'Unknown Event')
                                    
#                                     change_emoji = {
#                                         'Added': '➕',
#                                         'Deleted': '➖',
#                                         'Modified': '🔄',
#                                         'Unchanged': '➡️'
#                                     }.get(change_type, '❓')
                                    
#                                     risk_emoji = {
#                                         'Low': '🟢',
#                                         'Medium': '🟡',
#                                         'High': '🟠',
#                                         'Critical': '🔴'
#                                     }.get(risk_level, '⚪')
                                    
#                                     with st.expander(f"{change_emoji} {event} - {risk_emoji} {risk_level}", expanded=False):
#                                         col1, col2 = st.columns(2)
                                        
#                                         with col1:
#                                             st.write("**Document A Timeline:**")
#                                             st.write(diff.get('document_a_timeline', 'Not specified'))
                                        
#                                         with col2:
#                                             st.write("**Document B Timeline:**")
#                                             st.write(diff.get('document_b_timeline', 'Not specified'))
                                        
#                                         st.write("**Reasoning:**")
#                                         st.write(diff.get('reasoning', 'No reasoning provided'))
                            
#                             # Processing metadata
#                             if show_metadata:
#                                 st.subheader("📊 Processing Information")
#                                 processing_metadata = result.get('processing_metadata', {})
                                
#                                 col1, col2, col3, col4 = st.columns(4)
                                
#                                 with col1:
#                                     st.metric("Documents Compared", processing_metadata.get('total_documents_compared', 0))
                                
#                                 with col2:
#                                     st.metric("Total Differences", processing_metadata.get('total_differences', 0))
                                
#                                 with col3:
#                                     st.metric("High-Risk Changes", processing_metadata.get('high_risk_count', 0))
                                
#                                 with col4:
#                                     st.metric("Processing Time", f"{processing_metadata.get('processing_time', 0):.2f}s")
                            
#                             # Full comparison details
#                             with st.expander("📄 Full Comparison Details", expanded=False):
#                                 st.json(result)
                    
#                     else:
#                         st.error(f"❌ Comparison failed with status {response.status_code}")
#                         st.error(f"Error: {response.text}")
                        
#                 except requests.exceptions.ConnectionError:
#                     st.error("❌ Cannot connect to backend server. Please make sure the FastAPI backend is running on http://localhost:8000")
#                 except requests.exceptions.Timeout:
#                     st.error("❌ Request timed out. The comparison is taking longer than expected.")
#                 except Exception as e:
#                     st.error(f"❌ Unexpected error: {str(e)}")
    
#     elif file_a or file_b:
#         st.warning("⚠️ Please upload both documents to enable comparison")
#         st.info("💡 Tip: Upload both PDF files to compare them")
    
#     else:
#         st.info("💡 Upload two PDF documents to compare them and identify differences")
    
#     # Instructions and tips
#     st.subheader("💡 Tips for Document Comparison")
#     st.info("""
#     **For effective document comparison:**
#     - Upload two PDF documents to compare
#     - Give each document a descriptive title for easy identification
#     - Choose the appropriate comparison focus based on your needs
#     - Review high-risk changes carefully as they may have legal/financial implications
#     - Use word-level changes for critical clauses that need detailed analysis
#     - Consider the risk threshold based on your organization's risk tolerance
#     """)

def get_severity_color(severity):
    """Get color for severity level"""
    colors = {
        'Low': 'green',
        'Medium': 'orange', 
        'High': 'red',
        'Critical': 'darkred'
    }
    return colors.get(severity, 'gray')

# Main file uploader
uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Convert UploadedFile to bytes for API calls
    file_bytes = uploaded_file.getvalue()
    # Determine MIME type based on file extension
    file_ext = uploaded_file.name.lower().split('.')[-1] if uploaded_file.name else 'pdf'
    mime_types = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg'
    }
    mime_type = mime_types.get(file_ext, 'application/pdf')
    files = {"file": (uploaded_file.name or f"document.{file_ext}", file_bytes, mime_type)}
    
    if analysis_mode == "📝 Text Extraction":
        st.subheader("📝 Document Text Extraction & Analysis")
        st.markdown("**Extract text from PDFs or images (PNG, JPG, JPEG) and analyze document structure**")
        
        with st.spinner("Extracting and tagging text..."):
            result = call_backend_api("/extract-text/", files=files)
        
        if result:
            st.success("✅ Text extraction completed")
            
            # Display tagged sections
            tagged_sections = result.get('tagged_sections', [])
            display_tagged_sections(tagged_sections)
            
            # Store in session state for other modes
            st.session_state.tagged_sections = tagged_sections
            
            # Show metadata
            st.subheader("📊 Processing Metadata")
            metadata = result.get('processing_metadata', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sections", result.get('total_sections', 0))
            with col2:
                st.metric("Text Length", result.get('raw_text_length', 0))
            with col3:
                st.metric("Pages Processed", "First page only")
    
    elif analysis_mode == "📋 Clause Analysis":
        st.subheader("📋 AI Clause Analysis")
        st.markdown("**Identify and analyze legal clauses with AI insights**")
        
        # Document type selection
        document_type = st.selectbox(
            "Document Type:",
            ["legal_document", "contract", "agreement", "policy", "terms_of_service", "privacy_policy", "other"],
            help="Type of document being analyzed"
        )
        
        with st.spinner("Analyzing clauses..."):
            # Use the document upload endpoint
            result = call_backend_api("/analyze-clauses/", files=files, params={"document_type": document_type})
        
        if result:
            st.success("✅ Clause analysis completed")
            display_clause_analysis(result.get('clause_analysis', []))
            
            # Show metadata
            st.subheader("📊 Processing Metadata")
            metadata = result.get('processing_metadata', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sections", result.get('total_sections', 0))
            with col2:
                st.metric("Input Type", "Document Upload")
    
    elif analysis_mode == "📅 Chronology Builder":
        st.subheader("📅 AI Chronology Builder")
        st.markdown("**Extract and organize chronological events from documents**")
        
        # Prepare parameters for the API call
        params = {}
        if document_date:
            params["document_date"] = document_date.isoformat()
        
        with st.spinner("Extracting chronology..."):
            # Use the document upload endpoint
            result = call_backend_api("/extract-chronology/", files=files, params=params)
        
        if result:
            st.success("✅ Chronology extraction completed")
            display_chronology_timeline(result)
            
            # Show metadata
            st.subheader("📊 Processing Metadata")
            metadata = result.get('processing_metadata', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sections", metadata.get('total_sections', 0))
            with col2:
                st.metric("Text Length", metadata.get('combined_text_length', 0))
            with col3:
                st.metric("Document Date", document_date.isoformat() if document_date else "Not provided")
    
    elif analysis_mode == "💬 Document Chat":
        st.subheader("💬 AI Document Chat Assistant")
        st.markdown("**Interactive Q&A with your document using advanced AI**")
        
        # Chat input
        user_input = st.text_input("Ask a question about the document:", placeholder="e.g., What are the key terms of this contract?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", type="primary")
        with col2:
            clear_button = st.button("Clear Chat")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if send_button and user_input:
            with st.spinner("Thinking..."):
                # Use the document upload endpoint
                result = call_backend_api("/chat-with-document/", files=files, params={
                    "user_message": user_input,
                    "document_type": "legal_document"
                })
            
            if result:
                st.success("✅ Response received")
                
                # Display the response
                st.subheader("🤖 AI Response")
                st.write(result.get('assistant_message', 'No response received'))
                
                # Show chat metadata
                st.subheader("📊 Chat Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Response Status", "Success")
                with col2:
                    st.metric("Document Type", "Document Upload")
            else:
                st.error("Failed to get response from the chat service")
    
    elif analysis_mode == "📄 Document Summarization":
        st.subheader("📄 AI Document Summarization")
        st.markdown("**Generate intelligent summaries with customizable instructions**")
        
        # Summarization options
        col1, col2 = st.columns(2)
        
        with col1:
            summary_type = st.selectbox(
                "Summary Type:",
                ["comprehensive", "executive", "bullet_points", "custom"],
                help="Choose the type of summary to generate"
            )
            
            max_length = st.slider(
                "Maximum Length (words):",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                help="Maximum number of words for each summary"
            )
        
        with col2:
            include_key_points = st.checkbox(
                "Include Key Points",
                value=True,
                help="Extract and display key points from each document"
            )
            
            compare_documents = False  # Always False since comparison is disabled
        
        # Summary instructions
        st.subheader("📝 Summary Instructions")
        default_instructions = {
            "comprehensive": "Provide a comprehensive summary covering all major sections and key information.",
            "executive": "Create an executive summary focusing on the most important points for decision-makers.",
            "bullet_points": "Summarize using bullet points and key takeaways.",
            "custom": "Customize your summary instructions below."
        }
        
        if summary_type == "custom":
            summary_instructions = st.text_area(
                "Custom Summary Instructions:",
                placeholder="e.g., Focus on legal risks, payment terms, and termination clauses...",
                height=100
            )
        else:
            summary_instructions = st.text_area(
                "Summary Instructions:",
                value=default_instructions[summary_type],
                height=100
            )
        
        # Generate summary button
        if st.button("Generate Summary", type="primary"):
            if not summary_instructions.strip():
                st.error("Please provide summary instructions")
            else:
                with st.spinner("Generating summaries..."):
                    # Use the document upload endpoint
                    result = call_backend_api("/summarize-documents/", files=files, params={
                        "summary_instructions": summary_instructions,
                        "summary_type": summary_type,
                        "max_length": max_length,
                        "include_key_points": include_key_points,
                        "compare_documents": compare_documents
                    })
                
                if result:
                    st.success("✅ Summary generated successfully!")
                    
                    # Display summaries
                    summaries = result.get('summaries', [])
                    for summary in summaries:
                        st.subheader(f"📄 {summary['title']}")
                        
                        # Summary content
                        st.write("**Summary:**")
                        st.write(summary['summary'])
                        
                        # Key points
                        if summary.get('key_points') and include_key_points:
                            st.write("**Key Points:**")
                            for point in summary['key_points']:
                                st.write(f"• {point}")
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", summary.get('word_count', 0))
                        with col2:
                            st.metric("Confidence", f"{summary.get('confidence_score', 0):.2f}")
                        with col3:
                            st.metric("Document Type", summary.get('document_type', 'Unknown'))
                        
                        st.write("---")
                    
                    # Processing metadata
                    st.subheader("📊 Processing Information")
                    metadata = result.get('processing_metadata', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Documents", result.get('total_documents', 0))
                    with col2:
                        st.metric("Processing Time", f"{result.get('total_processing_time', 0):.2f}s")
                    with col3:
                        st.metric("Summary Type", metadata.get('summary_type', 'Unknown'))
                else:
                    st.error("Failed to generate summary")
    
    elif analysis_mode == "⚠️ Risk Assessment":
        st.subheader("⚠️ AI Legal Risk Assessment")
        st.markdown("**Identify and analyze legal risks with actionable recommendations**")
        
        # Risk assessment options
        col1, col2 = st.columns(2)
        
        with col1:
            document_type = st.selectbox(
                "Document Type:",
                ["contract", "agreement", "policy", "terms_of_service", "privacy_policy", "other"],
                help="Type of document being assessed"
            )
            
            assessment_focus = st.selectbox(
                "Assessment Focus:",
                ["comprehensive", "financial", "legal", "compliance", "operational"],
                help="Focus area for risk assessment"
            )
        
        with col2:
            include_recommendations = st.checkbox(
                "Include Recommendations",
                value=True,
                help="Include suggested actions for each identified risk"
            )
            
            # Risk categories
            risk_categories = st.multiselect(
                "Risk Categories:",
                ["financial", "legal", "compliance", "operational", "reputational", "technical"],
                default=["financial", "legal", "compliance", "operational"],
                help="Categories of risks to assess"
            )
        
        # Custom instructions
        st.subheader("📝 Assessment Instructions")
        default_instructions = {
            "comprehensive": "Provide a comprehensive risk assessment covering all major risk areas including financial, legal, compliance, and operational risks.",
            "financial": "Focus on financial risks including payment terms, liability exposure, cost implications, and financial obligations.",
            "legal": "Concentrate on legal risks including liability, indemnification, termination rights, and legal compliance issues.",
            "compliance": "Assess compliance risks including regulatory requirements, data protection, industry standards, and legal obligations.",
            "operational": "Evaluate operational risks including service delivery, performance obligations, resource requirements, and operational dependencies."
        }
        
        custom_instructions = st.text_area(
            "Custom Assessment Instructions:",
            value=default_instructions.get(assessment_focus, default_instructions["comprehensive"]),
            height=100,
            help="Customize the risk assessment focus and criteria"
        )
        
        # Generate risk assessment button
        if st.button("Assess Risks", type="primary"):
            if not custom_instructions.strip():
                st.error("Please provide assessment instructions")
            elif not risk_categories:
                st.error("Please select at least one risk category")
            else:
                with st.spinner("Analyzing risks..."):
                    # Use the document upload endpoint
                    result = call_backend_api("/assess-risks/", files=files, params={
                        "document_type": document_type,
                        "assessment_focus": assessment_focus,
                        "risk_categories": ",".join(risk_categories),
                        "include_recommendations": include_recommendations,
                        "custom_instructions": custom_instructions
                    })
                
                if result:
                    st.success("✅ Risk assessment completed!")
                    
                    # Display overall risk level
                    overall_risk = result.get('overall_risk_level', 'Unknown')
                    risk_color = {
                        'Low': '🟢',
                        'Medium': '🟡', 
                        'High': '🟠',
                        'Critical': '🔴'
                    }.get(overall_risk, '⚪')
                    
                    st.subheader(f"{risk_color} Overall Risk Level: {overall_risk}")
                    
                    # Display risk summary
                    risk_summary = result.get('risk_summary', {})
                    if risk_summary:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Risks", risk_summary.get('total_risks', 0))
                        with col2:
                            st.metric("Critical", risk_summary.get('critical_risks', 0))
                        with col3:
                            st.metric("High", risk_summary.get('high_risks', 0))
                        with col4:
                            st.metric("Medium", risk_summary.get('medium_risks', 0))
                        with col5:
                            st.metric("Low", risk_summary.get('low_risks', 0))
                    
                    # Display individual risks
                    risks = result.get('risks', [])
                    if risks:
                        st.subheader("📋 Risk Assessment Table")
                        
                        # Create risk table
                        for i, risk in enumerate(risks):
                            with st.expander(f"Risk {i+1}: {risk['clause_type']} - {risk['severity']}", expanded=True):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write("**Risk Description:**")
                                    st.write(risk['risk_description'])
                                    
                                    if include_recommendations:
                                        st.write("**Suggested Action:**")
                                        st.write(risk['suggested_action'])
                                    
                                    st.write("**Legal Basis:**")
                                    st.write(risk['legal_basis'])
                                
                                with col2:
                                    # Severity indicator
                                    severity_emoji = {
                                        'Low': '🟢',
                                        'Medium': '🟡',
                                        'High': '🟠', 
                                        'Critical': '🔴'
                                    }.get(risk['severity'], '⚪')
                                    
                                    st.metric("Severity", f"{severity_emoji} {risk['severity']}")
                                    st.metric("Score", f"{risk['severity_score']}/5")
                                    st.metric("Impact Area", risk['impact_area'])
                                    st.metric("Confidence", f"{risk['confidence_score']:.2f}")
                    else:
                        st.info("No risks identified in the document.")
                    
                    # Processing metadata
                    st.subheader("📊 Assessment Information")
                    metadata = result.get('processing_metadata', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Document Type", metadata.get('document_type', 'Unknown'))
                    with col2:
                        st.metric("Assessment Focus", metadata.get('assessment_focus', 'Unknown'))
                    with col3:
                        st.metric("Sections Analyzed", metadata.get('total_sections_analyzed', 0))
                else:
                    st.error("Failed to assess risks")
    
    elif analysis_mode == "🏷️ Document Classification":
        st.subheader("🏷️ AI Document Classification")
        st.markdown("**Classify documents by type, subject, and importance level**")
        
        # Classification options
        col1, col2 = st.columns(2)
        
        with col1:
            document_type_hint = st.text_input(
                "Document Type Hint (Optional):",
                placeholder="e.g., Software License Agreement, Employment Contract, NDA",
                help="Provide a hint about what type of document this might be"
            )
        
        with col2:
            classification_focus = st.selectbox(
                "Classification Focus:",
                ["General classification", "Document type identification", "Subject area focus", "Risk assessment", "Compliance check"],
                help="Focus area for the classification process"
            )
        
        # Additional options
        st.subheader("📝 Classification Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_confidence = st.checkbox(
                "Show Confidence Scores",
                value=True,
                help="Display confidence scores for each classification"
            )
            
            show_reasoning = st.checkbox(
                "Show Detailed Reasoning",
                value=True,
                help="Display detailed reasoning for each classification decision"
            )
        
        with col2:
            show_metadata = st.checkbox(
                "Show Processing Metadata",
                value=False,
                help="Display technical processing information"
            )
        
        # Generate classification button
        if st.button("Classify Document", type="primary"):
            with st.spinner("Analyzing document..."):
                # Use the document upload endpoint
                result = call_backend_api("/classify-document/", files=files, params={
                    "document_type_hint": document_type_hint if document_type_hint.strip() else None,
                    "classification_focus": classification_focus
                })
            
            if result:
                st.success("✅ Document classification completed!")
                
                classification = result.get('classification', {})
                
                # Display main classification results
                st.subheader("📋 Classification Results")
                
                # Main classification cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    doc_type = classification.get('document_type', 'Unknown')
                    st.metric("Document Type", doc_type)
                
                with col2:
                    subject = classification.get('subject', 'Unknown')
                    st.metric("Subject/Focus", subject)
                
                with col3:
                    importance = classification.get('importance', 'Unknown')
                    # Add color coding for importance
                    importance_emoji = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }.get(importance, '⚪')
                    st.metric("Importance", f"{importance_emoji} {importance}")
                
                # Display reasoning if requested
                if show_reasoning:
                    st.subheader("💭 Classification Reasoning")
                    reasoning = classification.get('reasoning', {})
                    
                    for key, value in reasoning.items():
                        with st.expander(f"Reasoning for {key.replace('_', ' ').title()}", expanded=True):
                            st.write(value)
                
                # Display confidence scores if requested
                if show_confidence:
                    st.subheader("🎯 Confidence Scores")
                    confidence_scores = classification.get('confidence_scores', {})
                    
                    if confidence_scores:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            doc_type_conf = confidence_scores.get('document_type', 0)
                            st.metric("Document Type Confidence", f"{doc_type_conf:.2f}")
                        
                        with col2:
                            subject_conf = confidence_scores.get('subject', 0)
                            st.metric("Subject Confidence", f"{subject_conf:.2f}")
                        
                        with col3:
                            importance_conf = confidence_scores.get('importance', 0)
                            st.metric("Importance Confidence", f"{importance_conf:.2f}")
                        
                        # Overall confidence
                        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                        st.metric("Overall Confidence", f"{overall_confidence:.2f}")
                
                # Display processing metadata if requested
                if show_metadata:
                    st.subheader("📊 Processing Information")
                    processing_metadata = result.get('processing_metadata', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sections Analyzed", processing_metadata.get('total_sections_analyzed', 0))
                    
                    with col2:
                        st.metric("Content Length", processing_metadata.get('document_content_length', 0))
                    
                    with col3:
                        st.metric("Processing Time", f"{processing_metadata.get('processing_time', 0):.2f}s")
                    
                    with col4:
                        st.metric("Classification Timestamp", result.get('classification_timestamp', 'N/A')[:19])
                
                # Display full classification details in an expandable section
                with st.expander("📄 Full Classification Details", expanded=False):
                    st.json(result)
                
                # Classification insights and recommendations
                st.subheader("💡 Classification Insights")
                
                # Generate insights based on classification
                doc_type = classification.get('document_type', '')
                subject = classification.get('subject', '')
                importance = classification.get('importance', '')
                
                insights = []
                
                if importance == 'High':
                    insights.append("🔴 **High Priority**: This document requires immediate attention and careful review.")
                elif importance == 'Medium':
                    insights.append("🟡 **Medium Priority**: This document is important but can be handled as part of routine operations.")
                else:
                    insights.append("🟢 **Low Priority**: This document is informational and can be processed as reference material.")
                
                if doc_type in ['Contract', 'Employment Agreement', 'Licensing Agreement']:
                    insights.append("📋 **Contract Document**: Consider legal review and ensure all terms are clearly understood.")
                
                if subject in ['Payment Terms', 'Financial']:
                    insights.append("💰 **Financial Impact**: Pay special attention to payment schedules, amounts, and financial obligations.")
                
                if subject in ['Confidentiality', 'Intellectual Property']:
                    insights.append("🔒 **Confidentiality**: Ensure proper handling of sensitive information and IP rights.")
                
                if subject in ['Compliance', 'Regulatory']:
                    insights.append("⚖️ **Compliance**: Verify adherence to relevant regulations and industry standards.")
                
                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.info("No specific insights available for this classification.")
                
            else:
                st.error("Failed to classify document")
    
    # elif analysis_mode == "🔄 Document Comparison":
    #     st.subheader("🔄 AI Document Comparison")
    #     st.markdown("**Compare two PDF documents and identify differences in clauses, obligations, and timelines**")
    #     display_comparison_interface()

# Footer
st.markdown("---")
st.markdown("### 🔗 API Information")
st.markdown("**Backend Server**: http://localhost:8000 | **Interactive API Docs**: http://localhost:8000/docs")

st.markdown("### 📋 Available API Endpoints")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Document Processing:**
    - `POST /extract-text/` - Extract text from PDF documents or images (PNG, JPG, JPEG)
    - `POST /tag-documents/` - Analyze document structure and tag sections
    
    **Analysis Features:**
    - `POST /extract-chronology/` - Extract chronological events
    - `POST /chat-with-document/` - Chat with AI about document content
    - `POST /summarize-documents/` - Generate document summaries
    - `POST /assess-risks/` - Assess legal risks with recommendations
    """)

with col2:
    st.markdown("""
    **Advanced Features:**
    - `POST /classify-document/` - Classify documents by type and importance
    
    **System:**
    - `GET /health/` - Check system health
    - `GET /docs` - Interactive API documentation
    - `GET /` - API information
    """)

st.markdown("### 🚀 Getting Started")
st.markdown("""
1. **Upload a PDF or image** (PNG, JPG, JPEG) using any analysis mode
2. **Configure options** for your specific needs (if applicable)
3. **Click the analysis button** to process your document
4. **View results** with detailed insights and recommendations

**💡 Pro Tip**: You can now upload a PDF or image directly to any analysis mode - no need to extract text first! Each mode will automatically process your document and provide specialized analysis.
""")
