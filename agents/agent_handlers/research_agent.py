# agent_handlers/research_agent.py - Research Agent - Legal research and case law

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Research Agent - Deep dive into case law and legal precedents"""
    
    def __init__(self):
        super().__init__("research", "Research Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process legal research request using LLM
        """
        try:
            llm = self._initialize_llm()
            
            prompt = f"""You are a legal research expert specializing in Indian law and international legal precedents.

Your task is to research and provide comprehensive information on the following legal query:

RESEARCH QUERY: {message}

Please provide:
1. **Relevant Case Law**: Key cases that relate to this topic with case names, years, and brief summaries
2. **Legal Precedents**: Important precedents established by courts
3. **Statutory Provisions**: Relevant sections from applicable Acts and laws
4. **Legal Principles**: Key legal principles that apply to this situation
5. **Recent Developments**: Any recent changes in law or notable judgments (if applicable)

Format your response clearly with sections and bullet points.
Cite case names in proper legal format (e.g., "State of Maharashtra v. XYZ, 2020 SCC...")

Important: If you're not certain about specific case citations, provide general legal principles instead of making up case names.

YOUR RESEARCH RESPONSE:"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "success": True,
                "response": answer,
                "sources": [{"type": "legal_research", "reference": "AI Legal Research"}],
                "tokens_used": 0  # Would need proper token counting
            }
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
