# agent_handlers/draft_agent.py - Draft Agent - Routes to existing drafting service

import httpx
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, get_httpx_client

logger = logging.getLogger(__name__)

# Backend service URL (using 127.0.0.1 for faster Windows resolution)
ASK_DRAFT_URL = "http://127.0.0.1:8000"


class DraftAgent(BaseAgent):
    """Draft Agent - Routes to existing ASK/DRAFT drafting service"""
    
    def __init__(self):
        super().__init__("draft", "Draft Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process drafting request by calling the existing drafting API
        """
        try:
            context = context or {}
            
            # Extract drafting parameters from context or message
            doc_type = context.get("doc_type", "Legal Document")
            style = context.get("style", "Formal Legal")
            length = context.get("length", "Standard")
            clauses = context.get("clauses", [])
            special_provisions = context.get("special_provisions", "")
            
            # Prepare request to existing drafting API
            payload = {
                "doc_type": doc_type,
                "requirements": message,
                "style": style,
                "length": length,
                "clauses": clauses,
                "special_provisions": special_provisions
            }
            
            # Get auth token if provided
            headers = {"Content-Type": "application/json"}
            if context.get("auth_token"):
                headers["Authorization"] = context["auth_token"]
            
            # Call existing drafting API using shared client
            response = await get_httpx_client().post(
                f"{ASK_DRAFT_URL}/api/drafting/generate",
                json=payload,
                headers=headers
            )
                
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("document", ""),
                    "doc_type": data.get("doc_type", doc_type),
                    "style": data.get("style", style),
                    "word_count": data.get("word_count", 0),
                    "tokens_used": data.get("tokens_used", 0),
                    "metadata": data.get("metadata", {})
                }
            else:
                logger.error(f"Drafting API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Drafting service error: {response.status_code}"
                }
                    
        except httpx.TimeoutException:
            logger.error("Timeout calling drafting service")
            return {
                "success": False,
                "error": "Request timed out. Document generation takes time, please try again."
            }
        except Exception as e:
            logger.error(f"Draft agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
