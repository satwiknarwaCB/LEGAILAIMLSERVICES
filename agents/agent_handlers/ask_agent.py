# agent_handlers/ask_agent.py - Ask Agent - Routes to existing chatbot

import os
import httpx
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, get_httpx_client

logger = logging.getLogger(__name__)

# Backend service URL (mapped via environment for Docker support)
ASK_DRAFT_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000")


class AskAgent(BaseAgent):
    """Ask Agent - Routes to existing ASK/DRAFT chatbot service"""
    
    def __init__(self):
        super().__init__("ask", "Ask Agent")
    
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process message by calling the existing chatbot API
        """
        try:
            context = context or {}
            
            # Determine chatbot mode
            chatbot_mode = "Document Only"
            if context.get("include_case_law"):
                chatbot_mode = "Hybrid (Smart)"
            elif context.get("layman_mode"):
                chatbot_mode = "Layman Explanation"
            
            # Prepare request to existing chatbot
            payload = {
                "message": message,
                "chatbot_mode": chatbot_mode,
                "chat_session_id": session_id,
                "layman_mode": context.get("layman_mode", False)
            }
            
            # Get auth token if provided
            headers = {"Content-Type": "application/json"}
            if context.get("auth_token"):
                headers["Authorization"] = context["auth_token"]
            
            # Call existing chatbot API using shared client
            response = await get_httpx_client().post(
                f"{ASK_DRAFT_URL}/api/chat",
                json=payload,
                headers=headers
            )
                
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("answer", ""),
                    "sources": data.get("sources", []),
                    "tokens_used": data.get("tokens_used", 0)
                }
            else:
                logger.error(f"Chatbot API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Chatbot service error: {response.status_code}"
                }
                    
        except httpx.TimeoutException:
            logger.error("Timeout calling chatbot service")
            return {
                "success": False,
                "error": "Request timed out. Please try again."
            }
        except Exception as e:
            logger.error(f"Ask agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
