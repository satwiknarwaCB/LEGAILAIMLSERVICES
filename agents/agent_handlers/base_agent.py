# agent_handlers/base_agent.py - Base class for all agent handlers

import os
import logging
import httpx
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global HTTP client - Will be initialized on first use
_httpx_client: Optional[httpx.AsyncClient] = None

def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=120.0)
    return _httpx_client

class BaseAgent(ABC):
    """Base class for all agent handlers"""
    
    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.llm = None
    
    def _initialize_llm(self):
        """Initialize the LLM based on environment"""
        if self.llm is not None:
            return self.llm
        
        app_env = os.getenv('APP_ENV', 'local')
        
        if app_env == 'production':
            from langchain_groq import ChatGroq
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            self.llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
                groq_api_key=groq_api_key
            )
        else:
            from langchain_ollama import ChatOllama
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.25:11434')
            model = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')
            
            self.llm = ChatOllama(
                model=model,
                temperature=0.3,
                base_url=base_url,
                timeout=120
            )
        
        return self.llm
    
    @abstractmethod
    async def process_message(
        self, 
        message: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a message and return response"""
        pass
    
    async def process_document(
        self, 
        filename: str, 
        file_content: bytes,
        session_id: str
    ) -> Dict[str, Any]:
        """Process an uploaded document (for document-based agents)"""
        return {
            "success": False,
            "error": f"Agent {self.agent_id} does not support document upload"
        }
    
    def build_prompt(self, system_prompt: str, user_message: str, context: str = "") -> str:
        """Build a prompt with system context"""
        prompt = f"""{system_prompt}

{f"CONTEXT:{chr(10)}{context}{chr(10)}" if context else ""}

USER MESSAGE: {user_message}

YOUR RESPONSE:"""
        return prompt
