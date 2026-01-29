# agents_api.py - Main Agents API Service
# This is the unified gateway for all agents

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import our modules
from agent_registry import get_all_agents, get_agent, is_valid_agent
from session_manager import get_session_manager, AgentSession
from agent_handlers import (
    AskAgent,
    DraftAgent,
    InteractAgent,
    ResearchAgent,
    ComplianceAgent,
    ReviewAgent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# JWT Configuration (same as other services)
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-super-secret-jwt-key-change-this-in-production')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')

# Initialize FastAPI
app = FastAPI(
    title="Legal AI Agents API",
    description="Unified API for all legal AI agents",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent instances (singleton)
_agent_instances: Dict[str, Any] = {}

def get_agent_instance(agent_id: str):
    """Get or create agent instance"""
    global _agent_instances
    
    if agent_id not in _agent_instances:
        if agent_id == "ask":
            _agent_instances[agent_id] = AskAgent()
        elif agent_id == "draft":
            _agent_instances[agent_id] = DraftAgent()
        elif agent_id == "interact":
            _agent_instances[agent_id] = InteractAgent()
        elif agent_id == "research":
            _agent_instances[agent_id] = ResearchAgent()
        elif agent_id == "compliance":
            _agent_instances[agent_id] = ComplianceAgent()
        elif agent_id == "review":
            _agent_instances[agent_id] = ReviewAgent()
        else:
            return None
    
    return _agent_instances.get(agent_id)


# ==================== AUTH ====================

async def verify_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Verify JWT token and return user_id (optional auth)"""
    if not authorization:
        return None
    
    try:
        import jwt
        
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("email") or payload.get("sub")
        return user_id
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception:
        return None


# ==================== PYDANTIC MODELS ====================

class AgentChatRequest(BaseModel):
    session_id: str
    message: str
    context: Optional[Dict[str, Any]] = None

class AgentChatResponse(BaseModel):
    success: bool
    agent_id: str
    session_id: str
    response: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    tokens_used: Optional[int] = None
    timestamp: str
    error: Optional[str] = None

class SessionResponse(BaseModel):
    success: bool
    session_id: str
    agent_id: str
    agent_name: str
    created_at: str

class DocumentUploadResponse(BaseModel):
    success: bool
    session_id: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    char_count: Optional[int] = None
    chunk_count: Optional[int] = None
    preview: Optional[str] = None
    error: Optional[str] = None


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Legal AI Agents API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "list_agents": "GET /api/agents",
            "create_session": "POST /api/agents/{agent_id}/session",
            "chat": "POST /api/agents/{agent_id}/chat",
            "history": "GET /api/agents/{agent_id}/session/{session_id}/history",
            "upload": "POST /api/agents/{agent_id}/upload",
            "end_session": "DELETE /api/agents/{agent_id}/session/{session_id}"
        }
    }


@app.get("/api/agents")
async def list_agents():
    """1️⃣ List all available agents"""
    return {
        "success": True,
        "agents": get_all_agents()
    }


@app.post("/api/agents/{agent_id}/session", response_model=SessionResponse)
async def create_session(
    agent_id: str,
    user_id: Optional[str] = Depends(verify_token)
):
    """2️⃣ Start/Create agent session"""
    # Validate agent
    agent_def = get_agent(agent_id)
    if not agent_def:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    # Create session
    session_mgr = get_session_manager()
    session = session_mgr.create_session(
        agent_id=agent_id,
        agent_name=agent_def.name,
        user_id=user_id
    )
    
    logger.info(f"Created session {session.session_id} for agent {agent_id}")
    
    return SessionResponse(
        success=True,
        session_id=session.session_id,
        agent_id=agent_id,
        agent_name=agent_def.name,
        created_at=session.created_at
    )


@app.post("/api/agents/{agent_id}/chat", response_model=AgentChatResponse)
async def chat_with_agent(
    agent_id: str,
    request: AgentChatRequest,
    user_id: Optional[str] = Depends(verify_token)
):
    """3️⃣ Chat with agent"""
    # Validate agent
    if not is_valid_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    # Validate session
    session_mgr = get_session_manager()
    session = session_mgr.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Create a session first.")
    
    if session.agent_id != agent_id:
        raise HTTPException(status_code=400, detail=f"Session belongs to agent '{session.agent_id}', not '{agent_id}'")
    
    # Get agent instance
    agent = get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=500, detail="Failed to initialize agent")
    
    # Add context
    context = request.context or {}
    if user_id:
        context["user_id"] = user_id
    
    # Process message
    try:
        # Add user message to session
        session_mgr.add_message(request.session_id, "user", request.message)
        
        # Get response from agent
        result = await agent.process_message(
            message=request.message,
            session_id=request.session_id,
            context=context
        )
        
        if result.get("success"):
            # Add assistant message to session
            response_text = result.get("response", "")
            session_mgr.add_message(request.session_id, "assistant", response_text)
            
            return AgentChatResponse(
                success=True,
                agent_id=agent_id,
                session_id=request.session_id,
                response=response_text,
                sources=result.get("sources", []),
                tokens_used=result.get("tokens_used", 0),
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        else:
            return AgentChatResponse(
                success=False,
                agent_id=agent_id,
                session_id=request.session_id,
                error=result.get("error", "Unknown error"),
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return AgentChatResponse(
            success=False,
            agent_id=agent_id,
            session_id=request.session_id,
            error=str(e),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )


@app.get("/api/agents/{agent_id}/session/{session_id}/history")
async def get_session_history(
    agent_id: str,
    session_id: str,
    user_id: Optional[str] = Depends(verify_token)
):
    """4️⃣ Get agent session history"""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.agent_id != agent_id:
        raise HTTPException(status_code=400, detail=f"Session belongs to agent '{session.agent_id}'")
    
    messages = [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp
        }
        for msg in session.messages
    ]
    
    return {
        "success": True,
        "session_id": session_id,
        "agent_id": agent_id,
        "messages": messages,
        "total_messages": len(messages)
    }


@app.delete("/api/agents/{agent_id}/session/{session_id}")
async def end_session(
    agent_id: str,
    session_id: str,
    user_id: Optional[str] = Depends(verify_token)
):
    """5️⃣ End/Delete agent session"""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.agent_id != agent_id:
        raise HTTPException(status_code=400, detail=f"Session belongs to agent '{session.agent_id}'")
    
    success = session_mgr.delete_session(session_id)
    
    if success:
        return {"success": True, "message": "Agent session ended successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to end session")


@app.post("/api/agents/{agent_id}/upload", response_model=DocumentUploadResponse)
async def upload_document(
    agent_id: str,
    file: UploadFile = File(...),
    session_id: str = None,
    user_id: Optional[str] = Depends(verify_token)
):
    """6️⃣ Upload document to agent (for interact/review agents)"""
    # Validate agent
    agent_def = get_agent(agent_id)
    if not agent_def:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    if not agent_def.requires_document and agent_id not in ["interact", "review"]:
        raise HTTPException(status_code=400, detail=f"Agent '{agent_id}' does not support document upload")
    
    # Get or create session
    session_mgr = get_session_manager()
    if session_id:
        session = session_mgr.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = session_mgr.create_session(
            agent_id=agent_id,
            agent_name=agent_def.name,
            user_id=user_id
        )
        session_id = session.session_id
    
    # Get agent and process document
    agent = get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=500, detail="Failed to initialize agent")
    
    try:
        file_content = await file.read()
        ext = file.filename.split('.')[-1].lower() if file.filename else "unknown"
        
        result = await agent.process_document(
            filename=file.filename,
            file_content=file_content,
            session_id=session_id
        )
        
        if result.get("success"):
            # Update session with document info
            session_mgr.set_document(
                session_id=session_id,
                document_id=result.get("document_id", session_id),
                filename=file.filename
            )
            
            return DocumentUploadResponse(
                success=True,
                session_id=session_id,
                document_id=result.get("document_id"),
                filename=file.filename,
                file_type=ext,
                char_count=result.get("char_count"),
                chunk_count=result.get("chunk_count"),
                preview=result.get("preview")
            )
        else:
            return DocumentUploadResponse(
                success=False,
                session_id=session_id,
                error=result.get("error", "Document processing failed")
            )
            
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        return DocumentUploadResponse(
            success=False,
            session_id=session_id,
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agents",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ==================== LIFECYCLE ====================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Agents API Service...")
    # Pre-initialize agents if needed

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Agents API Service...")
    from agent_handlers.base_agent import httpx_client
    if httpx_client:
        await httpx_client.aclose()


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
