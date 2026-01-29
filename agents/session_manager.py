# session_manager.py - Manages agent sessions

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class AgentMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str


class AgentSession(BaseModel):
    session_id: str
    agent_id: str
    agent_name: str
    user_id: Optional[str] = None
    created_at: str
    messages: List[AgentMessage] = []
    document_id: Optional[str] = None
    document_filename: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SessionManager:
    """Manages agent sessions in memory"""
    
    def __init__(self):
        # session_id -> AgentSession
        self._sessions: Dict[str, AgentSession] = {}
        
        # user_id -> list of session_ids
        self._user_sessions: Dict[str, List[str]] = {}
    
    def create_session(
        self, 
        agent_id: str, 
        agent_name: str, 
        user_id: Optional[str] = None
    ) -> AgentSession:
        """Create a new agent session"""
        session_id = f"agent_session_{uuid.uuid4().hex[:12]}"
        
        session = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            user_id=user_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            messages=[],
            metadata={}
        )
        
        self._sessions[session_id] = session
        
        # Track by user
        user_key = user_id or "anonymous"
        if user_key not in self._user_sessions:
            self._user_sessions[user_key] = []
        self._user_sessions[user_key].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID"""
        return self._sessions.get(session_id)
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str
    ) -> Optional[AgentMessage]:
        """Add a message to a session"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        message = AgentMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            role=role,
            content=content,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        session.messages.append(message)
        return message
    
    def get_messages(self, session_id: str) -> List[AgentMessage]:
        """Get all messages from a session"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.messages
    
    def set_document(
        self, 
        session_id: str, 
        document_id: str, 
        filename: str
    ) -> bool:
        """Set document for a session (for document-based agents)"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.document_id = document_id
        session.document_filename = filename
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Remove from user sessions
        user_key = session.user_id or "anonymous"
        if user_key in self._user_sessions:
            if session_id in self._user_sessions[user_key]:
                self._user_sessions[user_key].remove(session_id)
        
        # Remove session
        del self._sessions[session_id]
        return True
    
    def get_user_sessions(
        self, 
        user_id: Optional[str], 
        agent_id: Optional[str] = None
    ) -> List[AgentSession]:
        """Get all sessions for a user, optionally filtered by agent"""
        user_key = user_id or "anonymous"
        session_ids = self._user_sessions.get(user_key, [])
        
        sessions = []
        for sid in session_ids:
            session = self._sessions.get(sid)
            if session:
                if agent_id is None or session.agent_id == agent_id:
                    sessions.append(session)
        
        return sessions
    
    def update_metadata(
        self, 
        session_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update session metadata"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.metadata.update(metadata)
        return True


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
