"""
Conversation Context Manager for ScreenMonitorMCP
Provides conversation history and context preservation
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from .cache_manager import get_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Single conversation message"""
    id: str
    timestamp: float
    role: str  # 'user' or 'assistant'
    content: str
    context_data: Optional[Dict[str, Any]] = None
    screenshot_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'role': self.role,
            'content': self.content,
            'context_data': self.context_data,
            'screenshot_id': self.screenshot_id
        }

@dataclass
class ConversationSession:
    """Conversation session with history"""
    id: str
    created_at: float
    last_activity: float
    messages: List[ConversationMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, context_data: Optional[Dict[str, Any]] = None, screenshot_id: Optional[str] = None):
        """Add message to conversation"""
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            role=role,
            content=content,
            context_data=context_data,
            screenshot_id=screenshot_id
        )
        self.messages.append(message)
        self.last_activity = time.time()
        return message
    
    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """Get recent messages"""
        return self.messages[-limit:] if self.messages else []
    
    def get_context_summary(self) -> str:
        """Get conversation context summary"""
        if not self.messages:
            return "No conversation history"
        
        recent_messages = self.get_recent_messages(5)
        summary_parts = []
        
        for msg in recent_messages:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{role_prefix}: {content_preview}")
        
        return "\n".join(summary_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'messages': [msg.to_dict() for msg in self.messages],
            'context': self.context
        }
class ConversationManager:
    """Manages conversation sessions and context"""
    
    def __init__(self, session_timeout_hours: int = 24):
        self.session_timeout = session_timeout_hours * 3600  # Convert to seconds
        self.cache_manager = get_cache_manager()
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        logger.info(f"Conversation manager initialized - session timeout: {session_timeout_hours} hours")
    
    def create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Create new conversation session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session = ConversationSession(
            id=session_id,
            created_at=time.time(),
            last_activity=time.time()
        )
        
        self.active_sessions[session_id] = session
        self._cache_session(session)
        
        logger.info(f"New conversation session created: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by ID"""
        # Check active sessions first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if not self._is_session_expired(session):
                return session
            else:
                # Remove expired session
                del self.active_sessions[session_id]
        
        # Try to load from cache
        cached_session = self._load_session_from_cache(session_id)
        if cached_session and not self._is_session_expired(cached_session):
            self.active_sessions[session_id] = cached_session
            return cached_session
        
        return None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(session_id)
    
    def add_message(self, session_id: str, role: str, content: str, 
                   context_data: Optional[Dict[str, Any]] = None,
                   screenshot_id: Optional[str] = None) -> ConversationMessage:
        """Add message to conversation session"""
        session = self.get_or_create_session(session_id)
        message = session.add_message(role, content, context_data, screenshot_id)
        
        # Update cache
        self._cache_session(session)
        
        logger.debug(f"Message added to conversation - session: {session_id}, role: {role}, message: {message.id}")
        
        return message    
    def get_conversation_context(self, session_id: str, include_screenshots: bool = False) -> Dict[str, Any]:
        """Get conversation context for AI analysis"""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        recent_messages = session.get_recent_messages(10)
        context = {
            "session_id": session_id,
            "conversation_summary": session.get_context_summary(),
            "message_count": len(session.messages),
            "session_duration": time.time() - session.created_at,
            "recent_messages": []
        }
        
        for msg in recent_messages:
            msg_data = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "relative_time": time.time() - msg.timestamp
            }
            
            if include_screenshots and msg.screenshot_id:
                # Try to get screenshot from cache
                screenshot = self.cache_manager.get("screenshots", msg.screenshot_id)
                if screenshot:
                    msg_data["has_screenshot"] = True
                    msg_data["screenshot_data"] = screenshot
            
            context["recent_messages"].append(msg_data)
        
        return context
    
    def _cache_session(self, session: ConversationSession):
        """Cache conversation session"""
        self.cache_manager.set(
            "conversations", 
            session.id, 
            session.to_dict(), 
            ttl=self.session_timeout
        )
    
    def _load_session_from_cache(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from cache"""
        try:
            session_data = self.cache_manager.get("conversations", session_id)
            if session_data:
                session = ConversationSession(
                    id=session_data['id'],
                    created_at=session_data['created_at'],
                    last_activity=session_data['last_activity'],
                    context=session_data.get('context', {})
                )
                
                # Restore messages
                for msg_data in session_data.get('messages', []):
                    message = ConversationMessage(
                        id=msg_data['id'],
                        timestamp=msg_data['timestamp'],
                        role=msg_data['role'],
                        content=msg_data['content'],
                        context_data=msg_data.get('context_data'),
                        screenshot_id=msg_data.get('screenshot_id')
                    )
                    session.messages.append(message)
                
                return session
        except Exception as e:
            logger.error(f"Failed to load session from cache - session: {session_id}, error: {str(e)}")
        
        return None    
    def _is_session_expired(self, session: ConversationSession) -> bool:
        """Check if session is expired"""
        return time.time() - session.last_activity > self.session_timeout
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if self._is_session_expired(session)
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.debug(f"Expired session removed: {session_id}")
        
        return len(expired_sessions)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions"""
        self.cleanup_expired_sessions()
        
        sessions = []
        for session in self.active_sessions.values():
            sessions.append({
                "id": session.id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "message_count": len(session.messages),
                "duration": time.time() - session.created_at
            })
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete conversation session"""
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove from cache
        self.cache_manager.delete("conversations", session_id)
        
        logger.info(f"Conversation session deleted: {session_id}")
        return True

# Global conversation manager instance
_conversation_manager: Optional[ConversationManager] = None

def get_conversation_manager() -> ConversationManager:
    """Get global conversation manager instance"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager