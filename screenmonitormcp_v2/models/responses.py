"""Response models for ScreenMonitorMCP v2."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ResponseType(str, Enum):
    """MCP v3 response types."""
    ACK = "ack"
    ERROR = "error"
    PONG = "pong"
    CONNECTED = "connected"
    STREAM_STARTED = "stream_started"
    STREAM_STOPPED = "stream_stopped"


class WebSocketResponse(BaseModel):
    """
    Base WebSocket response model for MCP v3 protocol.
    Used for JSON control messages (not binary data).
    Binary image data is sent separately via websocket.send_bytes().
    """
    type: ResponseType = Field(..., description="Response type")
    command: Optional[str] = Field(None, description="Original command that triggered this response")
    success: bool = Field(True, description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Original request ID")
    stream_id: Optional[str] = Field(None, description="Stream identifier for stream-related responses")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")


class BaseResponse(BaseModel):
    """Base response model."""
    
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Original request ID")
    duration_ms: Optional[float] = Field(None, description="Operation duration in milliseconds")


class ToolResponse(BaseResponse):
    """Response model for MCP tool execution."""
    
    tool_name: str = Field(..., description="Name of the executed tool")
    result: Any = Field(None, description="Tool execution result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
    streaming: bool = Field(False, description="Whether this is a streaming response")


class ScreenCaptureResponse(BaseResponse):
    """Response model for screen capture operations."""
    
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_size: Dict[str, int] = Field(..., description="Image dimensions")
    file_size: int = Field(..., description="Image file size in bytes")
    format: str = Field(..., description="Image format")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image metadata")
    monitor_info: Dict[str, Any] = Field(default_factory=dict, description="Monitor information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Capture timestamp")


class StreamResponse(BaseResponse):
    """Response model for streaming operations."""
    
    stream_id: str = Field(..., description="Unique stream identifier")
    stream_type: str = Field(..., description="Type of stream")
    status: str = Field(..., description="Stream status: active, paused, stopped")
    connections: int = Field(0, description="Active connections count")
    fps: int = Field(..., description="Current FPS")
    quality: int = Field(..., description="Current quality")
    format: str = Field(..., description="Stream format")
    url: Optional[str] = Field(None, description="Stream endpoint URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stream metadata")


class HealthCheckResponse(BaseResponse):
    """Response model for health checks."""
    
    version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")
    status: str = Field(..., description="Overall health status")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")
    dependencies: Dict[str, bool] = Field(default_factory=dict, description="Dependency status")
    performance: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    active_streams: int = Field(0, description="Number of active streams")
    active_connections: int = Field(0, description="Number of active connections")


class ErrorResponse(BaseResponse):
    """Response model for errors."""
    
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    traceback: Optional[str] = Field(None, description="Error traceback")
    suggestion: Optional[str] = Field(None, description="Suggested action")


class StreamingEvent(BaseModel):
    """Model for streaming events."""
    
    event_type: str = Field(..., description="Type of streaming event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    stream_id: str = Field(..., description="Associated stream ID")
    sequence: int = Field(..., description="Event sequence number")


class WebSocketResponse(BaseModel):
    """WebSocket response model."""
    
    type: str = Field(..., description="Response type")
    data: Dict[str, Any] = Field(..., description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    error: Optional[ErrorResponse] = Field(None, description="Error information")


class AIModel(BaseModel):
    """AI model information."""
    
    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Model owner")
    context_window: Optional[int] = Field(None, description="Context window size")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")


class AIModelListResponse(BaseResponse):
    """Response model for AI model listing."""
    
    data: List[AIModel] = Field(..., description="List of available models")
    object: str = Field("list", description="Object type")


class AIImageAnalysisResponse(BaseResponse):
    """Response model for AI image analysis."""
    
    analysis: str = Field(..., description="AI analysis result")
    model: str = Field(..., description="Model used for analysis")
    prompt: str = Field(..., description="Analysis prompt used")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class AIChatResponse(BaseResponse):
    """Response model for AI chat completion."""
    
    message: Dict[str, Any] = Field(..., description="AI response message")
    model: str = Field(..., description="Model used for completion")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: str = Field(..., description="Completion finish reason")


class AIStatusResponse(BaseResponse):
    """Response model for AI service status."""
    
    service_available: bool = Field(..., description="Whether AI service is available")
    configured: bool = Field(..., description="Whether AI service is configured")
    provider: str = Field(..., description="AI provider name")
    base_url: str = Field(..., description="API base URL")
    models_available: int = Field(..., description="Number of available models")
    error: Optional[str] = Field(None, description="Error message if service unavailable")


class ScreenAnalysisResponse(BaseResponse):
    """Response model for screen analysis operations."""
    
    analysis: str = Field(..., description="AI analysis of the screen content")
    model: str = Field(..., description="AI model used for analysis")
    prompt: str = Field(..., description="Analysis prompt used")
    capture_info: Dict[str, Any] = Field(..., description="Information about captured screen")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
