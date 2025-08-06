"""Request models for ScreenMonitorMCP v2, defining MCP v3 commands."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class CommandType(str, Enum):
    """
    Enumeration for MCP v3 commands sent from the Agent to the Server.
    """
    SUBSCRIBE_PREVIEW = "subscribe_preview"
    REQUEST_HQ_FRAME = "request_hq_frame"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"


class WebSocketCommand(BaseModel):
    """
    Represents a command sent from the Agent to the Server via WebSocket.
    This is the main model for parsing incoming JSON control messages.
    """
    command: CommandType = Field(..., description="The command to be executed.")
    stream_id: Optional[str] = Field(
        None, 
        description="Identifier for the stream, required for stream-related commands like 'subscribe_preview' and 'unsubscribe'."
    )
    request_id: Optional[str] = Field(
        None, 
        description="Optional unique identifier for tracking requests, especially for 'request_hq_frame'."
    )
    timestamp: Optional[float] = Field(
        None, 
        description="Timestamp of command generation (Unix timestamp)."
    )

    @validator('stream_id', always=True)
    def check_stream_id_for_stream_commands(cls, v, values):
        """Validator to ensure 'stream_id' is provided for commands that require it."""
        command = values.get('command')
        if command in {CommandType.SUBSCRIBE_PREVIEW, CommandType.UNSUBSCRIBE} and v is None:
            raise ValueError(f"'stream_id' is required for the '{command.value}' command.")
        return v


class StreamQuality(str, Enum):
    """Stream quality levels for MCP v3."""
    PREVIEW = "preview"  # Low quality, continuous stream
    HQ = "hq"           # High quality, on-demand


class BaseRequest(BaseModel):
    """Base request model for HTTP requests."""
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    timestamp: Optional[float] = Field(None, description="Request timestamp")


class ScreenCaptureRequest(BaseRequest):
    """Request model for screen capture operations."""
    capture_mode: str = Field("monitor", description="Capture mode: monitor, window, region, all")
    monitor: int = Field(1, description="Monitor number for capture")
    capture_active_window: bool = Field(False, description="Capture active window")
    region: Optional[Dict[str, int]] = Field(None, description="Region coordinates")
    output_format: str = Field("png", description="Output format: png, jpeg")
    quality: int = Field(95, description="JPEG quality (1-100)")
    max_width: Optional[int] = Field(None, description="Maximum width for resizing")
    max_height: Optional[int] = Field(None, description="Maximum height for resizing")
    include_metadata: bool = Field(True, description="Include metadata in response")


class StreamRequest(BaseRequest):
    """Request model for creating a stream."""
    stream_type: str = Field("screen", description="Type of stream: screen, monitoring, events")
    fps: int = Field(10, description="Frames per second for streaming")
    quality: int = Field(80, description="Streaming quality (1-100)")
    format: str = Field("jpeg", description="Streaming format")
    max_connections: int = Field(10, description="Maximum concurrent connections")
    filters: Optional[Dict[str, Any]] = Field(None, description="Streaming filters")


class AIImageAnalysisRequest(BaseRequest):
    """Request model for AI image analysis."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    prompt: str = Field("What's in this image?", description="Analysis prompt")
    model: Optional[str] = Field(None, description="AI model to use")
    max_tokens: int = Field(300, description="Maximum tokens in response")


class AIChatRequest(BaseRequest):
    """Request model for AI chat completion."""
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="AI model to use")
    max_tokens: int = Field(1000, description="Maximum tokens in response")
    temperature: float = Field(0.7, description="Response temperature")


class AIModelListRequest(BaseRequest):
    """Request model for listing AI models."""
    provider: Optional[str] = Field(None, description="AI provider to query")


class ScreenAnalysisRequest(BaseRequest):
    """Request model for screen analysis."""
    monitor: int = Field(1, description="Monitor number to analyze")
    region: Optional[Dict[str, int]] = Field(None, description="Region coordinates")
    prompt: str = Field("Analyze this screen content", description="Analysis prompt")
    model: Optional[str] = Field(None, description="AI model to use")
    max_tokens: int = Field(500, description="Maximum tokens in response")
