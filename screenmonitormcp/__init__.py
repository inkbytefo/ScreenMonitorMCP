"""
ScreenMonitorMCP - Revolutionary AI Vision Server

Give AI real-time sight and screen interaction capabilities.

This package provides a comprehensive MCP (Model Context Protocol) server that enables
AI assistants like Claude to monitor screens, analyze visual content, and interact
with user interfaces in real-time.

Key Features:
- Real-time screen monitoring and analysis
- Natural language UI interaction (smart clicking)
- OCR text extraction from screen regions
- AI-powered visual analysis of screenshots and videos
- Intelligent event detection and monitoring
- Cross-platform support (Windows, macOS, Linux)
- Advanced caching and performance optimization
- Streaming capabilities for real-time AI analysis

Author: inkbytefo
License: MIT
"""

__version__ = "1.0.0"
__author__ = "inkbytefo"
__email__ = "inkbytefo@example.com"
__license__ = "MIT"

# Core modules
from .main import main
from .ai_providers import OpenAIProvider
from .application_monitor import ApplicationMonitor, ApplicationEvent
from .smart_monitoring import SmartMonitor, SmartMonitoringConfig, SmartEvent
from .video_recorder import VideoRecorder, VideoAnalyzer, VideoRecordingConfig
from .cache_manager import get_cache_manager
from .conversation_context import get_conversation_manager
from .system_metrics import get_metrics_manager
from .batch_processor import get_batch_processor, BatchPriority
from .image_optimizer import get_image_optimizer, OptimizationConfig
from .error_recovery import get_error_recovery_manager
from .platform_support import get_platform_manager
from .input_simulator import get_input_simulator, KeyboardInput, MouseInput, MouseButton
from .screen_streamer import get_global_stream_manager, StreamConfig
from .ui_detection import get_ui_detector, get_smart_clicker

__all__ = [
    # Core functions
    "main",
    
    # AI and Analysis
    "OpenAIProvider",
    
    # Monitoring and Events
    "ApplicationMonitor",
    "ApplicationEvent", 
    "SmartMonitor",
    "SmartMonitoringConfig",
    "SmartEvent",
    
    # Video and Recording
    "VideoRecorder",
    "VideoAnalyzer", 
    "VideoRecordingConfig",
    
    # System Management
    "get_cache_manager",
    "get_conversation_manager",
    "get_metrics_manager",
    "get_batch_processor",
    "BatchPriority",
    "get_image_optimizer",
    "OptimizationConfig",
    "get_error_recovery_manager",
    "get_platform_manager",
    
    # Input and Interaction
    "get_input_simulator",
    "KeyboardInput",
    "MouseInput", 
    "MouseButton",
    
    # Streaming and UI
    "get_global_stream_manager",
    "StreamConfig",
    "get_ui_detector",
    "get_smart_clicker",
]

# Package metadata
__package_info__ = {
    "name": "screenmonitormcp",
    "version": __version__,
    "description": "Revolutionary AI Vision MCP Server - Give AI real-time sight and screen interaction capabilities",
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/inkbytefo/ScreenMonitorMCP",
    "keywords": ["mcp", "ai", "screen-monitoring", "computer-vision", "automation"],
    "python_requires": ">=3.8",
}
