"""ScreenMonitorMCP v2 - Model Context Protocol (MCP) Server Implementation

This module implements the MCP server for ScreenMonitorMCP v2, providing
screen capture and analysis capabilities through the Model Context Protocol.

The server operates using the official MCP Python SDK FastMCP API and provides tools for:
- Screen capture
- Screen analysis with AI
- Real-time monitoring

Author: ScreenMonitorMCP Team
Version: 2.0.0
License: MIT
"""

import asyncio
import logging
import base64
import sys
from typing import Any, Optional
from datetime import datetime

# Official MCP Python SDK FastMCP imports
from mcp.server.fastmcp import FastMCP

try:
    from .screen_capture import ScreenCapture
    from .ai_service import ai_service
    from .streaming import stream_manager
    from .performance_monitor import performance_monitor
    from .config import Config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.screen_capture import ScreenCapture
    from core.ai_service import ai_service
    from core.streaming import stream_manager
    from core.performance_monitor import performance_monitor
    from core.config import Config

# Configure logger to use stderr for MCP mode
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)  # Only show errors in MCP mode

# Initialize FastMCP server
mcp = FastMCP("screenmonitormcp-v2")

# Initialize components
config = Config()
screen_capture = ScreenCapture()

@mcp.tool()
async def capture_screen(
    monitor: int = 0,
    region: Optional[dict] = None,
    format: str = "png"
) -> str:
    """Capture a screenshot of the current screen
    
    Args:
        monitor: Monitor number to capture (0 for primary)
        region: Specific region to capture with x, y, width, height
        format: Image format (png or jpeg)
    
    Returns:
        Base64 encoded image data
    """
    try:
        if region:
            # Convert region dict to proper format for screen_capture
            region_dict = {
                'x': region.get('x', 0),
                'y': region.get('y', 0),
                'width': region.get('width', 1920),
                'height': region.get('height', 1080)
            }
            image_data = await screen_capture.capture_screen(monitor, region_dict, format)
        else:
            image_data = await screen_capture.capture_screen(monitor, None, format)
        
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Screen capture failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def analyze_screen(
    query: str,
    monitor: int = 0,
    detail_level: str = "high"
) -> str:
    """Analyze the current screen content using AI vision
    
    Args:
        query: What to analyze or look for in the screen
        monitor: Monitor number to analyze (0 for primary)
        detail_level: Level of detail for analysis (low or high)
    
    Returns:
        Analysis result as text
    """
    try:
        if not ai_service.is_available():
            return "Error: AI service is not available. Please configure your AI provider."
        
        image_data = await screen_capture.capture_screen(monitor)
        # Convert image data to base64 for ai_service
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        result = await ai_service.analyze_image(image_base64, query)
        
        if result.get("success"):
            return result.get("response", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Screen analysis failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def analyze_image(
    image_base64: str,
    prompt: str = "What's in this image?",
    model: Optional[str] = None,
    max_tokens: int = 300
) -> str:
    """Analyze a provided image using AI vision capabilities
    
    Args:
        image_base64: Base64 encoded image data
        prompt: What to analyze or look for in the image
        model: AI model to use for analysis
        max_tokens: Maximum tokens for response
    
    Returns:
        Analysis result as text
    """
    try:
        if not ai_service.is_available():
            return "Error: AI service is not available. Please configure your AI provider."
        
        # Use ai_service.analyze_image instead of ai_analyzer.analyze_image
        result = await ai_service.analyze_image(image_base64, prompt, model, max_tokens)
        
        if result.get("success"):
            return result.get("response", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def chat_completion(
    messages: list,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Generate chat completion using AI models
    
    Args:
        messages: Array of chat messages with role and content
        model: AI model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for response generation
    
    Returns:
        AI response as text
    """
    try:
        if not ai_service.is_available():
            return "Error: AI service is not available. Please configure your AI provider."
        
        result = await ai_service.chat_completion(messages, model, max_tokens, temperature)
        
        if result.get("success"):
            return result.get("response", "No response available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def list_ai_models() -> str:
    """List available AI models from the configured provider
    
    Returns:
        List of available models as text
    """
    try:
        if not ai_service.is_available():
            return "Error: AI service is not available. Please configure your AI provider."
        
        result = await ai_service.list_models()
        
        if result.get("success"):
            models = result.get("models", [])
            if models:
                return f"Available models: {', '.join(models)}"
            else:
                return "No models available"
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Failed to list AI models: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_ai_status() -> str:
    """Get AI service configuration status
    
    Returns:
        AI service status information
    """
    try:
        status = ai_service.get_status()
        return f"AI Service Status: {status}"
    except Exception as e:
        logger.error(f"Failed to get AI status: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_performance_metrics() -> str:
    """Get detailed performance metrics and system health
    
    Returns:
        Performance metrics as text
    """
    try:
        metrics = performance_monitor.get_metrics()
        return f"Performance Metrics: {metrics}"
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_system_status() -> str:
    """Get overall system status and health information
    
    Returns:
        System status information
    """
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "ai_service": ai_service.is_available(),
            "screen_capture": screen_capture.is_available(),
            "performance_monitor": performance_monitor.is_running(),
            "stream_manager": stream_manager.is_running()
        }
        return f"System Status: {status}"
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def create_stream(
    monitor: int = 0,
    fps: int = 10,
    quality: int = 80,
    format: str = "jpeg"
) -> str:
    """Create a new screen streaming session
    
    Args:
        monitor: Monitor number to stream (0 for primary)
        fps: Frames per second for streaming
        quality: Image quality (1-100)
        format: Image format (jpeg or png)
    
    Returns:
        Stream ID or error message
    """
    try:
        stream_id = await stream_manager.create_stream("screen", fps, quality, format)
        return f"Stream created with ID: {stream_id}"
    except Exception as e:
        logger.error(f"Failed to create stream: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def list_streams() -> str:
    """List all active streaming sessions
    
    Returns:
        List of active streams
    """
    try:
        streams = stream_manager.list_streams()
        return f"Active streams: {streams}"
    except Exception as e:
        logger.error(f"Failed to list streams: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_stream_info(stream_id: str) -> str:
    """Get information about a specific stream
    
    Args:
        stream_id: Stream ID to get information for
    
    Returns:
        Stream information
    """
    try:
        info = await stream_manager.get_stream_info(stream_id)
        return f"Stream info: {info}"
    except Exception as e:
        logger.error(f"Failed to get stream info: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def stop_stream(stream_id: str) -> str:
    """Stop a specific streaming session
    
    Args:
        stream_id: Stream ID to stop
    
    Returns:
        Success or error message
    """
    try:
        result = await stream_manager.stop_stream(stream_id)
        return f"Stream stopped: {result}"
    except Exception as e:
        logger.error(f"Failed to stop stream: {e}")
        return f"Error: {str(e)}"

def setup_logging():
    """Setup logging configuration for MCP mode."""
    # Disable all loggers except critical errors
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Disable specific noisy loggers
    for logger_name in ['httpx', 'openai', 'urllib3', 'requests']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).disabled = True

def run_mcp_server():
    """Run the MCP server."""
    setup_logging()
    
    try:
        # Run the FastMCP server (it handles its own event loop)
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    run_mcp_server()