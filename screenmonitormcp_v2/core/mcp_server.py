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
    from .ai_analyzer import AIAnalyzer
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
    from core.ai_analyzer import AIAnalyzer
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
ai_analyzer = AIAnalyzer()

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
async def analyze_scene_from_memory(
    query: str,
    stream_id: Optional[str] = None,
    time_range_hours: int = 1
) -> str:
    """Analyze scene based on stored memory data
    
    Args:
        query: Scene analysis query (e.g., "What happened in the last hour?", "What objects were visible?")
        stream_id: Optional stream ID to filter analysis
        time_range_hours: Hours to look back in memory (default: 1)
    
    Returns:
        Scene analysis result based on memory
    """
    try:
        if not ai_service.is_available():
            return "Error: AI service is not available. Please configure your AI provider."
        
        result = await ai_service.analyze_scene_from_memory(
            query=query,
            stream_id=stream_id,
            time_range_hours=time_range_hours
        )
        
        if result.get("success"):
            response = result.get("response", "No analysis available")
            context_count = result.get("context_entries", 0)
            return f"Scene Analysis: {response}\n\nBased on {context_count} memory entries from the last {time_range_hours} hour(s)."
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Scene analysis from memory failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def query_memory(
    query: str,
    entry_type: Optional[str] = None,
    stream_id: Optional[str] = None,
    limit: int = 10
) -> str:
    """Query the memory system for stored analysis data
    
    Args:
        query: Search query for memory entries
        entry_type: Filter by entry type ('analysis', 'scene', 'context')
        stream_id: Filter by stream ID
        limit: Maximum number of results (default: 10)
    
    Returns:
        Memory query results
    """
    try:
        result = await ai_service.query_memory_direct(
            query=query,
            entry_type=entry_type,
            stream_id=stream_id,
            limit=limit
        )
        
        if result.get("success"):
            results = result.get("results", [])
            count = result.get("count", 0)
            
            if count == 0:
                return f"No memory entries found for query: '{query}'"
            
            response_lines = [f"Found {count} memory entries for query: '{query}'\n"]
            
            for i, entry in enumerate(results[:5], 1):  # Show first 5 results
                timestamp = entry.get("timestamp", "Unknown")
                entry_type = entry.get("entry_type", "Unknown")
                content = entry.get("content", {})
                
                if entry_type == "analysis":
                    analysis_text = content.get("response", "No analysis text")
                    response_lines.append(f"{i}. [{timestamp}] Analysis: {analysis_text[:200]}...")
                elif entry_type == "scene":
                    description = content.get("description", "No description")
                    objects = content.get("objects", [])
                    response_lines.append(f"{i}. [{timestamp}] Scene: {description} (Objects: {', '.join(objects[:5])})")
                else:
                    response_lines.append(f"{i}. [{timestamp}] {entry_type}: {str(content)[:200]}...")
            
            if count > 5:
                response_lines.append(f"\n... and {count - 5} more entries.")
            
            return "\n".join(response_lines)
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_memory_statistics() -> str:
    """Get memory system statistics and health information
    
    Returns:
        Memory system statistics
    """
    try:
        result = await ai_service.get_memory_statistics()
        
        if result.get("success"):
            stats = result.get("statistics", {})
            
            response_lines = [
                "Memory System Statistics:",
                f"- Total entries: {stats.get('total_entries', 0)}",
                f"- Recent entries (24h): {stats.get('recent_entries_24h', 0)}",
                f"- Database path: {stats.get('database_path', 'Unknown')}",
                f"- Initialized: {stats.get('initialized', False)}"
            ]
            
            entries_by_type = stats.get("entries_by_type", {})
            if entries_by_type:
                response_lines.append("\nEntries by type:")
                for entry_type, count in entries_by_type.items():
                    response_lines.append(f"- {entry_type}: {count}")
            
            return "\n".join(response_lines)
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Failed to get memory statistics: {e}")
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

# Memory System Tools

@mcp.tool()
async def analyze_scene_from_memory(
    query: str,
    stream_id: Optional[str] = None,
    time_range_minutes: int = 30,
    limit: int = 10
) -> str:
    """Analyze scene based on stored memory data
    
    Args:
        query: What to analyze or look for in the stored scenes
        stream_id: Specific stream to analyze (optional)
        time_range_minutes: Time range to search in minutes (default: 30)
        limit: Maximum number of results to analyze (default: 10)
    
    Returns:
        Scene analysis based on memory data
    """
    try:
        result = await ai_service.analyze_scene_from_memory(
            query=query,
            stream_id=stream_id,
            time_range_minutes=time_range_minutes,
            limit=limit
        )
        return f"Scene analysis: {result}"
    except Exception as e:
        logger.error(f"Failed to analyze scene from memory: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def query_memory(
    query: str,
    stream_id: Optional[str] = None,
    time_range_minutes: int = 60,
    limit: int = 20
) -> str:
    """Query the memory system for stored analysis data
    
    Args:
        query: Search query for memory entries
        stream_id: Filter by specific stream ID (optional)
        time_range_minutes: Time range to search in minutes (default: 60)
        limit: Maximum number of results (default: 20)
    
    Returns:
        Memory query results
    """
    try:
        result = await ai_service.query_memory_direct(
            query=query,
            stream_id=stream_id,
            time_range_minutes=time_range_minutes,
            limit=limit
        )
        return f"Memory query results: {result}"
    except Exception as e:
        logger.error(f"Failed to query memory: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_memory_statistics() -> str:
    """Get memory system statistics and health information
    
    Returns:
        Memory system statistics
    """
    try:
        stats = await ai_service.get_memory_statistics()
        return f"Memory statistics: {stats}"
    except Exception as e:
        logger.error(f"Failed to get memory statistics: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_stream_memory_stats() -> str:
    """Get memory system statistics for streaming
    
    Returns:
        Streaming memory statistics
    """
    try:
        stats = stream_manager.get_memory_stats()
        return f"Stream memory statistics: {stats}"
    except Exception as e:
        logger.error(f"Failed to get stream memory stats: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def configure_stream_memory(
    enabled: bool = True,
    analysis_interval: int = 5
) -> str:
    """Configure memory system for streaming
    
    Args:
        enabled: Enable or disable memory system for streaming
        analysis_interval: Analysis interval in frames (default: 5)
    
    Returns:
        Configuration result
    """
    try:
        stream_manager.enable_memory_system(enabled)
        if analysis_interval > 0:
            stream_manager.set_analysis_interval(analysis_interval)
        
        return f"Stream memory configured: enabled={enabled}, interval={analysis_interval}"
    except Exception as e:
        logger.error(f"Failed to configure stream memory: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_memory_usage() -> str:
    """Get detailed memory usage and performance metrics
    
    Returns:
        Detailed memory usage statistics
    """
    try:
        from .memory_system import memory_system
        
        usage_stats = await memory_system.get_memory_usage()
        
        if "error" in usage_stats:
            return f"Error getting memory usage: {usage_stats['error']}"
        
        response_lines = [
            "Memory Usage Statistics:",
            f"- Database size: {usage_stats.get('database_size_mb', 0)} MB",
            f"- Process memory: {usage_stats.get('process_memory_mb', 0)} MB",
            f"- Total entries: {usage_stats.get('total_entries', 0)}",
            f"- Recent entries (1h): {usage_stats.get('recent_entries_1h', 0)}",
            f"- Auto cleanup enabled: {usage_stats.get('auto_cleanup_enabled', False)}"
        ]
        
        cleanup_stats = usage_stats.get('cleanup_stats', {})
        if cleanup_stats:
            response_lines.extend([
                "\nCleanup Statistics:",
                f"- Cleanup runs: {cleanup_stats.get('cleanup_runs', 0)}",
                f"- Last cleanup: {cleanup_stats.get('last_cleanup', 'Never')}"
            ])
        
        return "\n".join(response_lines)
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def configure_auto_cleanup(
    enabled: bool,
    max_age_days: int = 7
) -> str:
    """Configure automatic memory cleanup settings
    
    Args:
        enabled: Enable or disable auto cleanup
        max_age_days: Maximum age for entries in days (default: 7)
    
    Returns:
        Configuration result
    """
    try:
        from .memory_system import memory_system
        
        # Stop current scheduler if running
        if memory_system._cleanup_task and not memory_system._cleanup_task.done():
            await memory_system.stop_cleanup_scheduler()
        
        # Update configuration
        memory_system.auto_cleanup = enabled
        
        # Start new scheduler if enabled
        if enabled:
            memory_system._cleanup_task = asyncio.create_task(
                memory_system._auto_cleanup_scheduler()
            )
            
            # Perform immediate cleanup with new settings
            deleted_count = await memory_system.cleanup_old_entries(
                max_age=timedelta(days=max_age_days)
            )
            
            return f"Auto cleanup configured: enabled={enabled}, max_age={max_age_days} days. Immediate cleanup removed {deleted_count} entries."
        else:
            return f"Auto cleanup disabled."
            
    except Exception as e:
         logger.error(f"Failed to configure auto cleanup: {e}")
         return f"Error: {str(e)}"

@mcp.tool()
def get_stream_resource_stats() -> str:
    """Get streaming resource usage statistics
    
    Returns:
        Streaming resource usage statistics
    """
    try:
        stats = stream_manager.get_resource_stats()
        
        response_lines = [
            "Streaming Resource Statistics:",
            f"- Memory usage: {stats.get('memory_usage_mb', 'N/A')} MB",
            f"- Memory limit: {stats.get('memory_limit_mb', 'N/A')} MB",
            f"- Active streams: {stats.get('active_streams', 0)}",
            f"- Max streams: {stats.get('max_streams', 'N/A')}",
            f"- Last cleanup: {stats.get('last_cleanup', 'Never')}",
            f"- Cleanup interval: {stats.get('cleanup_interval', 'N/A')} seconds"
        ]
        
        frame_buffers = stats.get('frame_buffers', {})
        if frame_buffers:
            response_lines.append("\nFrame Buffers:")
            for stream_id, buffer_size in frame_buffers.items():
                response_lines.append(f"- {stream_id}: {buffer_size} frames")
        
        return "\n".join(response_lines)
    except Exception as e:
        logger.error(f"Failed to get stream resource stats: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def configure_stream_resources(
    max_memory_mb: Optional[int] = None,
    max_streams: Optional[int] = None,
    frame_buffer_size: Optional[int] = None,
    cleanup_interval: Optional[int] = None
) -> str:
    """Configure streaming resource limits
    
    Args:
        max_memory_mb: Maximum memory usage in MB (optional)
        max_streams: Maximum concurrent streams (optional)
        frame_buffer_size: Maximum frames to buffer per stream (optional)
        cleanup_interval: Cleanup interval in seconds (optional)
    
    Returns:
        Configuration result
    """
    try:
        stream_manager.configure_resource_limits(
            max_memory_mb=max_memory_mb,
            max_streams=max_streams,
            frame_buffer_size=frame_buffer_size,
            cleanup_interval=cleanup_interval
        )
        
        config_items = []
        if max_memory_mb is not None:
            config_items.append(f"max_memory_mb={max_memory_mb}")
        if max_streams is not None:
            config_items.append(f"max_streams={max_streams}")
        if frame_buffer_size is not None:
            config_items.append(f"frame_buffer_size={frame_buffer_size}")
        if cleanup_interval is not None:
            config_items.append(f"cleanup_interval={cleanup_interval}")
        
        return f"Stream resource limits configured: {', '.join(config_items)}"
    except Exception as e:
        logger.error(f"Failed to configure stream resources: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_database_pool_stats() -> str:
    """Get database connection pool statistics
    
    Returns:
        Database pool usage statistics
    """
    try:
        from .memory_system import memory_system
        
        if not memory_system._db_pool:
            return "Database pool not initialized"
        
        import asyncio
        stats = asyncio.run(memory_system._db_pool.get_stats())
        
        response_lines = [
            "Database Pool Statistics:",
            f"- Total connections: {stats.total_connections}",
            f"- Active connections: {stats.active_connections}",
            f"- Idle connections: {stats.idle_connections}",
            f"- Total queries: {stats.total_queries}",
            f"- Failed queries: {stats.failed_queries}",
            f"- Average query time: {stats.average_query_time:.4f}s",
            f"- Pool created: {stats.pool_created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        if stats.last_cleanup:
            response_lines.append(f"- Last cleanup: {stats.last_cleanup.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            response_lines.append("- Last cleanup: Never")
        
        return "\n".join(response_lines)
    except Exception as e:
        logger.error(f"Failed to get database pool stats: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def database_pool_health_check() -> str:
    """Perform database pool health check
    
    Returns:
        Database pool health status
    """
    try:
        from .memory_system import memory_system
        
        if not memory_system._db_pool:
            return "Database pool not initialized"
        
        import asyncio
        health = asyncio.run(memory_system._db_pool.health_check())
        
        if health["healthy"]:
            response_lines = [
                "Database Pool Health: HEALTHY ✓",
                f"- Total connections: {health['total_connections']}",
                f"- Active connections: {health['active_connections']}",
                f"- Idle connections: {health['idle_connections']}",
                f"- Pool utilization: {health['pool_utilization']:.1%}",
                f"- Average query time: {health['average_query_time']:.4f}s"
            ]
        else:
            response_lines = [
                "Database Pool Health: UNHEALTHY ✗",
                f"- Error: {health.get('error', 'Unknown error')}"
            ]
        
        return "\n".join(response_lines)
    except Exception as e:
        logger.error(f"Failed to perform database health check: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def detect_ui_elements(
    monitor: int = 0
) -> str:
    """Detect and classify UI elements in the current screen
    
    Args:
        monitor: Monitor number to analyze (0 for primary)
    
    Returns:
        UI elements detection results as text
    """
    try:
        if not ai_analyzer.client:
            return "Error: AI analyzer is not available. Please configure your OpenAI API key."
        
        image_data = await screen_capture.capture_screen(monitor)
        result = await ai_analyzer.detect_ui_elements(image_data)
        
        if result.get("success"):
            return result.get("analysis", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"UI elements detection failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def assess_system_performance(
    monitor: int = 0
) -> str:
    """Assess system performance indicators visible on screen
    
    Args:
        monitor: Monitor number to analyze (0 for primary)
    
    Returns:
        Performance assessment results as text
    """
    try:
        if not ai_analyzer.client:
            return "Error: AI analyzer is not available. Please configure your OpenAI API key."
        
        image_data = await screen_capture.capture_screen(monitor)
        result = await ai_analyzer.assess_system_performance(image_data)
        
        if result.get("success"):
            return result.get("analysis", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"System performance assessment failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def detect_anomalies(
    monitor: int = 0,
    baseline_description: str = ""
) -> str:
    """Detect visual anomalies and unusual patterns in the screen
    
    Args:
        monitor: Monitor number to analyze (0 for primary)
        baseline_description: Optional description of normal state for comparison
    
    Returns:
        Anomaly detection results as text
    """
    try:
        if not ai_analyzer.client:
            return "Error: AI analyzer is not available. Please configure your OpenAI API key."
        
        image_data = await screen_capture.capture_screen(monitor)
        result = await ai_analyzer.detect_anomalies(image_data, baseline_description)
        
        if result.get("success"):
            return result.get("analysis", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def generate_monitoring_report(
    monitor: int = 0,
    context: str = ""
) -> str:
    """Generate comprehensive monitoring report from screen analysis
    
    Args:
        monitor: Monitor number to analyze (0 for primary)
        context: Additional context for the report
    
    Returns:
        Comprehensive monitoring report as text
    """
    try:
        if not ai_analyzer.client:
            return "Error: AI analyzer is not available. Please configure your OpenAI API key."
        
        image_data = await screen_capture.capture_screen(monitor)
        result = await ai_analyzer.generate_monitoring_report(image_data, context)
        
        if result.get("success"):
            return result.get("analysis", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    except Exception as e:
        logger.error(f"Monitoring report generation failed: {e}")
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