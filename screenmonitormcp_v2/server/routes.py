"""API routes for ScreenMonitorMCP v2."""

import asyncio
import json
from datetime import datetime
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, List, Optional

from ..models.requests import ScreenCaptureRequest, StreamRequest, AIImageAnalysisRequest, AIChatRequest, AIModelListRequest, ScreenAnalysisRequest
from ..models.responses import ScreenCaptureResponse, StreamResponse, AIModelListResponse, AIImageAnalysisResponse, AIChatResponse, AIStatusResponse, ScreenAnalysisResponse
from ..core.connection import connection_manager
from ..core.streaming import stream_manager, screen_streamer, stream_analysis_generator
from ..core.ai_service import ai_service
from ..core.performance_monitor import performance_monitor
from .config import config

logger = logging.getLogger(__name__)

# Create routers
api_router = APIRouter()
ws_router = APIRouter()

# API key security
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    token = credentials.credentials
    if token != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token

@api_router.get("/status")
async def get_status():
    """Get server status."""
    connection_stats = await connection_manager.get_stats()
    active_streams = await stream_manager.get_active_streams()
    
    return {
        "success": True,
        "message": "Status retrieved successfully",
        "data": {
            "connections": connection_stats,
            "streams": active_streams
        }
    }

@api_router.post("/capture")
async def capture_screen(request: ScreenCaptureRequest):
    """Capture screen endpoint."""
    try:
        screen_data = await screen_streamer.capture_screen(
            monitor=request.monitor,
            region=request.region,
            quality=request.quality,
            format=request.output_format
        )
        
        return ScreenCaptureResponse(
            success=True,
            message="Screen captured successfully",
            image_data=screen_data["image_data"],
            image_size={
                "width": screen_data["width"],
                "height": screen_data["height"]
            },
            file_size=screen_data["size"],
            format=screen_data["format"],
            metadata={
                "monitor": screen_data["monitor"],
                "timestamp": screen_data["timestamp"]
            },
            monitor_info={
                "monitor": screen_data["monitor"]
            }
        )
    except Exception as e:
        logger.error("Screen capture failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@api_router.get("/streams")
async def list_streams():
    """List all active streams."""
    streams = await stream_manager.get_active_streams()
    
    return {
        "success": True,
        "message": "Streams retrieved successfully",
        "data": list(streams.values())
    }

@api_router.post("/streams")
async def create_stream(request: StreamRequest):
    """Create a new stream."""
    try:
        stream_id = await stream_manager.create_stream(
            stream_type=request.stream_type,
            fps=request.fps,
            quality=request.quality,
            format=request.format,
            filters=request.filters
        )
        
        return StreamResponse(
            success=True,
            message="Stream created successfully",
            stream_id=stream_id,
            stream_type=request.stream_type,
            status="created",
            fps=request.fps,
            quality=request.quality,
            format=request.format,
            url=f"/api/v2/streams/{stream_id}/sse"
        )
    except Exception as e:
        logger.error("Stream creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@api_router.get("/streams/{stream_id}/sse")
async def stream_sse(stream_id: str):
    """Server-Sent Events endpoint for streaming."""
    if stream_id not in await stream_manager.get_active_streams():
        raise HTTPException(
            status_code=404,
            detail=f"Stream {stream_id} not found"
        )
    
    async def event_generator():
        """Generate SSE events with real stream data."""
        try:
            # Add connection to stream
            connection_id = await connection_manager.add_connection(
                client_ip="sse-client",
                metadata={"type": "sse", "stream_id": stream_id}
            )
            
            await connection_manager.add_to_stream(connection_id, stream_id)
            
            yield 'data: {\n'
            yield '  "type": "connected",\n'
            yield f'  "stream_id": "{stream_id}",\n'
            yield f'  "timestamp": "{datetime.now().isoformat()}"\n'
            yield '}\n\n'
            
            # Create a queue for receiving stream data
            data_queue = asyncio.Queue()
            
            # Background task to receive stream data
            async def receive_stream_data():
                try:
                    # Get stream info for configuration
                    stream_info = await stream_manager.get_stream_info(stream_id)
                    if not stream_info:
                        return
                    
                    # Create a custom generator for this SSE connection
                    from ..core.streaming import screen_streamer
                    
                    async for data in screen_streamer.stream_screen(
                        stream_id=stream_id,
                        fps=stream_info["fps"],
                        quality=stream_info["quality"]
                    ):
                        if stream_info["status"] != "active":
                            break
                        
                        # Put data in queue for SSE
                        await data_queue.put(data)
                        
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error("Error receiving stream data", error=str(e))
                    await data_queue.put(None)  # Signal end
            
            # Start background task
            receive_task = asyncio.create_task(receive_stream_data())
            
            try:
                # Send real stream data
                while True:
                    try:
                        data = await asyncio.wait_for(data_queue.get(), timeout=30.0)
                        if data is None:  # End signal
                            break
                        
                        # Format SSE event with real image data
                        sse_data = {
                            "type": "stream_data",
                            "stream_id": stream_id,
                            "timestamp": datetime.now().isoformat(),
                            "sequence": data.get("sequence", 0),
                            "image_data": data.get("image_data", ""),
                            "width": data.get("width", 0),
                            "height": data.get("height", 0),
                            "format": data.get("format", "jpeg")
                        }
                        
                        yield f"data: {json.dumps(sse_data)}\n\n"
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat with performance metrics
                        heartbeat_data = {
                            'type': 'heartbeat', 
                            'timestamp': datetime.now().isoformat(),
                            'performance': {
                                'active_streams': len(await stream_manager.get_active_streams()),
                                'connection_count': len(await connection_manager.get_active_connections()),
                                'server_load': 'normal'  # Could be enhanced with actual metrics
                            }
                        }
                        yield f"data: {json.dumps(heartbeat_data)}\n\n"
                        
            finally:
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                
        except asyncio.CancelledError:
            logger.info("SSE connection closed", stream_id=stream_id)
        except Exception as e:
            logger.error("SSE error", error=str(e))
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
        finally:
            if 'connection_id' in locals():
                await connection_manager.remove_connection(connection_id)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@api_router.get("/streams/{stream_id}")
async def get_stream_info(stream_id: str):
    """Get stream information."""
    stream_info = await stream_manager.get_stream_info(stream_id)
    if not stream_info:
        raise HTTPException(
            status_code=404,
            detail=f"Stream {stream_id} not found"
        )
    
    connections = await connection_manager.get_stream_connections(stream_id)
    
    return StreamResponse(
        success=True,
        message="Stream information retrieved",
        stream_id=stream_id,
        stream_type=stream_info["stream_type"],
        status=stream_info["status"],
        connections=len(connections),
        fps=stream_info["fps"],
        quality=stream_info["quality"],
        format=stream_info["format"]
    )

@api_router.delete("/streams/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop a stream."""
    success = await stream_manager.stop_stream(stream_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Stream {stream_id} not found"
        )
    
    return {
        "success": True,
        "message": "Stream stopped successfully",
        "stream_id": stream_id
    }

@api_router.post("/streams/{stream_id}/start")
async def start_stream(stream_id: str):
    """Start a stream."""
    stream_info = await stream_manager.get_stream_info(stream_id)
    if not stream_info:
        raise HTTPException(
            status_code=404,
            detail=f"Stream {stream_id} not found"
        )
    
    # Determine data generator based on stream type
    stream_type = stream_info.get("stream_type", "screen")
    
    if stream_type == "ai_analysis":
        # AI Analysis stream
        prompt = stream_info.get("prompt", "Analyze this screen content and provide a detailed summary of what's happening.")
        interval_seconds = stream_info.get("interval_seconds", 10)
        model = stream_info.get("model")
        max_tokens = stream_info.get("max_tokens", 300)
        
        async def ai_analysis_generator(stream_id: str):
            async for data in stream_analysis_generator(
                stream_id=stream_id,
                interval_seconds=interval_seconds,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens
            ):
                yield data
        
        data_generator = ai_analysis_generator
        
    else:
        # Regular screen streaming
        from ..core.streaming import screen_streamer
        
        async def screen_generator(stream_id: str):
            async for data in screen_streamer.stream_screen(
                stream_id=stream_id,
                fps=stream_info["fps"],
                quality=stream_info["quality"]
            ):
                yield data
        
        data_generator = screen_generator
    
    success = await stream_manager.start_stream(
        stream_id,
        data_generator
    )
    
    if success:
        return {
            "success": True,
            "message": "Stream started successfully",
            "stream_id": stream_id,
            "stream_type": stream_type
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to start stream {stream_id}"
        )


@api_router.post("/streams/{stream_id}/pause")
async def pause_stream(stream_id: str):
    """Pause a stream."""
    success = await stream_manager.pause_stream(stream_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Stream {stream_id} not found"
        )
    
    return {
        "success": True,
        "message": "Stream paused successfully",
        "stream_id": stream_id
    }


@api_router.post("/streams/{stream_id}/resume")
async def resume_stream(stream_id: str):
    """Resume a stream."""
    success = await stream_manager.resume_stream(stream_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Stream {stream_id} not found"
        )
    
    return {
        "success": True,
        "message": "Stream resumed successfully",
        "stream_id": stream_id
    }


@api_router.get("/connections")
async def list_connections():
    """List active connections."""
    connections = await connection_manager.get_active_connections()
    
    return {
        "success": True,
        "message": "Connections retrieved successfully",
        "data": {
            conn_id: {
                "client_ip": conn.client_ip,
                "duration": conn.duration.total_seconds(),
                "idle_time": conn.idle_time.total_seconds(),
                "streams": list(conn.stream_types)
            }
            for conn_id, conn in connections.items()
        }
    }


@ws_router.websocket("/mcp")
async def websocket_mcp_v3(websocket: WebSocket):
    """
    MCP v3 WebSocket endpoint implementing dual-channel interactive analysis architecture.
    
    Dual Channel Architecture:
    - Observation Channel: Continuous low-quality binary stream (websocket.send_bytes)
    - Command & Analysis Channel: JSON request/response for control and HQ frames
    
    Features:
    - Non-blocking concurrent tasks (client_listener_task + stream_publisher_task)
    - CPU-intensive operations in thread pool executor
    - Graceful error handling and cleanup
    """
    try:
        await websocket.accept()
        logger.info("MCP v3 WebSocket connection accepted")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket connection: {e}")
        raise
    
    connection_id = None
    client_listener_task = None
    
    try:
        # Import command handler
        from ..core.command_handler import command_handler
        from ..models.requests import WebSocketCommand
        from ..models.responses import WebSocketResponse, ResponseType
        
        # Add connection
        connection_id = await connection_manager.add_connection(
            client_ip=websocket.client.host,
            user_agent=websocket.headers.get("user-agent"),
            metadata={"protocol": "mcp_v3"}
        )
        
        # Store WebSocket reference
        conn = await connection_manager.get_connection(connection_id)
        if conn:
            conn.websocket = websocket
        
        # Send connection acknowledgment (JSON control message)
        connection_response = WebSocketResponse(
            type=ResponseType.CONNECTED,
            success=True,
            message="MCP v3 connection established",
            data={"connection_id": connection_id},
            timestamp=datetime.now().isoformat()
        )
        await websocket.send_text(connection_response.json())
        
        logger.info(
            f"MCP v3 WebSocket connection established - connection_id: {connection_id}, client_ip: {websocket.client.host}"
        )
        
        # Start client listener task (handles JSON commands)
        # This task runs continuously and processes incoming commands
        # Commands like subscribe_preview will trigger stream_publisher_task
        # Commands like request_hq_frame will send single binary frames
        client_listener_task = asyncio.create_task(
            _handle_client_messages(websocket, connection_id, command_handler)
        )
        
        # Wait for client listener to complete (or disconnect)
        await client_listener_task
        
    except WebSocketDisconnect:
        logger.info(f"MCP v3 WebSocket disconnected - connection_id: {connection_id}")
    except Exception as e:
        logger.error(
            f"MCP v3 WebSocket error - connection_id: {connection_id}, error: {str(e)}",
            exc_info=True
        )
    finally:
        # Cleanup: Cancel all tasks and clean up resources
        if client_listener_task and not client_listener_task.done():
            client_listener_task.cancel()
            try:
                await client_listener_task
            except asyncio.CancelledError:
                pass
        
        if connection_id:
            # Clean up command handler resources (preview tasks, etc.)
            await command_handler.cleanup_connection(connection_id)
            await connection_manager.remove_connection(connection_id)
            
        logger.info(
            f"MCP v3 WebSocket cleanup completed - connection_id: {connection_id}"
        )


async def _handle_client_messages(
    websocket: WebSocket, 
    connection_id: str, 
    cmd_handler
) -> None:
    """
    Handle incoming JSON command messages from client.
    
    This function implements the Command & Analysis Channel of the dual-channel architecture.
    It processes JSON commands and delegates to command_handler for execution.
    
    Supported Commands:
    - subscribe_preview: Start continuous low-quality binary stream
    - request_hq_frame: Request single high-quality frame (binary response)
    - unsubscribe: Stop preview stream
    - ping: Health check
    """
    from ..models.requests import WebSocketCommand
    from ..models.responses import WebSocketResponse, ResponseType
    
    try:
        while True:
            # Receive message (expecting JSON text commands only)
            message = await websocket.receive()
            
            # Update connection activity
            await connection_manager.update_activity(connection_id)
            
            if "text" in message:
                # Process JSON command message
                try:
                    command_data = json.loads(message["text"])
                    command = WebSocketCommand(**command_data)
                    
                    logger.debug(
                        "Received MCP v3 command",
                        connection_id=connection_id,
                        command=command.command.value,
                        request_id=command.request_id,
                        stream_id=getattr(command, 'stream_id', None)
                    )
                    
                    # Delegate command processing to command handler
                    # The command handler will:
                    # 1. Process the command (subscribe_preview, request_hq_frame, etc.)
                    # 2. Send binary data directly via websocket.send_bytes() when needed
                    # 3. Return JSON response for acknowledgments/errors
                    response = await cmd_handler.handle_command(
                        websocket, connection_id, command
                    )
                    
                    # Send JSON response if command handler returns one
                    if response:
                        await websocket.send_text(response.json())
                        logger.debug(
                            "Sent command response",
                            connection_id=connection_id,
                            response_type=response.type.value,
                            request_id=command.request_id
                        )
                        
                except json.JSONDecodeError as e:
                    logger.error(
                        "Invalid JSON received",
                        connection_id=connection_id,
                        error=str(e)
                    )
                    # Send error response
                    error_response = WebSocketResponse(
                        type=ResponseType.ERROR,
                        success=False,
                        message=f"Invalid JSON: {str(e)}",
                        timestamp=datetime.now().isoformat()
                    )
                    await websocket.send_text(error_response.json())
                    
                except Exception as e:
                    logger.error(
                        "Command processing error",
                        connection_id=connection_id,
                        error=str(e),
                        exc_info=True
                    )
                    # Send error response
                    error_response = WebSocketResponse(
                        type=ResponseType.ERROR,
                        success=False,
                        message=f"Command failed: {str(e)}",
                        timestamp=datetime.now().isoformat()
                    )
                    await websocket.send_text(error_response.json())
                    
            elif "bytes" in message:
                # Binary data received (not expected in MCP v3 command channel)
                logger.warning(
                    "Unexpected binary data received on command channel",
                    connection_id=connection_id,
                    size=len(message["bytes"])
                )
                # Send error response
                error_response = WebSocketResponse(
                    type=ResponseType.ERROR,
                    success=False,
                    message="Binary data not supported on command channel",
                    timestamp=datetime.now().isoformat()
                )
                await websocket.send_text(error_response.json())
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected - connection_id: {connection_id}")
        raise
    except Exception as e:
        logger.error(
            f"Client message handler error - connection_id: {connection_id}, error: {str(e)}",
            exc_info=True
        )
        raise


@ws_router.websocket("/test")
async def websocket_test(websocket: WebSocket):
    """Simple WebSocket test endpoint."""
    await websocket.accept()
    await websocket.send_text('{"type": "connected", "message": "Test OK"}')


@ws_router.websocket("/{client_id}")
async def websocket_client_legacy(websocket: WebSocket, client_id: str):
    """Legacy WebSocket endpoint for backward compatibility."""
    await websocket.accept()
    
    connection_id = None
    try:
        # Add connection with WebSocket object
        connection_id = await connection_manager.add_connection(
            client_ip=websocket.client.host,
            user_agent=websocket.headers.get("user-agent"),
            metadata={"client_id": client_id, "protocol": "legacy"}
        )
        
        # Store WebSocket reference for broadcasting
        conn = await connection_manager.get_connection(connection_id)
        if conn:
            conn.websocket = websocket
        
        await websocket.send_json({
            "type": "connected",
            "connection_id": connection_id,
            "client_id": client_id
        })
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_json()
                
                # Update activity
                await connection_manager.update_activity(connection_id)
                
                # Handle different message types
                msg_type = message.get("type")
                
                if msg_type == "subscribe":
                    stream_id = message.get("stream_id")
                    if stream_id:
                        success = await connection_manager.add_to_stream(connection_id, stream_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "stream_id": stream_id,
                            "success": success
                        })
                        
                elif msg_type == "unsubscribe":
                    stream_id = message.get("stream_id")
                    if stream_id:
                        success = await connection_manager.remove_from_stream(connection_id, stream_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "stream_id": stream_id,
                            "success": success
                        })
                        
                else:
                    # Echo back for testing
                    await websocket.send_json({
                        "type": "echo",
                        "data": message,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        logger.error("Legacy WebSocket error", error=str(e))
    finally:
        if connection_id:
            await connection_manager.remove_connection(connection_id)


# Screen Analysis Endpoint
@api_router.post("/analyze/screen")
async def analyze_screen(request: ScreenAnalysisRequest):
    """
    Analyze screen content using AI vision.
    
    Captures the current screen and analyzes it with AI vision.
    """
    try:
        from ..core.screen_capture import ScreenCapture
        
        # Initialize components
        screen_capture = ScreenCapture()
        
        # Validate AI service
        if not ai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service not configured or unavailable"
            )
        
        # Capture screen
        capture_result = await screen_capture.capture_screen(
            monitor=request.monitor,
            region=request.region,
            quality=80,
            format="jpeg"
        )
        
        if not capture_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to capture screen: {capture_result.get('message', 'Unknown error')}"
            )
        
        # Analyze with AI using unified service
        analysis_result = await ai_service.analyze_image(
            image_base64=capture_result["image_data"],
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens
        )
        
        # Check if analysis was successful
        if not analysis_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"AI analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        # Create response
        return {
            "success": True,
            "message": "Screen analyzed successfully",
            "data": {
                "analysis": analysis_result["response"],
                "model": analysis_result["model"],
                "prompt": request.prompt,
                "capture_info": {
                    "timestamp": datetime.now().isoformat(),
                    "monitor": request.monitor,
                    "image_size": {
                        "width": capture_result["width"],
                        "height": capture_result["height"]
                    },
                    "format": capture_result["format"],
                    "file_size": capture_result["size"]
                },
                "usage": analysis_result.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }),
                "memory_id": analysis_result.get("memory_id")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in screen analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Screen analysis failed: {str(e)}"
        )


# AI Endpoints
@api_router.get("/ai/models")
async def list_ai_models():
    """List available AI models from the configured provider."""
    if not ai_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="AI service not configured. Please set OPENAI_API_KEY."
        )
    
    result = await ai_service.list_models()
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {result.get('error', 'Unknown error')}"
        )
    
    return {
        "success": True,
        "message": "Models retrieved successfully",
        "data": result["models"]
    }


@api_router.post("/ai/analyze")
async def analyze_image(
    image_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze an image using AI vision capabilities.
    
    Request body:
    {
        "image_base64": "base64_encoded_image",
        "prompt": "What's in this image?",
        "model": "gpt-4o-mini",  # optional
        "max_tokens": 300  # optional
    }
    """
    if not ai_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="AI service not configured. Please set OPENAI_API_KEY."
        )
    
    image_base64 = image_data.get("image_base64")
    if not image_base64:
        raise HTTPException(
            status_code=400,
            detail="image_base64 is required"
        )
    
    prompt = image_data.get("prompt", "What's in this image?")
    model = image_data.get("model")
    max_tokens = image_data.get("max_tokens", 300)
    
    result = await ai_service.analyze_image(
        image_base64=image_base64,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
        )
    
    return {
        "success": True,
        "message": "Image analyzed successfully",
        "data": result
    }


@api_router.post("/ai/chat")
async def chat_completion(
    chat_request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate chat completion using any OpenAI compatible model.
    
    Request body:
    {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "model": "gpt-4o-mini",  # optional
        "max_tokens": 1000,  # optional
        "temperature": 0.7  # optional
    }
    """
    if not ai_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="AI service not configured. Please set OPENAI_API_KEY."
        )
    
    messages = chat_request.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(
            status_code=400,
            detail="messages is required and must be a list"
        )
    
    model = chat_request.get("model")
    max_tokens = chat_request.get("max_tokens", 1000)
    temperature = chat_request.get("temperature", 0.7)
    
    result = await ai_service.chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Chat completion failed: {result.get('error', 'Unknown error')}"
        )
    
    return {
        "success": True,
        "message": "Chat completed successfully",
        "data": result
    }


@api_router.get("/ai/status")
async def ai_status():
    """Get AI service configuration status."""
    return {
        "success": True,
        "message": "AI service status retrieved",
        "data": {
            "configured": ai_service.is_available(),
            "base_url": config.openai_base_url or "https://api.openai.com/v1",
            "model": config.openai_model,
            "timeout": config.openai_timeout
        }
    }


@api_router.get("/performance")
async def get_performance_metrics():
    """Get detailed performance metrics and system health."""
    try:
        health_status = await performance_monitor.get_health_status()
        metrics = performance_monitor.get_metrics()
        
        return {
            "success": True,
            "message": "Performance metrics retrieved successfully",
            "health": health_status,
            "detailed_metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "active_connections": metrics.active_connections,
                "active_streams": metrics.active_streams,
                "avg_response_time": metrics.avg_response_time,
                "failed_connections": metrics.failed_connections,
                "data_throughput_mb_s": metrics.data_throughput,
                "last_updated": metrics.last_updated.isoformat()
            },
            "recommendations": _get_performance_recommendations(health_status)
        }
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


def _get_performance_recommendations(health_status: dict) -> list:
    """Get performance recommendations based on current status."""
    recommendations = []
    
    if health_status['health_score'] < 70:
        recommendations.append("Consider reducing the number of concurrent streams")
        recommendations.append("Lower stream quality or FPS to reduce system load")
    
    if "High connection count" in health_status['issues']:
        recommendations.append("Monitor client connections and implement connection pooling")
    
    if "Slow response times" in health_status['issues']:
        recommendations.append("Check system resources and optimize stream processing")
    
    if "High data throughput" in health_status['issues']:
        recommendations.append("Consider implementing adaptive bitrate streaming")
    
    if not recommendations:
        recommendations.append("System is performing well - no immediate action needed")
    
    return recommendations