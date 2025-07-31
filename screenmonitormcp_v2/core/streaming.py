"""Streaming management for ScreenMonitorMCP v2."""

import asyncio
import base64
import io
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, Callable
from datetime import datetime
import structlog
from PIL import Image
import mss
import numpy as np

try:
    from ..models.responses import StreamingEvent
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.responses import StreamingEvent
try:
    from ..server.config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from server.config import config
from .connection import connection_manager

logger = structlog.get_logger()


class StreamManager:
    """Manages real-time streaming operations."""
    
    def __init__(self):
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
    async def create_stream(
        self,
        stream_type: str,
        fps: int = None,
        quality: int = None,
        format: str = "jpeg",
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new stream with enhanced safety controls."""
        stream_id = str(uuid.uuid4())
        
        # Apply defaults from config
        fps = fps or config.default_stream_fps
        quality = quality or config.default_stream_quality
        
        # Validate limits
        fps = min(fps, config.max_stream_fps)
        quality = min(quality, config.max_stream_quality)
        
        # Store original quality for adaptive adjustment
        original_quality = quality
        
        async with self._lock:
            # Check concurrent stream limit
            if len(self._active_streams) >= config.max_concurrent_streams:
                raise ValueError(f"Maximum concurrent streams limit reached: {config.max_concurrent_streams}")
            
            stream_config = {
                "stream_id": stream_id,
                "stream_type": stream_type,
                "fps": fps,
                "quality": quality,
                "original_quality": original_quality,
                "format": format,
                "filters": filters or {},
                "created_at": datetime.now(),
                "status": "created",
                "sequence": 0,
                "performance_stats": {
                    "avg_broadcast_time": 0.0,
                    "failed_sends": 0,
                    "quality_adjustments": 0
                }
            }
            
            self._active_streams[stream_id] = stream_config
            
            logger.info(
                "Stream created with safety controls",
                stream_id=stream_id,
                stream_type=stream_type,
                fps=fps,
                quality=quality,
                max_concurrent=config.max_concurrent_streams
            )
            
            return stream_id
    
    async def start_stream(
        self,
        stream_id: str,
        data_generator: Callable[[str], AsyncGenerator[Dict[str, Any], None]]
    ) -> bool:
        """Start a stream with a data generator."""
        async with self._lock:
            if stream_id not in self._active_streams:
                return False
            
            if stream_id in self._stream_tasks:
                logger.warning("Stream already running", stream_id=stream_id)
                return False
            
            task = asyncio.create_task(
                self._run_stream(stream_id, data_generator)
            )
            self._stream_tasks[stream_id] = task
            
            logger.info("Stream started", stream_id=stream_id)
            return True
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream."""
        async with self._lock:
            if stream_id not in self._active_streams:
                return False
            
            # Cancel the stream task
            if stream_id in self._stream_tasks:
                task = self._stream_tasks.pop(stream_id)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Update stream status
            self._active_streams[stream_id]["status"] = "stopped"
            
            logger.info("Stream stopped", stream_id=stream_id)
            return True
    
    async def pause_stream(self, stream_id: str) -> bool:
        """Pause a stream."""
        async with self._lock:
            if stream_id not in self._active_streams:
                return False
            
            self._active_streams[stream_id]["status"] = "paused"
            logger.info("Stream paused", stream_id=stream_id)
            return True
    
    async def resume_stream(self, stream_id: str) -> bool:
        """Resume a paused stream."""
        async with self._lock:
            if stream_id not in self._active_streams:
                return False
            
            self._active_streams[stream_id]["status"] = "active"
            logger.info("Stream resumed", stream_id=stream_id)
            return True
    
    async def _run_stream(
        self,
        stream_id: str,
        data_generator: Callable[[str], AsyncGenerator[Dict[str, Any], None]]
    ):
        """Run the stream loop with adaptive quality and backpressure control."""
        try:
            stream_config = self._active_streams[stream_id]
            fps = stream_config["fps"]
            interval = 1.0 / fps
            failed_sends = 0
            adaptive_quality = stream_config["quality"]
            
            async for data in data_generator(stream_id):
                if stream_config["status"] != "active":
                    break
                
                # Check frame size and apply limits
                frame_size = len(data.get("image_data", "")) if "image_data" in data else 0
                if frame_size > config.max_frame_size:
                    logger.warning(
                        "Frame size exceeds limit, skipping",
                        stream_id=stream_id,
                        frame_size=frame_size,
                        limit=config.max_frame_size
                    )
                    continue
                
                # Create streaming event
                event = StreamingEvent(
                    event_type="data",
                    data=data,
                    stream_id=stream_id,
                    sequence=stream_config["sequence"]
                )
                
                # Broadcast to all WebSocket connections
                broadcast_data = {
                    "type": "stream_data",
                    "stream_id": stream_id,
                    "sequence": event.sequence,
                    "timestamp": event.timestamp.isoformat(),
                    "data": data,
                    "adaptive_quality": adaptive_quality
                }
                
                # Measure broadcast time for backpressure detection
                start_time = asyncio.get_event_loop().time()
                sent_count = await connection_manager.broadcast_to_stream(
                    stream_id, broadcast_data
                )
                broadcast_time = asyncio.get_event_loop().time() - start_time
                
                # Adaptive quality control based on performance
                if broadcast_time > 0.5:  # If broadcast takes more than 500ms
                    failed_sends += 1
                    if failed_sends > 3 and adaptive_quality > 30:
                        adaptive_quality = max(30, adaptive_quality - 10)
                        logger.info(
                            "Reducing quality due to slow broadcast",
                            stream_id=stream_id,
                            new_quality=adaptive_quality
                        )
                        # Update stream config
                        stream_config["quality"] = adaptive_quality
                else:
                    failed_sends = max(0, failed_sends - 1)
                    # Gradually increase quality if performance is good
                    if failed_sends == 0 and adaptive_quality < stream_config.get("original_quality", 80):
                        adaptive_quality = min(stream_config.get("original_quality", 80), adaptive_quality + 5)
                        stream_config["quality"] = adaptive_quality
                
                logger.debug(
                    "Broadcasted stream data",
                    stream_id=stream_id,
                    sequence=event.sequence,
                    connections_sent=sent_count,
                    broadcast_time=broadcast_time,
                    adaptive_quality=adaptive_quality
                )
                
                stream_config["sequence"] += 1
                
                # Dynamic interval adjustment based on performance
                adjusted_interval = interval
                if broadcast_time > interval:
                    adjusted_interval = max(interval, broadcast_time * 1.2)
                
                await asyncio.sleep(adjusted_interval)
                
        except asyncio.CancelledError:
            logger.info("Stream cancelled", stream_id=stream_id)
        except Exception as e:
            logger.error(
                "Stream error",
                stream_id=stream_id,
                error=str(e),
                exc_info=True
            )
        finally:
            # Clean up
            if stream_id in self._active_streams:
                self._active_streams[stream_id]["status"] = "stopped"
    
    async def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get stream information."""
        return self._active_streams.get(stream_id)
    
    async def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get all active streams."""
        return self._active_streams.copy()
    
    def list_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get all active streams (sync version)."""
        return self._active_streams.copy()
    
    def is_running(self) -> bool:
        """Check if stream manager is running."""
        return len(self._active_streams) > 0

    async def cleanup(self):
        """Clean up all streams."""
        async with self._lock:
            for stream_id in list(self._active_streams.keys()):
                await self.stop_stream(stream_id)
            
            # Wait for all tasks to complete
            if self._stream_tasks:
                await asyncio.gather(
                    *self._stream_tasks.values(),
                    return_exceptions=True
                )
                self._stream_tasks.clear()
            
            self._active_streams.clear()
            logger.info("All streams cleaned up")


class ScreenStreamer:
    """Handles screen capture and streaming operations with dual-channel support."""
    
    def __init__(self):
        self._mss_instance = None
        self._executor = None
        
    async def _get_mss(self):
        """Get MSS instance."""
        if self._mss_instance is None:
            self._mss_instance = mss.mss()
        return self._mss_instance
    
    async def capture_screen(
        self,
        monitor: int = 1,
        region: Optional[Dict[str, int]] = None,
        quality: int = 80,
        format: str = "jpeg",
        resolution: Optional[tuple] = None,
        return_bytes: bool = False
    ) -> Dict[str, Any]:
        """Capture screen and return as base64."""
        try:
            mss_instance = await self._get_mss()
            
            if region:
                # Capture specific region
                monitor_info = {
                    "top": region.get("top", 0),
                    "left": region.get("left", 0),
                    "width": region.get("width", 1920),
                    "height": region.get("height", 1080)
                }
            else:
                # Capture entire monitor
                monitors = mss_instance.monitors
                if not (0 < monitor < len(monitors)):
                    return {
                        "success": False,
                        "message": f"Invalid monitor number: {monitor}. Available monitors: {len(monitors) - 1}"
                    }
                monitor_info = monitors[monitor]

            # Capture screenshot
            screenshot = mss_instance.grab(monitor_info)
            
            # Convert to PIL Image
            img = Image.frombytes(
                "RGB",
                (screenshot.width, screenshot.height),
                screenshot.rgb
            )
            
            # Enhanced resizing with quality-based optimization
            if resolution:
                img = img.resize(resolution, Image.Resampling.LANCZOS)
            else:
                # Adaptive resizing based on quality and performance
                if quality <= 30:  # Very low quality for poor connections
                    max_size = 480
                elif quality <= 50:  # Low quality for preview
                    max_size = 720
                elif quality <= 70:  # Medium quality
                    max_size = 1280
                else:  # High quality
                    max_size = 1920
                    
                if max(img.width, img.height) > max_size:
                    ratio = max_size / max(img.width, img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhanced compression with size validation
            buffer = io.BytesIO()
            if format.lower() == "jpeg":
                # Progressive JPEG for better streaming
                img.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
            else:
                img.save(buffer, format="PNG", optimize=True, compress_level=6)
            
            # Check if compressed size exceeds limits
            compressed_size = len(buffer.getvalue())
            max_size_bytes = getattr(config, 'max_frame_size', 2 * 1024 * 1024)
            
            if compressed_size > max_size_bytes:
                # Re-compress with lower quality if size is too large
                reduced_quality = max(20, quality - 20)
                buffer = io.BytesIO()
                if format.lower() == "jpeg":
                    img.save(buffer, format="JPEG", quality=reduced_quality, optimize=True, progressive=True)
                else:
                    # Convert to JPEG if PNG is too large
                    img.save(buffer, format="JPEG", quality=reduced_quality, optimize=True, progressive=True)
                    format = "JPEG"
                
                compressed_size = len(buffer.getvalue())
                logger.info(
                    "Reduced image quality due to size limit",
                    original_quality=quality,
                    reduced_quality=reduced_quality,
                    final_size=compressed_size
                )
            
            img_bytes = buffer.getvalue()
            
            result = {
                "success": True,
                "width": img.width,
                "height": img.height,
                "format": format.upper(),
                "size": len(img_bytes),
                "monitor": monitor,
                "timestamp": datetime.now().isoformat()
            }
            
            if return_bytes:
                result["image_bytes"] = img_bytes
            else:
                # Encode to base64 for backward compatibility
                img_base64 = base64.b64encode(img_bytes).decode()
                result["image_data"] = img_base64
            
            return result
            
        except Exception as e:
            logger.error("Screen capture failed", error=str(e), exc_info=True)
            return {"success": False, "message": str(e)}
    
    async def stream_screen(
        self,
        stream_id: str,
        fps: int = 2,
        quality: int = 80,
        monitor: int = 1,
        region: Optional[Dict[str, int]] = None,
        resolution: Optional[tuple] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate screen stream data."""
        try:
            while True:
                # Capture screen
                screen_data = await self.capture_screen(
                    monitor=monitor,
                    region=region,
                    quality=quality,
                    resolution=resolution
                )
                
                yield screen_data
                
                # Control FPS
                await asyncio.sleep(1.0 / fps)
                
        except asyncio.CancelledError:
            logger.info("Screen stream cancelled", stream_id=stream_id)
            raise
        except Exception as e:
            logger.error("Screen stream error", error=str(e), exc_info=True)
            raise


# Global instances
stream_manager = StreamManager()
screen_streamer = ScreenStreamer()
