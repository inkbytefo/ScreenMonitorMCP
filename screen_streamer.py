"""
Real-time Screen Streaming Module for ScreenMonitorMCP

This module provides real-time screen capture streaming capabilities with base64 encoding
for MCP clients like Claude Desktop. It supports multiple concurrent streams with
configurable quality, FPS, and format settings.

Author: inkbytefo
"""

import time
import uuid
import base64
import threading
import cv2
import numpy as np
import mss
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class StreamFrame:
    """Represents a single frame in the stream"""
    timestamp: datetime
    data: str  # base64 encoded image data
    format: str  # jpeg or png
    frame_number: int
    size_bytes: int
    width: int
    height: int

@dataclass
class StreamConfig:
    """Configuration for a screen stream"""
    fps: int = 5
    quality: int = 70  # JPEG quality (0-100)
    format: Literal["jpeg", "png"] = "jpeg"
    scale: float = 1.0  # Scale factor for resolution
    capture_mode: Literal["all", "monitor", "window", "region"] = "all"
    monitor_number: int = 1
    region: Optional[Dict[str, int]] = None
    max_buffer_size: int = 30  # Maximum frames to keep in buffer
    change_detection: bool = True  # Only capture frames when changes detected
    change_threshold: float = 0.05  # Minimum change percentage to trigger capture (5%)
    adaptive_quality: bool = True  # Automatically adjust quality based on content
    max_frame_size_kb: int = 500  # Maximum frame size in KB
    # AI Analysis triggers
    auto_analysis: bool = False  # Enable automatic AI analysis
    analysis_triggers: List[str] = field(default_factory=list)  # Triggers for AI analysis
    analysis_threshold: float = 0.1  # Threshold for triggering analysis (10% change)
    analysis_prompt: str = "Bu frame'de önemli değişiklikler var mı?"

class ScreenStreamer:
    """
    Real-time screen streaming class that captures screen continuously
    and provides base64 encoded frames for MCP clients.
    """
    
    def __init__(self, stream_id: str, config: StreamConfig):
        self.stream_id = stream_id
        self.config = config
        self.is_streaming = False
        self.frame_buffer: deque[StreamFrame] = deque(maxlen=config.max_buffer_size)
        self.stream_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'start_time': None,
            'frames_captured': 0,
            'current_fps': 0.0,
            'last_frame_time': None,
            'total_bytes_captured': 0,
            'average_frame_size': 0,
            'errors_count': 0
        }
        
        # Performance tracking
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._last_capture_time = 0

        # Change detection
        self._previous_frame_hash = None
        self._previous_frame_gray = None
        self._frames_skipped = 0

        # AI Analysis
        self._analysis_callbacks: List[callable] = []
        self._last_analysis_time = None
        self._analysis_queue = []
        
        logger.info("ScreenStreamer initialized", 
                   stream_id=stream_id, 
                   config=config.__dict__)
    
    def start(self) -> bool:
        """Start the streaming process"""
        if self.is_streaming:
            logger.warning("Stream already running", stream_id=self.stream_id)
            return False
        
        try:
            self.is_streaming = True
            self.stats['start_time'] = datetime.now()
            self.stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.stream_thread.start()
            
            logger.info("Stream started successfully", stream_id=self.stream_id)
            return True
            
        except Exception as e:
            logger.error("Failed to start stream", stream_id=self.stream_id, error=str(e))
            self.is_streaming = False
            return False
    
    def stop(self) -> bool:
        """Stop the streaming process"""
        if not self.is_streaming:
            logger.warning("Stream not running", stream_id=self.stream_id)
            return False
        
        try:
            self.is_streaming = False
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
            
            logger.info("Stream stopped successfully", 
                       stream_id=self.stream_id,
                       total_frames=self.stats['frames_captured'])
            return True
            
        except Exception as e:
            logger.error("Error stopping stream", stream_id=self.stream_id, error=str(e))
            return False
    
    def get_current_frame(self) -> Optional[StreamFrame]:
        """Get the most recent frame from the buffer"""
        if not self.frame_buffer:
            return None
        return self.frame_buffer[-1]
    
    def get_frame_history(self, count: int = 5) -> List[StreamFrame]:
        """Get the last N frames from the buffer"""
        if not self.frame_buffer:
            return []
        return list(self.frame_buffer)[-count:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive stream status and statistics"""
        return {
            'stream_id': self.stream_id,
            'is_active': self.is_streaming,
            'config': {
                'fps': self.config.fps,
                'quality': self.config.quality,
                'format': self.config.format,
                'scale': self.config.scale,
                'capture_mode': self.config.capture_mode,
                'monitor_number': self.config.monitor_number,
                'region': self.config.region
            },
            'stats': self.stats.copy(),
            'buffer_info': {
                'current_size': len(self.frame_buffer),
                'max_size': self.config.max_buffer_size,
                'oldest_frame': self.frame_buffer[0].timestamp.isoformat() if self.frame_buffer else None,
                'newest_frame': self.frame_buffer[-1].timestamp.isoformat() if self.frame_buffer else None
            }
        }
    
    def _streaming_loop(self):
        """Main streaming loop that runs in a separate thread"""
        logger.info("Streaming loop started", stream_id=self.stream_id)
        
        try:
            with mss.mss() as sct:
                capture_area = self._get_capture_area(sct)
                
                while self.is_streaming:
                    loop_start = time.time()
                    
                    try:
                        # Capture screen
                        sct_img = sct.grab(capture_area)
                        frame = np.array(sct_img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                        # Process frame
                        processed_frame = self._process_frame(frame)

                        # Check for significant changes
                        has_change, change_percentage = self._detect_frame_change(processed_frame)
                        if not has_change:
                            # Skip this frame due to no significant changes
                            self._control_fps(loop_start)
                            continue

                        # Get adaptive quality for this frame
                        current_quality = self._adaptive_quality_adjustment(processed_frame)

                        # Encode to base64 with adaptive quality
                        encoded_data, size_bytes = self._encode_frame(processed_frame, current_quality)

                        # Check frame size limit
                        if size_bytes > self.config.max_frame_size_kb * 1024:
                            # Reduce quality and re-encode if frame is too large
                            reduced_quality = max(current_quality - 20, 20)
                            encoded_data, size_bytes = self._encode_frame(processed_frame, reduced_quality)
                            logger.debug("Frame size reduced",
                                       original_size_kb=size_bytes // 1024,
                                       quality_reduced_to=reduced_quality)

                        # Create frame object
                        stream_frame = StreamFrame(
                            timestamp=datetime.now(),
                            data=encoded_data,
                            format=self.config.format,
                            frame_number=self.stats['frames_captured'],
                            size_bytes=size_bytes,
                            width=processed_frame.shape[1],
                            height=processed_frame.shape[0]
                        )

                        # Add to buffer
                        self.frame_buffer.append(stream_frame)

                        # Check if AI analysis should be triggered
                        if self._should_trigger_analysis(processed_frame, change_percentage):
                            self._trigger_analysis(stream_frame)

                        # Update statistics
                        self._update_stats(stream_frame, loop_start)

                        # Control FPS
                        self._control_fps(loop_start)
                        
                    except Exception as e:
                        logger.error("Error in streaming loop", 
                                   stream_id=self.stream_id, 
                                   error=str(e))
                        self.stats['errors_count'] += 1
                        time.sleep(0.1)  # Prevent tight error loop
                        
        except Exception as e:
            logger.error("Fatal error in streaming loop", 
                        stream_id=self.stream_id, 
                        error=str(e))
        finally:
            logger.info("Streaming loop ended", stream_id=self.stream_id)
    
    def _get_capture_area(self, sct) -> Dict[str, int]:
        """Get the capture area based on configuration"""
        if self.config.capture_mode == "all":
            return sct.monitors[0]  # All monitors
        elif self.config.capture_mode == "monitor":
            if self.config.monitor_number <= len(sct.monitors) - 1:
                return sct.monitors[self.config.monitor_number]
            else:
                logger.warning("Invalid monitor number, using primary", 
                             monitor_number=self.config.monitor_number)
                return sct.monitors[0]
        elif self.config.capture_mode == "region" and self.config.region:
            return {
                "top": self.config.region.get("y", 0),
                "left": self.config.region.get("x", 0),
                "width": self.config.region.get("width", 800),
                "height": self.config.region.get("height", 600)
            }
        else:
            return sct.monitors[0]
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the captured frame (scaling, etc.)"""
        if self.config.scale != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.config.scale)
            new_height = int(height * self.config.scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _encode_frame(self, frame: np.ndarray, quality: Optional[int] = None) -> tuple[str, int]:
        """Encode frame to base64 string with optional quality override"""
        actual_quality = quality if quality is not None else self.config.quality

        if self.config.format == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), actual_quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
        else:  # png
            # For PNG, we can use compression level (0-9)
            compression_level = max(0, min(9, int((100 - actual_quality) / 11)))
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level]
            _, buffer = cv2.imencode('.png', frame, encode_param)

        base64_data = base64.b64encode(buffer).decode('utf-8')
        return base64_data, len(buffer)

    def _detect_frame_change(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect if frame has significant changes compared to previous frame"""
        if not self.config.change_detection:
            return True, 0.0  # Always capture if change detection is disabled

        # Convert to grayscale for comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._previous_frame_gray is None:
            self._previous_frame_gray = gray_frame
            return True, 0.0  # First frame, always capture

        # Calculate structural similarity
        try:
            # Resize frames to same size if needed
            if gray_frame.shape != self._previous_frame_gray.shape:
                gray_frame = cv2.resize(gray_frame,
                                      (self._previous_frame_gray.shape[1],
                                       self._previous_frame_gray.shape[0]))

            # Calculate mean squared error
            mse = np.mean((gray_frame - self._previous_frame_gray) ** 2)

            # Normalize MSE to percentage
            max_possible_mse = 255 ** 2
            change_percentage = mse / max_possible_mse

            # Check if change exceeds threshold
            has_significant_change = change_percentage > self.config.change_threshold

            if has_significant_change:
                self._previous_frame_gray = gray_frame
                return True, change_percentage
            else:
                self._frames_skipped += 1
                return False, change_percentage

        except Exception as e:
            logger.warning("Change detection failed", error=str(e))
            return True, 0.0  # Fallback to capturing frame

    def _adaptive_quality_adjustment(self, frame: np.ndarray) -> int:
        """Dynamically adjust quality based on frame content complexity"""
        if not self.config.adaptive_quality:
            return self.config.quality

        try:
            # Calculate image complexity using Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Adjust quality based on complexity
            if laplacian_var > 1000:  # High detail image
                return min(self.config.quality + 10, 95)
            elif laplacian_var < 100:  # Low detail image
                return max(self.config.quality - 15, 30)
            else:
                return self.config.quality

        except Exception as e:
            logger.warning("Adaptive quality adjustment failed", error=str(e))
            return self.config.quality

    def _update_stats(self, frame: StreamFrame, loop_start: float):
        """Update streaming statistics"""
        self.stats['frames_captured'] += 1
        self.stats['last_frame_time'] = frame.timestamp
        self.stats['total_bytes_captured'] += frame.size_bytes
        self.stats['average_frame_size'] = (
            self.stats['total_bytes_captured'] / self.stats['frames_captured']
        )

        # Add frames skipped to stats
        if not hasattr(self.stats, 'frames_skipped'):
            self.stats['frames_skipped'] = 0
        self.stats['frames_skipped'] = self._frames_skipped

        # Update FPS calculation
        self._fps_counter += 1
        current_time = time.time()
        if current_time - self._fps_timer >= 1.0:
            self.stats['current_fps'] = self._fps_counter / (current_time - self._fps_timer)
            self._fps_counter = 0
            self._fps_timer = current_time

        # Calculate processing time
        processing_time = time.time() - loop_start
        if not hasattr(self.stats, 'average_processing_time'):
            self.stats['average_processing_time'] = processing_time
        else:
            # Exponential moving average
            self.stats['average_processing_time'] = (
                0.9 * self.stats['average_processing_time'] + 0.1 * processing_time
            )
    
    def _control_fps(self, loop_start: float):
        """Control the frame rate to match target FPS"""
        target_interval = 1.0 / self.config.fps
        elapsed = time.time() - loop_start
        sleep_time = max(0, target_interval - elapsed)
        
        if sleep_time > 0:
            time.sleep(sleep_time)

    def add_analysis_callback(self, callback: callable):
        """Add a callback function for AI analysis triggers"""
        self._analysis_callbacks.append(callback)
        logger.info("Analysis callback added", stream_id=self.stream_id)

    def remove_analysis_callback(self, callback: callable):
        """Remove an analysis callback"""
        if callback in self._analysis_callbacks:
            self._analysis_callbacks.remove(callback)
            logger.info("Analysis callback removed", stream_id=self.stream_id)

    def _should_trigger_analysis(self, frame: np.ndarray, change_percentage: float) -> bool:
        """Determine if AI analysis should be triggered"""
        if not self.config.auto_analysis:
            return False

        # Check if enough time has passed since last analysis
        if self._last_analysis_time:
            time_since_last = time.time() - self._last_analysis_time
            if time_since_last < 5.0:  # Minimum 5 seconds between analyses
                return False

        # Check change threshold
        if change_percentage > self.config.analysis_threshold:
            return True

        return False

    def _trigger_analysis(self, frame: StreamFrame):
        """Trigger AI analysis for the given frame"""
        if not self._analysis_callbacks:
            return

        self._last_analysis_time = time.time()

        # Call all registered callbacks
        for callback in self._analysis_callbacks:
            try:
                callback(self.stream_id, frame, self.config.analysis_prompt)
            except Exception as e:
                logger.error("Analysis callback failed",
                           stream_id=self.stream_id,
                           error=str(e))


class StreamManager:
    """
    Manages multiple screen streams with automatic cleanup and resource management.
    """

    def __init__(self):
        self.streams: Dict[str, ScreenStreamer] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.max_concurrent_streams = 5  # Limit concurrent streams

        logger.info("StreamManager initialized")

    def create_stream(self,
                     fps: int = 5,
                     quality: int = 70,
                     format: Literal["jpeg", "png"] = "jpeg",
                     scale: float = 1.0,
                     capture_mode: Literal["all", "monitor", "window", "region"] = "all",
                     monitor_number: int = 1,
                     region: Optional[Dict[str, int]] = None,
                     change_detection: bool = True,
                     change_threshold: float = 0.05,
                     adaptive_quality: bool = True,
                     max_frame_size_kb: int = 500) -> Optional[str]:
        """Create a new screen stream"""

        # Check concurrent stream limit
        active_streams = sum(1 for s in self.streams.values() if s.is_streaming)
        if active_streams >= self.max_concurrent_streams:
            logger.warning("Maximum concurrent streams reached",
                          active=active_streams,
                          max=self.max_concurrent_streams)
            return None

        # Generate unique stream ID
        stream_id = str(uuid.uuid4())

        # Create stream configuration
        config = StreamConfig(
            fps=fps,
            quality=quality,
            format=format,
            scale=scale,
            capture_mode=capture_mode,
            monitor_number=monitor_number,
            region=region,
            change_detection=change_detection,
            change_threshold=change_threshold,
            adaptive_quality=adaptive_quality,
            max_frame_size_kb=max_frame_size_kb
        )

        # Create and start streamer
        try:
            streamer = ScreenStreamer(stream_id, config)
            if streamer.start():
                self.streams[stream_id] = streamer
                logger.info("Stream created successfully",
                           stream_id=stream_id,
                           total_streams=len(self.streams))
                return stream_id
            else:
                logger.error("Failed to start stream", stream_id=stream_id)
                return None

        except Exception as e:
            logger.error("Error creating stream", error=str(e))
            return None

    def get_stream(self, stream_id: str) -> Optional[ScreenStreamer]:
        """Get a stream by ID"""
        return self.streams.get(stream_id)

    def stop_stream(self, stream_id: str) -> bool:
        """Stop and remove a stream"""
        streamer = self.streams.get(stream_id)
        if not streamer:
            logger.warning("Stream not found", stream_id=stream_id)
            return False

        try:
            success = streamer.stop()
            if success:
                del self.streams[stream_id]
                logger.info("Stream stopped and removed",
                           stream_id=stream_id,
                           remaining_streams=len(self.streams))
            return success

        except Exception as e:
            logger.error("Error stopping stream", stream_id=stream_id, error=str(e))
            return False

    def get_all_streams_status(self) -> Dict[str, Any]:
        """Get status of all streams"""
        return {
            'total_streams': len(self.streams),
            'active_streams': sum(1 for s in self.streams.values() if s.is_streaming),
            'streams': {
                stream_id: streamer.get_status()
                for stream_id, streamer in self.streams.items()
            }
        }

    def cleanup_inactive_streams(self) -> int:
        """Clean up inactive streams and return number of cleaned streams"""
        current_time = time.time()

        # Check if cleanup is needed
        if current_time - self.last_cleanup < self.cleanup_interval:
            return 0

        self.last_cleanup = current_time
        inactive_streams = []

        # Find inactive streams
        for stream_id, streamer in self.streams.items():
            if not streamer.is_streaming:
                inactive_streams.append(stream_id)
            elif streamer.stats['last_frame_time']:
                # Check if stream has been inactive for too long
                last_frame_time = streamer.stats['last_frame_time']
                if isinstance(last_frame_time, datetime):
                    time_since_last_frame = (datetime.now() - last_frame_time).total_seconds()
                    if time_since_last_frame > 600:  # 10 minutes
                        inactive_streams.append(stream_id)

        # Clean up inactive streams
        cleaned_count = 0
        for stream_id in inactive_streams:
            if self.stop_stream(stream_id):
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info("Cleaned up inactive streams", count=cleaned_count)

        return cleaned_count

    def stop_all_streams(self) -> int:
        """Stop all streams and return count of stopped streams"""
        stream_ids = list(self.streams.keys())
        stopped_count = 0

        for stream_id in stream_ids:
            if self.stop_stream(stream_id):
                stopped_count += 1

        logger.info("All streams stopped", count=stopped_count)
        return stopped_count


# Global stream manager instance
_global_stream_manager: Optional[StreamManager] = None

def get_global_stream_manager() -> StreamManager:
    """Get the global stream manager instance"""
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = StreamManager()
    return _global_stream_manager

def set_global_stream_manager(manager: StreamManager):
    """Set the global stream manager instance"""
    global _global_stream_manager
    _global_stream_manager = manager
