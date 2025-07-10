"""
Enhanced Streaming Analysis System
Revolutionary feature: Continuous AI-powered screen analysis with smart streaming
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import mss
from typing import Dict, List, Optional, Callable, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import structlog
from collections import deque
import json

logger = structlog.get_logger()

@dataclass
class AnalysisResult:
    """AI analysis result"""
    timestamp: datetime
    analysis_text: str
    confidence: float
    screenshot_base64: Optional[str] = None
    change_type: str = "unknown"
    processing_time: float = 0.0

@dataclass
class StreamingConfig:
    """Streaming analysis configuration"""
    mode: Literal["smart", "continuous", "change_triggered"] = "smart"
    fps: int = 2
    analysis_interval: int = 5  # seconds
    analysis_prompt: str = "Analyze current screen activity and describe what's happening"
    adaptive: bool = True
    max_analysis_history: int = 50
    ring_buffer_size: int = 10
    min_change_threshold: float = 0.05
    ai_analysis_threshold: float = 0.15

class RingBuffer:
    """Ring buffer for storing recent frames"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def add_frame(self, frame: np.ndarray, timestamp: datetime):
        """Add frame to ring buffer"""
        self.buffer.append(frame.copy())
        self.timestamps.append(timestamp)
    
    def get_latest_frames(self, count: int = 1) -> List[tuple]:
        """Get latest N frames with timestamps"""
        if count > len(self.buffer):
            count = len(self.buffer)
        
        frames = []
        for i in range(count):
            idx = -(count - i)
            frames.append((self.buffer[idx], self.timestamps[idx]))
        
        return frames
    
    def get_frame_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64"""
        _, buffer = cv2.imencode('.png', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.timestamps.clear()

class AdaptiveFPSController:
    """Adaptive FPS controller based on activity"""
    
    def __init__(self, base_fps: int = 2, max_fps: int = 5, min_fps: int = 1):
        self.base_fps = base_fps
        self.max_fps = max_fps
        self.min_fps = min_fps
        self.current_fps = base_fps
        self.activity_history = deque(maxlen=10)
        self.last_adjustment = datetime.now()
    
    def update_activity(self, change_percentage: float):
        """Update activity level"""
        self.activity_history.append(change_percentage)
        
        # Adjust FPS every 10 seconds
        if datetime.now() - self.last_adjustment > timedelta(seconds=10):
            self._adjust_fps()
            self.last_adjustment = datetime.now()
    
    def _adjust_fps(self):
        """Adjust FPS based on recent activity"""
        if not self.activity_history:
            return
        
        avg_activity = sum(self.activity_history) / len(self.activity_history)
        
        if avg_activity > 0.3:  # High activity
            self.current_fps = min(self.max_fps, self.current_fps + 1)
        elif avg_activity < 0.05:  # Low activity
            self.current_fps = max(self.min_fps, self.current_fps - 1)
        else:  # Medium activity
            self.current_fps = self.base_fps
        
        logger.debug("FPS adjusted", 
                    avg_activity=avg_activity, 
                    new_fps=self.current_fps)
    
    def get_sleep_time(self) -> float:
        """Get sleep time for current FPS"""
        return 1.0 / self.current_fps

class StreamingAnalyzer:
    """Main streaming analysis system"""
    
    def __init__(self, config: StreamingConfig, ai_provider=None):
        self.config = config
        self.ai_provider = ai_provider
        self.is_streaming = False
        self.stream_thread = None
        
        # Components
        self.ring_buffer = RingBuffer(config.ring_buffer_size)
        self.fps_controller = AdaptiveFPSController(config.fps)
        
        # State
        self.analysis_history: List[AnalysisResult] = []
        self.last_analysis_time = datetime.now()
        self.previous_frame = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_analyses': 0,
            'start_time': None,
            'last_activity': None,
            'avg_processing_time': 0.0
        }
        
        # Callbacks
        self.analysis_callbacks: List[Callable] = []
    
    def add_analysis_callback(self, callback: Callable[[AnalysisResult], None]):
        """Add callback for analysis results"""
        self.analysis_callbacks.append(callback)
    
    def start_streaming(self) -> Dict[str, Any]:
        """Start streaming analysis"""
        if self.is_streaming:
            return {"status": "already_running", "message": "Streaming analysis already running"}
        
        if not self.ai_provider:
            logger.warning("AI provider not configured - streaming will work without AI analysis")
            # Continue without AI analysis - streaming still works for monitoring
        
        self.is_streaming = True
        self.stats['start_time'] = datetime.now()
        self.stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.stream_thread.start()
        
        logger.info("Streaming analysis started", 
                   mode=self.config.mode, 
                   fps=self.config.fps)
        
        return {
            "status": "started",
            "message": f"Streaming analysis started in {self.config.mode} mode",
            "config": {
                "mode": self.config.mode,
                "fps": self.config.fps,
                "analysis_interval": self.config.analysis_interval,
                "adaptive": self.config.adaptive
            }
        }
    
    def stop_streaming(self) -> Dict[str, Any]:
        """Stop streaming analysis"""
        if not self.is_streaming:
            return {"status": "not_running", "message": "Streaming analysis not running"}
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=3.0)
        
        duration = datetime.now() - self.stats['start_time']
        
        logger.info("Streaming analysis stopped",
                   duration=str(duration),
                   total_frames=self.stats['total_frames'],
                   total_analyses=self.stats['total_analyses'])
        
        return {
            "status": "stopped",
            "message": "Streaming analysis stopped",
            "stats": {
                **self.stats,
                "duration": str(duration),
                "analysis_rate": self.stats['total_analyses'] / max(1, self.stats['total_frames']) * 100
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get streaming status"""
        if not self.is_streaming:
            return {
                "status": "stopped",
                "message": "Streaming analysis not running"
            }
        
        duration = datetime.now() - self.stats['start_time']
        current_fps = self.fps_controller.current_fps if self.config.adaptive else self.config.fps
        
        return {
            "status": "running",
            "mode": self.config.mode,
            "current_fps": current_fps,
            "duration": str(duration),
            "stats": self.stats,
            "recent_analyses": len(self.analysis_history),
            "ring_buffer_frames": len(self.ring_buffer.buffer)
        }
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        recent_analyses = self.analysis_history[-limit:] if self.analysis_history else []
        
        return [
            {
                "timestamp": analysis.timestamp.isoformat(),
                "analysis_text": analysis.analysis_text,
                "confidence": analysis.confidence,
                "change_type": analysis.change_type,
                "processing_time": analysis.processing_time,
                "has_screenshot": analysis.screenshot_base64 is not None
            }
            for analysis in recent_analyses
        ]
    
    def get_latest_frames(self, count: int = 1) -> List[Dict[str, Any]]:
        """Get latest frames from ring buffer"""
        frames_data = self.ring_buffer.get_latest_frames(count)

        return [
            {
                "timestamp": timestamp.isoformat(),
                "frame_base64": self.ring_buffer.get_frame_base64(frame)
            }
            for frame, timestamp in frames_data
        ]

    async def _analyze_frame(self, frame: np.ndarray, change_type: str = "unknown") -> Optional[AnalysisResult]:
        """Analyze frame with AI"""
        if not self.ai_provider:
            return None

        start_time = time.time()

        try:
            # Convert frame to base64
            frame_base64 = self.ring_buffer.get_frame_base64(frame)

            # AI analysis
            analysis = await self.ai_provider.analyze_image(
                image_base64=frame_base64,
                prompt=self.config.analysis_prompt,
                max_tokens=500
            )

            processing_time = time.time() - start_time

            result = AnalysisResult(
                timestamp=datetime.now(),
                analysis_text=analysis,
                confidence=0.8,  # Default confidence
                screenshot_base64=frame_base64,
                change_type=change_type,
                processing_time=processing_time
            )

            # Update statistics
            self.stats['total_analyses'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['total_analyses'] - 1) + processing_time)
                / self.stats['total_analyses']
            )

            # Add to history
            self.analysis_history.append(result)
            if len(self.analysis_history) > self.config.max_analysis_history:
                self.analysis_history.pop(0)

            # Call callbacks
            for callback in self.analysis_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error("Analysis callback error", error=str(e))

            logger.debug("Frame analyzed",
                        processing_time=processing_time,
                        change_type=change_type)

            return result

        except Exception as e:
            logger.error("Frame analysis failed", error=str(e))
            return None

    def _detect_change(self, current_frame: np.ndarray) -> tuple[float, str]:
        """Detect change between frames"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return 0.0, "initial"

        # Calculate difference
        diff = cv2.absdiff(self.previous_frame, current_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Calculate change percentage
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels

        # Classify change
        if change_percentage < self.config.min_change_threshold:
            change_type = "none"
        elif change_percentage < self.config.ai_analysis_threshold:
            change_type = "minor"
        elif change_percentage < 0.3:
            change_type = "moderate"
        elif change_percentage < 0.6:
            change_type = "major"
        else:
            change_type = "critical"

        self.previous_frame = current_frame.copy()
        return change_percentage, change_type

    def _should_analyze(self, change_percentage: float, change_type: str) -> bool:
        """Determine if frame should be analyzed"""
        now = datetime.now()

        if self.config.mode == "continuous":
            # Analyze at regular intervals
            return (now - self.last_analysis_time).seconds >= self.config.analysis_interval

        elif self.config.mode == "change_triggered":
            # Analyze only on significant changes
            return change_percentage >= self.config.ai_analysis_threshold

        elif self.config.mode == "smart":
            # Smart combination
            time_since_last = (now - self.last_analysis_time).seconds

            # Force analysis after interval
            if time_since_last >= self.config.analysis_interval * 2:
                return True

            # Analyze on significant changes
            if change_percentage >= self.config.ai_analysis_threshold:
                return True

            # Analyze periodically if there's any activity
            if time_since_last >= self.config.analysis_interval and change_percentage > 0.01:
                return True

        return False

    def _streaming_loop(self):
        """Main streaming loop"""
        logger.info("Streaming loop started")

        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Primary monitor

            while self.is_streaming:
                try:
                    loop_start = time.time()

                    # Capture frame
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Add to ring buffer
                    self.ring_buffer.add_frame(frame, datetime.now())

                    # Detect changes
                    change_percentage, change_type = self._detect_change(frame)

                    # Update adaptive FPS
                    if self.config.adaptive:
                        self.fps_controller.update_activity(change_percentage)

                    # Update statistics
                    self.stats['total_frames'] += 1
                    if change_percentage > self.config.min_change_threshold:
                        self.stats['last_activity'] = datetime.now()

                    # Determine if analysis needed
                    if self._should_analyze(change_percentage, change_type):
                        # Schedule analysis for later (will be handled by callback)
                        # Note: Actual AI analysis will be triggered by callback system
                        self.last_analysis_time = datetime.now()

                    # FPS control
                    elapsed = time.time() - loop_start
                    if self.config.adaptive:
                        sleep_time = max(0, self.fps_controller.get_sleep_time() - elapsed)
                    else:
                        sleep_time = max(0, (1.0 / self.config.fps) - elapsed)

                    time.sleep(sleep_time)

                except Exception as e:
                    logger.error("Streaming loop error", error=str(e))
                    time.sleep(1.0)

# Global streaming analyzer instance
_global_streaming_analyzer: Optional[StreamingAnalyzer] = None

def get_global_streaming_analyzer() -> Optional[StreamingAnalyzer]:
    """Get global streaming analyzer instance"""
    return _global_streaming_analyzer

def set_global_streaming_analyzer(analyzer: StreamingAnalyzer):
    """Set global streaming analyzer instance"""
    global _global_streaming_analyzer
    _global_streaming_analyzer = analyzer
