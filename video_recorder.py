"""
Video Recording and Analysis Module for ScreenMonitorMCP
Provides screen recording capabilities with AI analysis integration
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import mss
from typing import Dict, List, Optional, Callable, Any, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import structlog
import os
import tempfile
from pathlib import Path

logger = structlog.get_logger(__name__)

@dataclass
class VideoRecordingConfig:
    """Video recording configuration"""
    duration: int = 10  # Recording duration in seconds
    fps: int = 2  # Frames per second
    capture_mode: Literal["all", "monitor", "window", "region"] = "all"
    monitor_number: int = 1
    region: Optional[Dict[str, int]] = None
    analysis_type: Literal["summary", "frame_by_frame", "key_moments"] = "summary"
    analysis_prompt: str = "Bu video kaydında ne olduğunu detaylıca analiz et"
    max_tokens: Optional[int] = None
    save_video: bool = False
    output_format: Literal["mp4", "avi"] = "mp4"
    video_quality: int = 80  # Video quality (0-100)
    key_moment_threshold: float = 0.15  # Change threshold for key moments

@dataclass
class VideoFrame:
    """Single video frame data"""
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    change_percentage: float = 0.0
    is_key_moment: bool = False

@dataclass
class VideoAnalysisResult:
    """Video analysis result"""
    analysis_text: str
    analysis_type: str
    total_frames: int
    duration: float
    key_moments: List[Dict[str, Any]]
    processing_time: float
    video_path: Optional[str] = None
    frame_analyses: List[Dict[str, Any]] = field(default_factory=list)

class VideoRecorder:
    """Screen video recording system"""
    
    def __init__(self, config: VideoRecordingConfig):
        self.config = config
        self.is_recording = False
        self.frames: List[VideoFrame] = []
        self.video_writer = None
        self.temp_video_path = None
        self.previous_frame = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'key_moments': 0,
            'start_time': None,
            'end_time': None,
            'recording_duration': 0.0
        }
    
    def _get_capture_area(self) -> Dict[str, int]:
        """Get capture area based on configuration"""
        with mss.mss() as sct:
            if self.config.capture_mode == "all":
                return sct.monitors[0]  # All monitors
            elif self.config.capture_mode == "monitor":
                if self.config.monitor_number <= len(sct.monitors) - 1:
                    return sct.monitors[self.config.monitor_number]
                else:
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
    
    def _detect_change(self, current_frame: np.ndarray) -> float:
        """Detect change percentage between frames"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return 0.0
        
        # Convert to grayscale for comparison
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray_current, gray_previous)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate change percentage
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels
        
        self.previous_frame = current_frame.copy()
        return change_percentage
    
    def _setup_video_writer(self, frame_shape: Tuple[int, int]) -> str:
        """Setup video writer and return temp file path"""
        if not self.config.save_video:
            return None
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screen_recording_{timestamp}.{self.config.output_format}"
        temp_path = os.path.join(temp_dir, filename)
        
        # Setup video codec
        if self.config.output_format == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:  # avi
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Create video writer
        height, width = frame_shape[:2]
        self.video_writer = cv2.VideoWriter(
            temp_path, fourcc, self.config.fps, (width, height)
        )
        
        return temp_path
    
    def start_recording(self) -> Dict[str, Any]:
        """Start video recording"""
        if self.is_recording:
            return {"status": "error", "message": "Recording already in progress"}
        
        self.is_recording = True
        self.frames = []
        self.stats['start_time'] = datetime.now()
        self.stats['total_frames'] = 0
        self.stats['key_moments'] = 0
        
        logger.info("Video recording started", 
                   duration=self.config.duration,
                   fps=self.config.fps,
                   mode=self.config.capture_mode)
        
        try:
            self._record_frames()
            return {
                "status": "completed",
                "message": f"Recording completed: {len(self.frames)} frames captured",
                "stats": self.stats
            }
        except Exception as e:
            self.is_recording = False
            logger.error("Recording failed", error=str(e))
            return {"status": "error", "message": f"Recording failed: {str(e)}"}
    
    def _record_frames(self):
        """Main recording loop"""
        capture_area = self._get_capture_area()
        frame_interval = 1.0 / self.config.fps
        end_time = time.time() + self.config.duration
        frame_number = 0
        
        with mss.mss() as sct:
            while time.time() < end_time and self.is_recording:
                loop_start = time.time()
                
                try:
                    # Capture frame
                    sct_img = sct.grab(capture_area)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Setup video writer on first frame
                    if self.video_writer is None and self.config.save_video:
                        self.temp_video_path = self._setup_video_writer(frame.shape)
                    
                    # Detect changes
                    change_percentage = self._detect_change(frame)
                    is_key_moment = change_percentage > self.config.key_moment_threshold
                    
                    # Create frame object
                    video_frame = VideoFrame(
                        frame=frame.copy(),
                        timestamp=datetime.now(),
                        frame_number=frame_number,
                        change_percentage=change_percentage,
                        is_key_moment=is_key_moment
                    )
                    
                    self.frames.append(video_frame)
                    
                    # Write to video file if enabled
                    if self.video_writer is not None:
                        self.video_writer.write(frame)
                    
                    # Update statistics
                    self.stats['total_frames'] += 1
                    if is_key_moment:
                        self.stats['key_moments'] += 1
                    
                    frame_number += 1
                    
                    # Control frame rate
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, frame_interval - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error("Frame capture error", error=str(e))
                    continue
        
        # Cleanup
        self.is_recording = False
        self.stats['end_time'] = datetime.now()
        self.stats['recording_duration'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        logger.info("Recording completed", 
                   frames=len(self.frames),
                   duration=self.stats['recording_duration'],
                   key_moments=self.stats['key_moments'])
    
    def get_frames_for_analysis(self) -> List[VideoFrame]:
        """Get frames based on analysis type"""
        if self.config.analysis_type == "frame_by_frame":
            return self.frames
        elif self.config.analysis_type == "key_moments":
            return [f for f in self.frames if f.is_key_moment]
        else:  # summary - return representative frames
            if len(self.frames) <= 5:
                return self.frames
            
            # Select frames evenly distributed across the recording
            step = len(self.frames) // 5
            return [self.frames[i] for i in range(0, len(self.frames), step)][:5]
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string"""
        _, buffer = cv2.imencode('.png', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    
    def get_video_path(self) -> Optional[str]:
        """Get saved video file path"""
        return self.temp_video_path if self.config.save_video else None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Optionally remove temp video file
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            if not self.config.save_video:
                try:
                    os.remove(self.temp_video_path)
                    logger.info("Temporary video file removed", path=self.temp_video_path)
                except Exception as e:
                    logger.warning("Failed to remove temp video file", error=str(e))

class VideoAnalyzer:
    """Video analysis system using AI"""

    def __init__(self, ai_provider=None, default_model: str = "gpt-4o"):
        self.ai_provider = ai_provider
        self.default_model = default_model
    
    async def analyze_video(self, recorder: VideoRecorder) -> VideoAnalysisResult:
        """Analyze recorded video using AI"""
        if not self.ai_provider:
            raise ValueError("AI provider not configured")

        start_time = time.time()
        frames_to_analyze = recorder.get_frames_for_analysis()

        logger.info("Starting video analysis",
                   analysis_type=recorder.config.analysis_type,
                   frames_count=len(frames_to_analyze),
                   total_frames=len(recorder.frames))

        if recorder.config.analysis_type == "summary":
            analysis_text = await self._analyze_summary(frames_to_analyze, recorder.config)
            frame_analyses = []
        elif recorder.config.analysis_type == "frame_by_frame":
            analysis_text, frame_analyses = await self._analyze_frame_by_frame(
                frames_to_analyze, recorder.config
            )
        else:  # key_moments
            analysis_text, frame_analyses = await self._analyze_key_moments(
                frames_to_analyze, recorder.config
            )

        processing_time = time.time() - start_time

        # Extract key moments information
        key_moments = [
            {
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp.isoformat(),
                "change_percentage": frame.change_percentage
            }
            for frame in recorder.frames if frame.is_key_moment
        ]

        result = VideoAnalysisResult(
            analysis_text=analysis_text,
            analysis_type=recorder.config.analysis_type,
            total_frames=len(recorder.frames),
            duration=recorder.stats['recording_duration'],
            key_moments=key_moments,
            processing_time=processing_time,
            video_path=recorder.get_video_path(),
            frame_analyses=frame_analyses
        )

        logger.info("Video analysis completed",
                   processing_time=processing_time,
                   analysis_length=len(analysis_text))

        return result

    async def _analyze_summary(self, frames: List[VideoFrame], config: VideoRecordingConfig) -> str:
        """Analyze video with summary approach"""
        # Convert frames to base64 for AI analysis
        frames_base64 = []
        for frame in frames:
            frame_base64 = self._frame_to_base64(frame.frame)
            frames_base64.append(frame_base64)

        # Create prompt with multiple frames
        prompt = f"{config.analysis_prompt}\n\n"
        prompt += f"Bu video {len(frames)} anahtar kare içeriyor. "
        prompt += f"Toplam süre: {config.duration} saniye. "
        prompt += "Lütfen bu video kaydında neler olduğunu detaylı olarak analiz edin."

        # Analyze with AI
        try:
            analysis = await self.ai_provider.analyze_image(
                image_base64=frames_base64[0],  # Use first frame as primary
                prompt=prompt,
                model="gpt-4o",  # Use vision model
                output_format="png",
                max_tokens=config.max_tokens or 1000,
                additional_images=frames_base64[1:] if len(frames_base64) > 1 else None
            )
            return analysis
        except Exception as e:
            logger.error("Summary analysis failed", error=str(e))
            return f"Video analizi başarısız oldu: {str(e)}"

    async def _analyze_frame_by_frame(self, frames: List[VideoFrame],
                                     config: VideoRecordingConfig) -> Tuple[str, List[Dict[str, Any]]]:
        """Analyze video frame by frame"""
        frame_analyses = []
        all_analyses = []

        for i, frame in enumerate(frames):
            frame_base64 = self._frame_to_base64(frame.frame)

            # Create frame-specific prompt
            frame_prompt = f"{config.analysis_prompt}\n\n"
            frame_prompt += f"Bu {i+1}/{len(frames)} numaralı kare. "
            frame_prompt += f"Zaman: {frame.timestamp.strftime('%H:%M:%S')}. "
            frame_prompt += f"Değişim oranı: %{frame.change_percentage*100:.1f}. "
            frame_prompt += "Bu karede ne görüyorsunuz?"

            try:
                analysis = await self.ai_provider.analyze_image(
                    image_base64=frame_base64,
                    prompt=frame_prompt,
                    model="gpt-4o",
                    output_format="png",
                    max_tokens=min(config.max_tokens or 500, 500)  # Limit tokens per frame
                )

                frame_analyses.append({
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp.isoformat(),
                    "analysis": analysis,
                    "change_percentage": frame.change_percentage
                })

                all_analyses.append(f"Kare {i+1}/{len(frames)}: {analysis}")

            except Exception as e:
                logger.error(f"Frame {i+1} analysis failed", error=str(e))
                frame_analyses.append({
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp.isoformat(),
                    "analysis": f"Analiz başarısız: {str(e)}",
                    "change_percentage": frame.change_percentage
                })

        # Create summary from all frame analyses
        summary = "## Video Kare-Kare Analizi\n\n"
        summary += "\n\n".join(all_analyses)
        summary += "\n\n## Özet\n\n"
        summary += "Video kaydında görülen ana olaylar ve değişiklikler yukarıda detaylandırılmıştır."

        return summary, frame_analyses

    async def _analyze_key_moments(self, frames: List[VideoFrame],
                                  config: VideoRecordingConfig) -> Tuple[str, List[Dict[str, Any]]]:
        """Analyze only key moments in the video"""
        if not frames:
            return "Video kaydında önemli bir değişiklik tespit edilmedi.", []

        frame_analyses = []
        key_moment_analyses = []

        for i, frame in enumerate(frames):
            frame_base64 = self._frame_to_base64(frame.frame)

            # Create key moment specific prompt
            moment_prompt = f"{config.analysis_prompt}\n\n"
            moment_prompt += f"Bu önemli bir değişiklik anı ({i+1}/{len(frames)}). "
            moment_prompt += f"Zaman: {frame.timestamp.strftime('%H:%M:%S')}. "
            moment_prompt += f"Değişim oranı: %{frame.change_percentage*100:.1f}. "
            moment_prompt += "Bu önemli anda ne değişti?"

            try:
                analysis = await self.ai_provider.analyze_image(
                    image_base64=frame_base64,
                    prompt=moment_prompt,
                    model=self.default_model,
                    output_format="png",
                    max_tokens=min(config.max_tokens or 500, 500)
                )

                frame_analyses.append({
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp.isoformat(),
                    "analysis": analysis,
                    "change_percentage": frame.change_percentage
                })

                key_moment_analyses.append(
                    f"Önemli An {i+1} ({frame.timestamp.strftime('%H:%M:%S')}): {analysis}"
                )

            except Exception as e:
                logger.error(f"Key moment {i+1} analysis failed", error=str(e))
                frame_analyses.append({
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp.isoformat(),
                    "analysis": f"Analiz başarısız: {str(e)}",
                    "change_percentage": frame.change_percentage
                })

        # Create summary from all key moment analyses
        summary = "## Video Önemli Anlar Analizi\n\n"
        summary += "\n\n".join(key_moment_analyses)
        summary += "\n\n## Genel Değerlendirme\n\n"
        summary += f"Bu {config.duration} saniyelik video kaydında {len(frames)} önemli değişiklik tespit edildi."

        return summary, frame_analyses

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string"""
        _, buffer = cv2.imencode('.png', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
