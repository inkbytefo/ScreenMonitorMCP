"""
Real-Time Screen Monitoring System
Revolutionary feature: Giving AI continuous vision capability
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import mss
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import structlog

logger = structlog.get_logger()

@dataclass
class ChangeEvent:
    """Ekran değişikliği olayını temsil eder"""
    timestamp: datetime
    change_type: str  # 'minor', 'major', 'critical'
    change_percentage: float
    affected_regions: List[Dict]
    screenshot_base64: Optional[str] = None
    description: str = ""

@dataclass
class MonitoringConfig:
    """Real-time monitoring configuration"""
    fps: int = 2
    change_threshold: float = 0.1  # 10% change threshold
    major_change_threshold: float = 0.3  # 30% major change
    critical_change_threshold: float = 0.6  # 60% critical change
    focus_regions: List[Dict] = field(default_factory=list)
    smart_detection: bool = True
    save_screenshots: bool = True
    max_history: int = 100

class SmartChangeDetector:
    """Smart change detection system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.previous_frame = None
        self.frame_history = []
        self.change_patterns = []
        
    def detect_changes(self, current_frame: np.ndarray) -> ChangeEvent:
        """Detects changes between current frame and previous frame"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return ChangeEvent(
                timestamp=datetime.now(),
                change_type='initial',
                change_percentage=0.0,
                affected_regions=[]
            )
        
        # Frame differencing
        diff = cv2.absdiff(self.previous_frame, current_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold uygula
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate change percentage
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels

        # Find change regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        affected_regions = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small changes
                x, y, w, h = cv2.boundingRect(contour)
                affected_regions.append({
                    'x': int(x), 'y': int(y), 
                    'width': int(w), 'height': int(h),
                    'area': int(cv2.contourArea(contour))
                })
        
        # Determine change type
        change_type = self._classify_change(change_percentage, affected_regions)

        # Convert screenshot to base64 (if needed)
        screenshot_base64 = None
        if self.config.save_screenshots and change_type != 'minor':
            screenshot_base64 = self._frame_to_base64(current_frame)
        
        # Önceki frame'i güncelle
        self.previous_frame = current_frame.copy()
        
        return ChangeEvent(
            timestamp=datetime.now(),
            change_type=change_type,
            change_percentage=change_percentage,
            affected_regions=affected_regions,
            screenshot_base64=screenshot_base64,
            description=self._generate_description(change_type, affected_regions)
        )
    
    def _classify_change(self, change_percentage: float, regions: List[Dict]) -> str:
        """Classifies change type"""
        if change_percentage >= self.config.critical_change_threshold:
            return 'critical'
        elif change_percentage >= self.config.major_change_threshold:
            return 'major'
        elif change_percentage >= self.config.change_threshold:
            return 'minor'
        else:
            return 'none'
    
    def _generate_description(self, change_type: str, regions: List[Dict]) -> str:
        """Generates description for change"""
        if change_type == 'none':
            return "No significant change detected"

        region_count = len(regions)
        if region_count == 0:
            return f"{change_type.title()} change detected"
        elif region_count == 1:
            return f"{change_type.title()} change detected (1 region)"
        else:
            return f"{change_type.title()} change detected ({region_count} regions)"
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Converts frame to base64 string"""
        _, buffer = cv2.imencode('.png', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

class RealTimeMonitor:
    """Real-time screen monitoring system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.detector = SmartChangeDetector(config)
        self.is_monitoring = False
        self.monitor_thread = None
        self.change_callbacks: List[Callable] = []
        self.ai_callbacks: List[Callable] = []  # New: AI analysis callbacks
        self.event_history: List[ChangeEvent] = []
        self.stats = {
            'total_frames': 0,
            'changes_detected': 0,
            'ai_analyses_triggered': 0,  # New: AI analysis counter
            'start_time': None,
            'last_activity': None
        }
    
    def add_change_callback(self, callback: Callable[[ChangeEvent], None]):
        """Adds callback to be called when change is detected"""
        self.change_callbacks.append(callback)

    def add_ai_callback(self, callback: Callable[[ChangeEvent], None]):
        """Adds AI analysis callback to be called on significant changes"""
        self.ai_callbacks.append(callback)

    def remove_ai_callback(self, callback: Callable[[ChangeEvent], None]):
        """Removes AI analysis callback"""
        if callback in self.ai_callbacks:
            self.ai_callbacks.remove(callback)
    
    def start_monitoring(self) -> Dict[str, Any]:
        """Starts monitoring"""
        if self.is_monitoring:
            return {"status": "already_running", "message": "Monitoring is already running"}

        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Real-time monitoring started", fps=self.config.fps)

        return {
            "status": "started",
            "message": f"Real-time monitoring started ({self.config.fps} FPS)",
            "config": {
                "fps": self.config.fps,
                "change_threshold": self.config.change_threshold,
                "smart_detection": self.config.smart_detection
            }
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stops monitoring"""
        if not self.is_monitoring:
            return {"status": "not_running", "message": "Monitoring is already stopped"}

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        duration = datetime.now() - self.stats['start_time']

        logger.info("Real-time monitoring stopped",
                   duration=str(duration),
                   total_frames=self.stats['total_frames'],
                   changes_detected=self.stats['changes_detected'])

        return {
            "status": "stopped",
            "message": "Real-time monitoring stopped",
            "stats": {
                "duration": str(duration),
                "total_frames": self.stats['total_frames'],
                "changes_detected": self.stats['changes_detected'],
                "avg_fps": self.stats['total_frames'] / duration.total_seconds() if duration.total_seconds() > 0 else 0
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Returns monitoring status"""
        return {
            "is_monitoring": self.is_monitoring,
            "stats": self.stats.copy(),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "change_type": event.change_type,
                    "change_percentage": event.change_percentage,
                    "description": event.description
                }
                for event in self.event_history[-10:]  # Last 10 events
            ]
        }
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Capture all screens

            while self.is_monitoring:
                try:
                    start_time = time.time()

                    # Capture screenshot
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Detect changes
                    change_event = self.detector.detect_changes(frame)
                    
                    # Update statistics
                    self.stats['total_frames'] += 1
                    if change_event.change_type != 'none':
                        self.stats['changes_detected'] += 1
                        self.stats['last_activity'] = datetime.now()
                        
                        # Event history'ye ekle
                        self.event_history.append(change_event)
                        if len(self.event_history) > self.config.max_history:
                            self.event_history.pop(0)
                        
                        # Callback'leri çağır
                        for callback in self.change_callbacks:
                            try:
                                callback(change_event)
                            except Exception as e:
                                logger.error("Callback error", error=str(e))

                        # AI callback'leri çağır (sadece önemli değişikliklerde)
                        if change_event.change_type in ['major', 'critical']:
                            self.stats['ai_analyses_triggered'] += 1
                            for ai_callback in self.ai_callbacks:
                                try:
                                    ai_callback(change_event)
                                except Exception as e:
                                    logger.error("AI callback error", error=str(e))
                    
                    # FPS kontrolü
                    elapsed = time.time() - start_time
                    sleep_time = max(0, (1.0 / self.config.fps) - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error("Monitor loop error", error=str(e))
                    time.sleep(1.0)  # Hata durumunda kısa bekle

# Global monitor instance
_global_monitor: Optional[RealTimeMonitor] = None

def get_global_monitor() -> Optional[RealTimeMonitor]:
    """Global monitor instance'ını döndürür"""
    return _global_monitor

def set_global_monitor(monitor: RealTimeMonitor):
    """Global monitor instance'ını ayarlar"""
    global _global_monitor
    _global_monitor = monitor
