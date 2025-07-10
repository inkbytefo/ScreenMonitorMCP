"""
Smart Monitoring System
Combines real-time monitoring with intelligent analysis and trigger system
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

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    gw = None
    PYGETWINDOW_AVAILABLE = False

logger = structlog.get_logger()

@dataclass
class SmartEvent:
    """Smart monitoring event"""
    timestamp: datetime
    event_type: str  # "significant_change", "error_detected", "new_window", etc.
    description: str
    change_percentage: float = 0.0
    screenshot_base64: Optional[str] = None
    analysis_result: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SmartMonitoringConfig:
    """Smart monitoring configuration"""
    triggers: List[str] = field(default_factory=lambda: ["significant_change"])
    analysis_prompt: str = "Ekranda ne değişti ve neden önemli?"
    fps: int = 2
    sensitivity: Literal["low", "medium", "high"] = "medium"
    adaptive_fps: bool = True
    max_event_history: int = 100
    
    # Trigger thresholds
    significant_change_threshold: float = 0.2  # 20% change
    error_detection_enabled: bool = True
    new_window_detection: bool = True
    text_monitoring_keywords: List[str] = field(default_factory=list)

class TriggerDetector:
    """Detects various trigger conditions"""
    
    def __init__(self, config: SmartMonitoringConfig):
        self.config = config
        self.previous_frame = None
        self.previous_window_title = None
        self.last_significant_change = datetime.now()
    
    def detect_triggers(self, current_frame: np.ndarray, window_title: str = "") -> List[SmartEvent]:
        """Detect all configured triggers"""
        events = []
        now = datetime.now()
        
        # Detect significant changes
        if "significant_change" in self.config.triggers:
            change_event = self._detect_significant_change(current_frame, now)
            if change_event:
                events.append(change_event)
        
        # Detect new window
        if "new_window" in self.config.triggers:
            window_event = self._detect_new_window(window_title, now)
            if window_event:
                events.append(window_event)
        
        # Detect errors (red text, error dialogs)
        if "error_detected" in self.config.triggers:
            error_event = self._detect_errors(current_frame, now)
            if error_event:
                events.append(error_event)
        
        # Detect application changes
        if "application_change" in self.config.triggers:
            app_event = self._detect_application_change(window_title, now)
            if app_event:
                events.append(app_event)
        
        # Detect code changes (if in code editor)
        if "code_change" in self.config.triggers:
            code_event = self._detect_code_change(current_frame, window_title, now)
            if code_event:
                events.append(code_event)
        
        return events
    
    def _detect_significant_change(self, current_frame: np.ndarray, timestamp: datetime) -> Optional[SmartEvent]:
        """Detect significant visual changes"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return None
        
        # Calculate difference
        diff = cv2.absdiff(self.previous_frame, current_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate change percentage
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels
        
        self.previous_frame = current_frame.copy()
        
        if change_percentage >= self.config.significant_change_threshold:
            # Convert frame to base64
            _, buffer = cv2.imencode('.png', current_frame)
            screenshot_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return SmartEvent(
                timestamp=timestamp,
                event_type="significant_change",
                description=f"Significant visual change detected ({change_percentage:.1%})",
                change_percentage=change_percentage,
                screenshot_base64=screenshot_base64,
                confidence=min(1.0, change_percentage * 2),
                metadata={"change_pixels": changed_pixels, "total_pixels": total_pixels}
            )
        
        return None
    
    def _detect_new_window(self, window_title: str, timestamp: datetime) -> Optional[SmartEvent]:
        """Detect new window opening"""
        if self.previous_window_title is None:
            self.previous_window_title = window_title
            return None
        
        if window_title != self.previous_window_title and window_title:
            self.previous_window_title = window_title
            return SmartEvent(
                timestamp=timestamp,
                event_type="new_window",
                description=f"New window opened: {window_title}",
                confidence=1.0,
                metadata={"window_title": window_title}
            )
        
        return None
    
    def _detect_errors(self, current_frame: np.ndarray, timestamp: datetime) -> Optional[SmartEvent]:
        """Detect error messages (red text, error dialogs)"""
        if not self.config.error_detection_enabled:
            return None
        
        # Convert to HSV for better red detection
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        
        # Define red color range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red colors
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Count red pixels
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = current_frame.shape[0] * current_frame.shape[1]
        red_percentage = red_pixels / total_pixels
        
        # If significant red content (potential error)
        if red_percentage > 0.01:  # 1% red pixels
            return SmartEvent(
                timestamp=timestamp,
                event_type="error_detected",
                description=f"Potential error detected (red content: {red_percentage:.1%})",
                confidence=min(1.0, red_percentage * 50),
                metadata={"red_pixels": red_pixels, "red_percentage": red_percentage}
            )
        
        return None
    
    def _detect_application_change(self, window_title: str, timestamp: datetime) -> Optional[SmartEvent]:
        """Detect application switching"""
        if self.previous_window_title is None:
            self.previous_window_title = window_title
            return None
        
        # Extract application name from window title
        current_app = self._extract_app_name(window_title)
        previous_app = self._extract_app_name(self.previous_window_title)
        
        if current_app != previous_app and current_app:
            return SmartEvent(
                timestamp=timestamp,
                event_type="application_change",
                description=f"Switched to application: {current_app}",
                confidence=1.0,
                metadata={"current_app": current_app, "previous_app": previous_app}
            )
        
        return None
    
    def _detect_code_change(self, current_frame: np.ndarray, window_title: str, timestamp: datetime) -> Optional[SmartEvent]:
        """Detect code editor changes"""
        # Check if we're in a code editor
        code_editors = ["Visual Studio Code", "PyCharm", "Sublime Text", "Atom", "Notepad++"]
        is_code_editor = any(editor.lower() in window_title.lower() for editor in code_editors)
        
        if not is_code_editor:
            return None
        
        # For now, just detect if we're in a code editor and there's a change
        # This could be enhanced with more sophisticated code change detection
        if self.previous_frame is not None:
            diff = cv2.absdiff(self.previous_frame, current_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            total_pixels = thresh.shape[0] * thresh.shape[1]
            changed_pixels = cv2.countNonZero(thresh)
            change_percentage = changed_pixels / total_pixels
            
            if change_percentage > 0.05:  # 5% change in code editor
                return SmartEvent(
                    timestamp=timestamp,
                    event_type="code_change",
                    description=f"Code change detected in {window_title}",
                    change_percentage=change_percentage,
                    confidence=0.8,
                    metadata={"editor": window_title, "change_percentage": change_percentage}
                )
        
        return None
    
    def _extract_app_name(self, window_title: str) -> str:
        """Extract application name from window title"""
        if not window_title:
            return ""
        
        # Common patterns to extract app name
        if " - " in window_title:
            return window_title.split(" - ")[-1]
        elif " | " in window_title:
            return window_title.split(" | ")[-1]
        else:
            return window_title.split()[0] if window_title.split() else window_title

class AdaptiveFPSController:
    """Controls FPS based on activity level"""
    
    def __init__(self, base_fps: int = 2):
        self.base_fps = base_fps
        self.current_fps = base_fps
        self.activity_history = deque(maxlen=10)
        self.last_adjustment = datetime.now()
    
    def update_activity(self, has_activity: bool):
        """Update activity level"""
        self.activity_history.append(1 if has_activity else 0)
        
        # Adjust FPS every 30 seconds
        if datetime.now() - self.last_adjustment > timedelta(seconds=30):
            self._adjust_fps()
            self.last_adjustment = datetime.now()
    
    def _adjust_fps(self):
        """Adjust FPS based on recent activity"""
        if not self.activity_history:
            return
        
        activity_rate = sum(self.activity_history) / len(self.activity_history)
        
        if activity_rate > 0.7:  # High activity
            self.current_fps = min(5, self.base_fps + 2)
        elif activity_rate < 0.2:  # Low activity
            self.current_fps = max(1, self.base_fps - 1)
        else:  # Normal activity
            self.current_fps = self.base_fps
        
        logger.debug("FPS adjusted", activity_rate=activity_rate, new_fps=self.current_fps)
    
    def get_sleep_time(self) -> float:
        """Get sleep time for current FPS"""
        return 1.0 / self.current_fps

class SmartMonitor:
    """Main smart monitoring system"""

    def __init__(self, config: SmartMonitoringConfig, ai_provider=None):
        self.config = config
        self.ai_provider = ai_provider
        self.is_monitoring = False
        self.monitor_thread = None

        # Components
        self.trigger_detector = TriggerDetector(config)
        self.fps_controller = AdaptiveFPSController(config.fps)

        # State
        self.event_history: List[SmartEvent] = []
        self.analysis_results: List[Dict[str, Any]] = []
        self.last_analysis_time = datetime.now()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'events_detected': 0,
            'analyses_performed': 0,
            'start_time': None,
            'last_activity': None
        }

        # Callbacks
        self.event_callbacks: List[Callable] = []

    def add_event_callback(self, callback: Callable[[SmartEvent], None]):
        """Add callback for events"""
        self.event_callbacks.append(callback)

    def start_monitoring(self) -> Dict[str, Any]:
        """Start smart monitoring"""
        if self.is_monitoring:
            return {"status": "already_running", "message": "Smart monitoring already running"}

        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Smart monitoring started",
                   triggers=self.config.triggers,
                   fps=self.config.fps)

        return {
            "status": "started",
            "message": f"Smart monitoring started with triggers: {', '.join(self.config.triggers)}",
            "config": {
                "triggers": self.config.triggers,
                "fps": self.config.fps,
                "sensitivity": self.config.sensitivity,
                "adaptive_fps": self.config.adaptive_fps
            }
        }

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop smart monitoring"""
        if not self.is_monitoring:
            return {"status": "not_running", "message": "Smart monitoring not running"}

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)

        duration = datetime.now() - self.stats['start_time']

        logger.info("Smart monitoring stopped",
                   duration=str(duration),
                   events_detected=self.stats['events_detected'])

        return {
            "status": "stopped",
            "message": "Smart monitoring stopped",
            "stats": {
                **self.stats,
                "duration": str(duration)
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        if not self.is_monitoring:
            return {
                "status": "stopped",
                "message": "Smart monitoring not running"
            }

        duration = datetime.now() - self.stats['start_time']

        return {
            "status": "running",
            "config": {
                "triggers": self.config.triggers,
                "fps": self.config.fps,
                "sensitivity": self.config.sensitivity
            },
            "duration": str(duration),
            "stats": self.stats,
            "recent_events": len(self.event_history),
            "current_fps": self.fps_controller.current_fps
        }

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events"""
        recent_events = self.event_history[-limit:] if self.event_history else []

        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "description": event.description,
                "change_percentage": event.change_percentage,
                "confidence": event.confidence,
                "has_screenshot": event.screenshot_base64 is not None,
                "has_analysis": event.analysis_result is not None,
                "metadata": event.metadata
            }
            for event in recent_events
        ]

    def get_monitoring_insights(self) -> Dict[str, Any]:
        """Get monitoring insights and analysis results"""
        if not self.event_history:
            return {
                "message": "No events detected yet",
                "total_events": 0,
                "analyses": []
            }

        # Categorize events
        event_types = {}
        for event in self.event_history:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        # Get recent analyses
        recent_analyses = self.analysis_results[-5:] if self.analysis_results else []

        return {
            "total_events": len(self.event_history),
            "event_breakdown": event_types,
            "recent_analyses": recent_analyses,
            "monitoring_duration": str(datetime.now() - self.stats['start_time']) if self.stats['start_time'] else "0:00:00",
            "current_triggers": self.config.triggers,
            "performance": {
                "avg_fps": self.fps_controller.current_fps,
                "total_frames": self.stats['total_frames'],
                "events_per_minute": len(self.event_history) / max(1, (datetime.now() - self.stats['start_time']).total_seconds() / 60) if self.stats['start_time'] else 0
            }
        }

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary report"""
        if not self.stats['start_time']:
            return {"message": "Monitoring not started yet"}

        duration = datetime.now() - self.stats['start_time']

        # Find most common event type
        event_types = {}
        for event in self.event_history:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        most_common_event = max(event_types.items(), key=lambda x: x[1]) if event_types else ("none", 0)

        return {
            "monitoring_session": {
                "duration": str(duration),
                "status": "running" if self.is_monitoring else "stopped",
                "frames_processed": self.stats['total_frames'],
                "events_detected": self.stats['events_detected']
            },
            "activity_summary": {
                "most_common_event": most_common_event[0],
                "most_common_count": most_common_event[1],
                "total_event_types": len(event_types),
                "analyses_performed": self.stats['analyses_performed']
            },
            "configuration": {
                "active_triggers": self.config.triggers,
                "sensitivity": self.config.sensitivity,
                "adaptive_fps": self.config.adaptive_fps,
                "current_fps": self.fps_controller.current_fps
            }
        }

    async def _analyze_event(self, event: SmartEvent) -> Optional[str]:
        """Analyze event with AI"""
        if not self.ai_provider or not event.screenshot_base64:
            return None

        try:
            analysis = await self.ai_provider.analyze_image(
                image_base64=event.screenshot_base64,
                prompt=f"{self.config.analysis_prompt}\n\nEvent: {event.description}",
                max_tokens=300
            )

            # Store analysis result
            analysis_result = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "analysis": analysis,
                "confidence": event.confidence
            }
            self.analysis_results.append(analysis_result)

            # Keep only recent analyses
            if len(self.analysis_results) > 20:
                self.analysis_results.pop(0)

            self.stats['analyses_performed'] += 1

            logger.info("Event analyzed",
                       event_type=event.event_type,
                       analysis_length=len(analysis))

            return analysis

        except Exception as e:
            logger.error("Event analysis failed", error=str(e))
            return None

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Smart monitoring loop started")

        if not PYGETWINDOW_AVAILABLE:
            logger.warning("pygetwindow not available, window detection disabled")

        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Primary monitor

            while self.is_monitoring:
                try:
                    loop_start = time.time()

                    # Capture frame
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Get current window title
                    window_title = ""
                    if PYGETWINDOW_AVAILABLE and gw:
                        try:
                            active_window = gw.getActiveWindow()
                            window_title = active_window.title if active_window else ""
                        except:
                            pass

                    # Detect triggers
                    events = self.trigger_detector.detect_triggers(frame, window_title)

                    # Process events
                    has_activity = len(events) > 0
                    for event in events:
                        # Add to history
                        self.event_history.append(event)
                        if len(self.event_history) > self.config.max_event_history:
                            self.event_history.pop(0)

                        # Update stats
                        self.stats['events_detected'] += 1
                        self.stats['last_activity'] = datetime.now()

                        # Analyze event if AI provider available
                        if self.ai_provider and event.screenshot_base64:
                            # Schedule analysis (run in background)
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    asyncio.create_task(self._analyze_and_update_event(event))
                                else:
                                    # If no event loop, skip analysis for now
                                    logger.debug("No event loop available for analysis")
                            except RuntimeError:
                                # No event loop in current thread, skip analysis
                                logger.debug("No event loop in monitoring thread, skipping analysis")

                        # Call event callbacks
                        for callback in self.event_callbacks:
                            try:
                                callback(event)
                            except Exception as e:
                                logger.error("Event callback error", error=str(e))

                        logger.debug("Event detected",
                                   event_type=event.event_type,
                                   description=event.description)

                    # Update adaptive FPS
                    if self.config.adaptive_fps:
                        self.fps_controller.update_activity(has_activity)

                    # Update stats
                    self.stats['total_frames'] += 1

                    # FPS control
                    elapsed = time.time() - loop_start
                    if self.config.adaptive_fps:
                        sleep_time = max(0, self.fps_controller.get_sleep_time() - elapsed)
                    else:
                        sleep_time = max(0, (1.0 / self.config.fps) - elapsed)

                    time.sleep(sleep_time)

                except Exception as e:
                    logger.error("Monitoring loop error", error=str(e))
                    time.sleep(1.0)

    async def _analyze_and_update_event(self, event: SmartEvent):
        """Analyze event and update with result"""
        try:
            analysis = await self._analyze_event(event)
            if analysis:
                event.analysis_result = analysis
        except Exception as e:
            logger.error("Failed to analyze event", error=str(e))

# Global smart monitor instance
_global_smart_monitor: Optional[SmartMonitor] = None

def get_global_smart_monitor() -> Optional[SmartMonitor]:
    """Get global smart monitor instance"""
    return _global_smart_monitor

def set_global_smart_monitor(monitor: SmartMonitor):
    """Set global smart monitor instance"""
    global _global_smart_monitor
    _global_smart_monitor = monitor
