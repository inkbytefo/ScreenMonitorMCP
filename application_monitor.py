"""
General Application Monitoring System
Detects and monitors changes in any application and relays them to AI clients
"""

import asyncio
import threading
import time
import pygetwindow as gw
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import structlog
import json
from enum import Enum

logger = structlog.get_logger()

class EventType(Enum):
    """Application event types"""
    WINDOW_FOCUS = "window_focus"
    WINDOW_CLOSE = "window_close"
    WINDOW_OPEN = "window_open"
    WINDOW_RESIZE = "window_resize"
    WINDOW_MOVE = "window_move"
    APPLICATION_START = "application_start"
    APPLICATION_EXIT = "application_exit"
    CONTENT_CHANGE = "content_change"
    CUSTOM_EVENT = "custom_event"

@dataclass
class ApplicationEvent:
    """Application event data structure"""
    timestamp: datetime
    event_type: EventType
    application_name: str
    window_title: str
    event_data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"  # system, application, user

@dataclass
class ApplicationInfo:
    """Application information"""
    name: str
    pid: int
    window_titles: List[str] = field(default_factory=list)
    is_active: bool = False
    last_activity: Optional[datetime] = None

class ApplicationDetector:
    """Detects running applications and their windows"""
    
    def __init__(self):
        self.known_applications: Dict[str, ApplicationInfo] = {}
        self.previous_windows: Set[str] = set()
        self.active_window: Optional[str] = None
    
    def get_running_applications(self) -> Dict[str, ApplicationInfo]:
        """Get all running applications with windows"""
        current_apps = {}
        
        try:
            # Get all windows
            windows = gw.getAllWindows()
            current_windows = set()
            
            for window in windows:
                if window.title and window.title.strip():
                    # Get process info
                    try:
                        # Extract application name from window title or process
                        app_name = self._extract_app_name(window.title)
                        current_windows.add(window.title)
                        
                        if app_name not in current_apps:
                            current_apps[app_name] = ApplicationInfo(
                                name=app_name,
                                pid=0,  # We'll get this later if needed
                                window_titles=[],
                                is_active=False
                            )
                        
                        current_apps[app_name].window_titles.append(window.title)
                        
                        # Check if this is the active window
                        try:
                            active_window = gw.getActiveWindow()
                            if active_window and active_window.title == window.title:
                                current_apps[app_name].is_active = True
                                current_apps[app_name].last_activity = datetime.now()
                                self.active_window = window.title
                        except:
                            pass
                            
                    except Exception as e:
                        logger.debug("Error processing window", window_title=window.title, error=str(e))
            
            # Detect new and closed windows
            new_windows = current_windows - self.previous_windows
            closed_windows = self.previous_windows - current_windows
            
            self.previous_windows = current_windows
            
            # Store detection results for event generation
            self._new_windows = new_windows
            self._closed_windows = closed_windows
            
        except Exception as e:
            logger.error("Error detecting applications", error=str(e))
        
        self.known_applications = current_apps
        return current_apps
    
    def _extract_app_name(self, window_title: str) -> str:
        """Extract application name from window title"""
        # Common application patterns
        app_patterns = {
            "blender": ["Blender"],
            "vscode": ["Visual Studio Code", "VSCode"],
            "chrome": ["Google Chrome", "Chrome"],
            "firefox": ["Mozilla Firefox", "Firefox"],
            "notepad": ["Notepad"],
            "explorer": ["File Explorer", "Windows Explorer"],
            "cmd": ["Command Prompt", "cmd.exe"],
            "powershell": ["PowerShell"],
            "discord": ["Discord"],
            "slack": ["Slack"],
            "teams": ["Microsoft Teams"],
            "zoom": ["Zoom"],
        }
        
        window_lower = window_title.lower()
        
        for app_name, patterns in app_patterns.items():
            for pattern in patterns:
                if pattern.lower() in window_lower:
                    return app_name
        
        # If no pattern matches, try to extract from common formats
        # "AppName - Document" or "Document - AppName"
        if " - " in window_title:
            parts = window_title.split(" - ")
            # Usually the app name is the last part
            return parts[-1].strip()
        
        # Default: use first word
        return window_title.split()[0] if window_title.split() else "unknown"

class ApplicationMonitor:
    """Main application monitoring system"""
    
    def __init__(self):
        self.detector = ApplicationDetector()
        self.is_monitoring = False
        self.monitor_thread = None
        self.event_callbacks: List[Callable[[ApplicationEvent], None]] = []
        self.event_history: List[ApplicationEvent] = []
        self.registered_applications: Set[str] = set()
        self.custom_event_handlers: Dict[str, Callable] = {}
        self.max_history = 100
        
        # Monitoring configuration
        self.poll_interval = 1.0  # seconds
        
    def add_event_callback(self, callback: Callable[[ApplicationEvent], None]):
        """Add callback for application events"""
        self.event_callbacks.append(callback)
    
    def register_application(self, app_name: str, event_types: List[str] = None):
        """Register an application for monitoring"""
        self.registered_applications.add(app_name.lower())
        logger.info("Application registered for monitoring", app_name=app_name, event_types=event_types)
    
    def add_custom_event_handler(self, app_name: str, handler: Callable):
        """Add custom event handler for specific application"""
        self.custom_event_handlers[app_name.lower()] = handler
    
    def start_monitoring(self) -> Dict[str, Any]:
        """Start application monitoring"""
        if self.is_monitoring:
            return {"status": "already_running", "message": "Application monitoring is already running"}
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Application monitoring started")
        
        return {
            "status": "started",
            "message": "Application monitoring started",
            "registered_applications": list(self.registered_applications),
            "poll_interval": self.poll_interval
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop application monitoring"""
        if not self.is_monitoring:
            return {"status": "not_running", "message": "Application monitoring is not running"}
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Application monitoring stopped")
        
        return {
            "status": "stopped",
            "message": "Application monitoring stopped",
            "events_captured": len(self.event_history)
        }
    
    def get_active_application(self) -> Dict[str, Any]:
        """Get currently active application"""
        apps = self.detector.get_running_applications()
        
        for app_name, app_info in apps.items():
            if app_info.is_active:
                return {
                    "application_name": app_name,
                    "window_title": self.detector.active_window,
                    "window_titles": app_info.window_titles,
                    "last_activity": app_info.last_activity.isoformat() if app_info.last_activity else None
                }
        
        return {"application_name": None, "message": "No active application detected"}
    
    def broadcast_application_change(self, app_name: str, event_type: str, event_data: Dict[str, Any]):
        """Broadcast custom application change event"""
        event = ApplicationEvent(
            timestamp=datetime.now(),
            event_type=EventType.CUSTOM_EVENT,
            application_name=app_name,
            window_title=event_data.get("window_title", ""),
            event_data={
                "custom_event_type": event_type,
                **event_data
            },
            source="application"
        )
        
        self._process_event(event)
        logger.info("Custom application event broadcasted", app_name=app_name, event_type=event_type)
    
    def get_recent_events(self, limit: int = 10, app_name: str = None) -> List[Dict[str, Any]]:
        """Get recent application events"""
        events = self.event_history[-limit:] if not app_name else [
            e for e in self.event_history[-limit*2:] 
            if e.application_name.lower() == app_name.lower()
        ][-limit:]
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "application_name": event.application_name,
                "window_title": event.window_title,
                "event_data": event.event_data,
                "source": event.source
            }
            for event in events
        ]
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        previous_active = None
        
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Detect applications
                apps = self.detector.get_running_applications()
                
                # Check for window focus changes
                current_active = None
                for app_name, app_info in apps.items():
                    if app_info.is_active:
                        current_active = (app_name, self.detector.active_window)
                        break
                
                if current_active != previous_active:
                    if current_active:
                        # Window focus event
                        event = ApplicationEvent(
                            timestamp=datetime.now(),
                            event_type=EventType.WINDOW_FOCUS,
                            application_name=current_active[0],
                            window_title=current_active[1],
                            event_data={"previous_window": previous_active[1] if previous_active else None}
                        )
                        self._process_event(event)
                    
                    previous_active = current_active
                
                # Check for new/closed windows
                if hasattr(self.detector, '_new_windows'):
                    for window_title in self.detector._new_windows:
                        app_name = self.detector._extract_app_name(window_title)
                        event = ApplicationEvent(
                            timestamp=datetime.now(),
                            event_type=EventType.WINDOW_OPEN,
                            application_name=app_name,
                            window_title=window_title
                        )
                        self._process_event(event)
                
                if hasattr(self.detector, '_closed_windows'):
                    for window_title in self.detector._closed_windows:
                        app_name = self.detector._extract_app_name(window_title)
                        event = ApplicationEvent(
                            timestamp=datetime.now(),
                            event_type=EventType.WINDOW_CLOSE,
                            application_name=app_name,
                            window_title=window_title
                        )
                        self._process_event(event)
                
                # Sleep for poll interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error("Application monitor loop error", error=str(e))
                time.sleep(1.0)
    
    def _process_event(self, event: ApplicationEvent):
        """Process and broadcast application event"""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Call callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Event callback error", error=str(e))
        
        # Call custom handlers
        app_name_lower = event.application_name.lower()
        if app_name_lower in self.custom_event_handlers:
            try:
                self.custom_event_handlers[app_name_lower](event)
            except Exception as e:
                logger.error("Custom event handler error", app_name=event.application_name, error=str(e))

# Global monitor instance
_global_app_monitor: Optional[ApplicationMonitor] = None

def get_global_app_monitor() -> Optional[ApplicationMonitor]:
    """Get global application monitor instance"""
    return _global_app_monitor

def set_global_app_monitor(monitor: ApplicationMonitor):
    """Set global application monitor instance"""
    global _global_app_monitor
    _global_app_monitor = monitor
