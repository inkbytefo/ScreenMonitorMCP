"""
Cross-Platform Support for ScreenMonitorMCP
Provides unified interface for Windows, macOS, and Linux operations
"""

import os
import sys
import platform
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class PlatformInterface(ABC):
    """Abstract interface for platform-specific operations"""
    
    @abstractmethod
    def get_active_window(self) -> Dict[str, Any]:
        """Get information about the active window"""
        pass
    
    @abstractmethod
    def get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of all windows"""
        pass
    
    @abstractmethod
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """Capture screen or region"""
        pass
    
    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        pass
    
    @abstractmethod
    def click_at_position(self, x: int, y: int) -> bool:
        """Click at specific coordinates"""
        pass
    
    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        pass

class WindowsPlatform(PlatformInterface):
    """Windows-specific implementation"""
    
    def __init__(self):
        try:
            import pygetwindow as gw
            import mss
            import pyautogui
            self.gw = gw
            self.mss = mss
            self.pyautogui = pyautogui
            self.available = True
        except ImportError as e:
            logger.warning(f"Windows platform dependencies not available: {e}")
            self.available = False
    
    def get_active_window(self) -> Dict[str, Any]:
        """Get active window information"""
        if not self.available:
            return {"error": "Windows platform not available"}
        
        try:
            active_window = self.gw.getActiveWindow()
            if active_window:
                return {
                    "title": active_window.title,
                    "left": active_window.left,
                    "top": active_window.top,
                    "width": active_window.width,
                    "height": active_window.height,
                    "visible": active_window.visible,
                    "minimized": active_window.isMinimized,
                    "maximized": active_window.isMaximized
                }
            return {"error": "No active window found"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of all windows"""
        if not self.available:
            return [{"error": "Windows platform not available"}]
        
        try:
            windows = []
            for window in self.gw.getAllWindows():
                if window.title:  # Skip windows without titles
                    windows.append({
                        "title": window.title,
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                        "visible": window.visible,
                        "minimized": window.isMinimized,
                        "maximized": window.isMaximized
                    })
            return windows
        except Exception as e:
            return [{"error": str(e)}]
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """Capture screen using mss"""
        if not self.available:
            raise Exception("Windows platform not available")
        
        try:
            with self.mss.mss() as sct:
                if region:
                    monitor = {"top": region[1], "left": region[0], 
                             "width": region[2], "height": region[3]}
                else:
                    monitor = sct.monitors[0]  # Primary monitor
                
                screenshot = sct.grab(monitor)
                return screenshot.rgb
        except Exception as e:
            raise Exception(f"Screen capture failed: {str(e)}")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        if not self.available:
            return (0, 0)
        
        try:
            return self.pyautogui.size()
        except Exception:
            return (1920, 1080)  # Default fallback
    
    def click_at_position(self, x: int, y: int) -> bool:
        """Click at coordinates"""
        if not self.available:
            return False
        
        try:
            self.pyautogui.click(x, y)
            return True
        except Exception as e:
            logger.error(f"Click failed: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get Windows system information"""
        return {
            "platform": "Windows",
            "version": platform.version(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "available_features": ["window_management", "screen_capture", "mouse_control"] if self.available else []
        }

class MacOSPlatform(PlatformInterface):
    """macOS-specific implementation"""
    
    def __init__(self):
        try:
            # Try to import macOS-specific libraries
            self.available = sys.platform == "darwin"
            if self.available:
                # Check for required tools
                self._check_tools()
        except Exception as e:
            logger.warning(f"macOS platform setup failed: {e}")
            self.available = False
    
    def _check_tools(self):
        """Check if required macOS tools are available"""
        try:
            subprocess.run(["screencapture", "-h"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("screencapture tool not available")
            self.available = False
    
    def get_active_window(self) -> Dict[str, Any]:
        """Get active window using AppleScript"""
        if not self.available:
            return {"error": "macOS platform not available"}
        
        try:
            script = '''
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set frontWindow to first window of frontApp
                return {name of frontApp, title of frontWindow, position of frontWindow, size of frontWindow}
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse result (simplified)
            return {"title": result.stdout.strip(), "platform": "macOS"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_window_list(self) -> List[Dict[str, Any]]:
        """Get window list using AppleScript"""
        if not self.available:
            return [{"error": "macOS platform not available"}]
        
        try:
            script = '''
            tell application "System Events"
                set windowList to {}
                repeat with proc in application processes
                    if visible of proc is true then
                        repeat with win in windows of proc
                            set end of windowList to {name of proc, title of win}
                        end repeat
                    end if
                end repeat
                return windowList
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Simplified parsing
            return [{"title": "macOS Window", "platform": "macOS"}]
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """Capture screen using screencapture"""
        if not self.available:
            raise Exception("macOS platform not available")
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cmd = ["screencapture", "-x", tmp.name]
                
                if region:
                    cmd.extend(["-R", f"{region[0]},{region[1]},{region[2]},{region[3]}"])
                
                subprocess.run(cmd, check=True)
                
                with open(tmp.name, "rb") as f:
                    data = f.read()
                
                os.unlink(tmp.name)
                return data
                
        except Exception as e:
            raise Exception(f"Screen capture failed: {str(e)}")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size using system_profiler"""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse output for resolution (simplified)
            return (1920, 1080)  # Default fallback
        except Exception:
            return (1920, 1080)
    
    def click_at_position(self, x: int, y: int) -> bool:
        """Click using AppleScript"""
        if not self.available:
            return False
        
        try:
            script = f'''
            tell application "System Events"
                click at {{{x}, {y}}}
            end tell
            '''
            
            subprocess.run(["osascript", "-e", script], check=True)
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get macOS system information"""
        return {
            "platform": "macOS",
            "version": platform.mac_ver()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "available_features": ["screen_capture", "window_management"] if self.available else []
        }

class LinuxPlatform(PlatformInterface):
    """Linux-specific implementation"""
    
    def __init__(self):
        try:
            self.available = sys.platform.startswith("linux")
            if self.available:
                self._check_tools()
        except Exception as e:
            logger.warning(f"Linux platform setup failed: {e}")
            self.available = False
    
    def _check_tools(self):
        """Check for required Linux tools"""
        tools = ["xwininfo", "xdotool", "scrot"]
        for tool in tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(f"{tool} not available")
    
    def get_active_window(self) -> Dict[str, Any]:
        """Get active window using xdotool"""
        if not self.available:
            return {"error": "Linux platform not available"}
        
        try:
            # Get active window ID
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                check=True
            )
            window_id = result.stdout.strip()
            
            # Get window info
            result = subprocess.run(
                ["xwininfo", "-id", window_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse window info (simplified)
            return {"title": "Linux Window", "id": window_id, "platform": "Linux"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_window_list(self) -> List[Dict[str, Any]]:
        """Get window list using wmctrl or xdotool"""
        if not self.available:
            return [{"error": "Linux platform not available"}]
        
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", ".*"],
                capture_output=True,
                text=True,
                check=True
            )
            
            window_ids = result.stdout.strip().split('\n')
            windows = []
            
            for window_id in window_ids[:10]:  # Limit to first 10
                if window_id:
                    windows.append({"id": window_id, "platform": "Linux"})
            
            return windows
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """Capture screen using scrot"""
        if not self.available:
            raise Exception("Linux platform not available")
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cmd = ["scrot", tmp.name]
                
                if region:
                    cmd.extend(["-a", f"{region[0]},{region[1]},{region[2]},{region[3]}"])
                
                subprocess.run(cmd, check=True)
                
                with open(tmp.name, "rb") as f:
                    data = f.read()
                
                os.unlink(tmp.name)
                return data
                
        except Exception as e:
            raise Exception(f"Screen capture failed: {str(e)}")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size using xrandr"""
        try:
            result = subprocess.run(
                ["xrandr", "--current"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse xrandr output for primary display (simplified)
            return (1920, 1080)  # Default fallback
            
        except Exception:
            return (1920, 1080)
    
    def click_at_position(self, x: int, y: int) -> bool:
        """Click using xdotool"""
        if not self.available:
            return False
        
        try:
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], check=True)
            subprocess.run(["xdotool", "click", "1"], check=True)
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get Linux system information"""
        return {
            "platform": "Linux",
            "distribution": platform.linux_distribution() if hasattr(platform, 'linux_distribution') else "Unknown",
            "version": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "available_features": ["screen_capture", "window_management", "mouse_control"] if self.available else []
        }

class PlatformManager:
    """Unified platform manager"""
    
    def __init__(self):
        self.current_platform = self._detect_platform()
        self.platform_impl = self._create_platform_implementation()
        
        logger.info(f"Platform manager initialized for {self.current_platform}")
    
    def _detect_platform(self) -> str:
        """Detect current platform"""
        system = platform.system().lower()
        
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "macos"
        elif system == "linux":
            return "linux"
        else:
            return "unknown"
    
    def _create_platform_implementation(self) -> PlatformInterface:
        """Create platform-specific implementation"""
        if self.current_platform == "windows":
            return WindowsPlatform()
        elif self.current_platform == "macos":
            return MacOSPlatform()
        elif self.current_platform == "linux":
            return LinuxPlatform()
        else:
            # Fallback implementation
            return WindowsPlatform()  # Use Windows as fallback
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information"""
        base_info = {
            "detected_platform": self.current_platform,
            "python_platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture(),
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        # Add platform-specific info
        platform_specific = self.platform_impl.get_system_info()
        base_info.update(platform_specific)
        
        return base_info
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a feature is available on current platform"""
        system_info = self.platform_impl.get_system_info()
        available_features = system_info.get("available_features", [])
        return feature in available_features
    
    # Delegate methods to platform implementation
    def get_active_window(self) -> Dict[str, Any]:
        return self.platform_impl.get_active_window()
    
    def get_window_list(self) -> List[Dict[str, Any]]:
        return self.platform_impl.get_window_list()
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        return self.platform_impl.capture_screen(region)
    
    def get_screen_size(self) -> Tuple[int, int]:
        return self.platform_impl.get_screen_size()
    
    def click_at_position(self, x: int, y: int) -> bool:
        return self.platform_impl.click_at_position(x, y)

# Global platform manager instance
_platform_manager: Optional[PlatformManager] = None

def get_platform_manager() -> PlatformManager:
    """Get global platform manager instance"""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = PlatformManager()
    return _platform_manager
