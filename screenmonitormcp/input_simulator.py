"""
Advanced Input Simulation for ScreenMonitorMCP
Provides keyboard input, drag & drop, multi-touch gestures, and hotkey combinations
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class InputType(Enum):
    """Types of input simulation"""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GESTURE = "gesture"
    HOTKEY = "hotkey"

class MouseButton(Enum):
    """Mouse button types"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"

@dataclass
class KeyboardInput:
    """Keyboard input configuration"""
    text: str = ""
    keys: List[str] = None
    modifiers: List[str] = None  # ctrl, alt, shift, cmd
    delay_between_keys: float = 0.05
    hold_duration: float = 0.1

@dataclass
class MouseInput:
    """Mouse input configuration"""
    x: int = 0
    y: int = 0
    button: MouseButton = MouseButton.LEFT
    clicks: int = 1
    delay_between_clicks: float = 0.1
    drag_to: Optional[Tuple[int, int]] = None
    scroll_direction: Optional[str] = None  # up, down, left, right
    scroll_amount: int = 3

@dataclass
class GestureInput:
    """Gesture input configuration"""
    gesture_type: str = ""  # swipe, pinch, rotate
    start_points: List[Tuple[int, int]] = None
    end_points: List[Tuple[int, int]] = None
    duration: float = 1.0
    steps: int = 10

class InputSimulator:
    """Advanced input simulation system"""
    
    def __init__(self):
        self.available_backends = self._detect_backends()
        self.active_backend = self._select_backend()
        
        # Input history
        self.input_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Timing
        self.default_delay = 0.1
        self.safety_delay = 0.05  # Minimum delay between actions
        
        # State
        self._lock = threading.RLock()
        
        logger.info(f"Input simulator initialized with backend: {self.active_backend}")
    
    def _detect_backends(self) -> Dict[str, bool]:
        """Detect available input backends"""
        backends = {
            "pyautogui": False,
            "pynput": False,
            "keyboard": False,
            "mouse": False,
            "platform_native": False
        }
        
        try:
            import pyautogui
            backends["pyautogui"] = True
        except ImportError:
            pass
        
        try:
            import pynput
            backends["pynput"] = True
        except ImportError:
            pass
        
        try:
            import keyboard
            backends["keyboard"] = True
        except ImportError:
            pass
        
        try:
            import mouse
            backends["mouse"] = True
        except ImportError:
            pass
        
        return backends
    
    def _select_backend(self) -> str:
        """Select the best available backend"""
        if self.available_backends.get("pyautogui"):
            return "pyautogui"
        elif self.available_backends.get("pynput"):
            return "pynput"
        elif self.available_backends.get("keyboard") and self.available_backends.get("mouse"):
            return "keyboard_mouse"
        else:
            return "none"
    
    def _record_input(self, input_type: InputType, config: Dict[str, Any], success: bool):
        """Record input action in history"""
        with self._lock:
            record = {
                "timestamp": time.time(),
                "type": input_type.value,
                "config": config,
                "success": success,
                "backend": self.active_backend
            }
            
            self.input_history.append(record)
            
            # Trim history if needed
            if len(self.input_history) > self.max_history:
                self.input_history = self.input_history[-self.max_history:]
    
    def simulate_keyboard_input(self, config: KeyboardInput) -> bool:
        """Simulate keyboard input"""
        if self.active_backend == "none":
            logger.error("No keyboard backend available")
            return False
        
        try:
            if self.active_backend == "pyautogui":
                return self._pyautogui_keyboard(config)
            elif self.active_backend == "pynput":
                return self._pynput_keyboard(config)
            elif self.active_backend == "keyboard_mouse":
                return self._keyboard_backend(config)
            
            return False
            
        except Exception as e:
            logger.error(f"Keyboard simulation failed: {str(e)}")
            self._record_input(InputType.KEYBOARD, config.__dict__, False)
            return False
    
    def _pyautogui_keyboard(self, config: KeyboardInput) -> bool:
        """Keyboard input using pyautogui"""
        import pyautogui
        
        try:
            # Type text
            if config.text:
                pyautogui.typewrite(config.text, interval=config.delay_between_keys)
            
            # Press key combinations
            if config.keys:
                if config.modifiers:
                    # Press with modifiers
                    pyautogui.hotkey(*config.modifiers, *config.keys)
                else:
                    # Press keys sequentially
                    for key in config.keys:
                        pyautogui.press(key)
                        time.sleep(config.delay_between_keys)
            
            self._record_input(InputType.KEYBOARD, config.__dict__, True)
            return True
            
        except Exception as e:
            logger.error(f"PyAutoGUI keyboard error: {str(e)}")
            return False
    
    def _pynput_keyboard(self, config: KeyboardInput) -> bool:
        """Keyboard input using pynput"""
        from pynput import keyboard
        
        try:
            kb = keyboard.Controller()
            
            # Type text
            if config.text:
                for char in config.text:
                    kb.type(char)
                    time.sleep(config.delay_between_keys)
            
            # Press key combinations
            if config.keys:
                # Convert modifier names
                modifier_map = {
                    "ctrl": keyboard.Key.ctrl,
                    "alt": keyboard.Key.alt,
                    "shift": keyboard.Key.shift,
                    "cmd": keyboard.Key.cmd
                }
                
                # Press modifiers
                pressed_modifiers = []
                if config.modifiers:
                    for mod in config.modifiers:
                        if mod in modifier_map:
                            kb.press(modifier_map[mod])
                            pressed_modifiers.append(modifier_map[mod])
                
                # Press keys
                for key in config.keys:
                    try:
                        # Try as special key first
                        special_key = getattr(keyboard.Key, key, None)
                        if special_key:
                            kb.press(special_key)
                            time.sleep(config.hold_duration)
                            kb.release(special_key)
                        else:
                            # Regular character
                            kb.press(key)
                            time.sleep(config.hold_duration)
                            kb.release(key)
                    except AttributeError:
                        # Fallback to character
                        kb.type(key)
                    
                    time.sleep(config.delay_between_keys)
                
                # Release modifiers
                for mod in reversed(pressed_modifiers):
                    kb.release(mod)
            
            self._record_input(InputType.KEYBOARD, config.__dict__, True)
            return True
            
        except Exception as e:
            logger.error(f"Pynput keyboard error: {str(e)}")
            return False
    
    def _keyboard_backend(self, config: KeyboardInput) -> bool:
        """Keyboard input using keyboard library"""
        import keyboard
        
        try:
            # Type text
            if config.text:
                keyboard.write(config.text, delay=config.delay_between_keys)
            
            # Press key combinations
            if config.keys:
                if config.modifiers:
                    # Build hotkey string
                    hotkey_parts = config.modifiers + config.keys
                    hotkey_string = "+".join(hotkey_parts)
                    keyboard.send(hotkey_string)
                else:
                    # Press keys sequentially
                    for key in config.keys:
                        keyboard.send(key)
                        time.sleep(config.delay_between_keys)
            
            self._record_input(InputType.KEYBOARD, config.__dict__, True)
            return True
            
        except Exception as e:
            logger.error(f"Keyboard backend error: {str(e)}")
            return False
    
    def simulate_mouse_input(self, config: MouseInput) -> bool:
        """Simulate mouse input"""
        if self.active_backend == "none":
            logger.error("No mouse backend available")
            return False
        
        try:
            if self.active_backend == "pyautogui":
                return self._pyautogui_mouse(config)
            elif self.active_backend == "pynput":
                return self._pynput_mouse(config)
            elif self.active_backend == "keyboard_mouse":
                return self._mouse_backend(config)
            
            return False
            
        except Exception as e:
            logger.error(f"Mouse simulation failed: {str(e)}")
            self._record_input(InputType.MOUSE, config.__dict__, False)
            return False
    
    def _pyautogui_mouse(self, config: MouseInput) -> bool:
        """Mouse input using pyautogui"""
        import pyautogui
        
        try:
            # Move to position
            pyautogui.moveTo(config.x, config.y)
            time.sleep(self.safety_delay)
            
            # Handle drag
            if config.drag_to:
                pyautogui.dragTo(config.drag_to[0], config.drag_to[1], 
                               duration=0.5, button=config.button.value)
            
            # Handle scroll
            elif config.scroll_direction:
                scroll_amount = config.scroll_amount
                if config.scroll_direction in ["down", "left"]:
                    scroll_amount = -scroll_amount
                
                if config.scroll_direction in ["up", "down"]:
                    pyautogui.scroll(scroll_amount)
                else:
                    pyautogui.hscroll(scroll_amount)
            
            # Handle clicks
            else:
                for _ in range(config.clicks):
                    pyautogui.click(button=config.button.value)
                    if config.clicks > 1:
                        time.sleep(config.delay_between_clicks)
            
            self._record_input(InputType.MOUSE, config.__dict__, True)
            return True
            
        except Exception as e:
            logger.error(f"PyAutoGUI mouse error: {str(e)}")
            return False
    
    def _pynput_mouse(self, config: MouseInput) -> bool:
        """Mouse input using pynput"""
        from pynput import mouse
        
        try:
            m = mouse.Controller()
            
            # Move to position
            m.position = (config.x, config.y)
            time.sleep(self.safety_delay)
            
            # Button mapping
            button_map = {
                MouseButton.LEFT: mouse.Button.left,
                MouseButton.RIGHT: mouse.Button.right,
                MouseButton.MIDDLE: mouse.Button.middle
            }
            
            button = button_map.get(config.button, mouse.Button.left)
            
            # Handle drag
            if config.drag_to:
                m.press(button)
                m.position = config.drag_to
                time.sleep(0.1)
                m.release(button)
            
            # Handle scroll
            elif config.scroll_direction:
                scroll_amount = config.scroll_amount
                if config.scroll_direction in ["down", "left"]:
                    scroll_amount = -scroll_amount
                
                if config.scroll_direction in ["up", "down"]:
                    m.scroll(0, scroll_amount)
                else:
                    m.scroll(scroll_amount, 0)
            
            # Handle clicks
            else:
                for _ in range(config.clicks):
                    m.click(button)
                    if config.clicks > 1:
                        time.sleep(config.delay_between_clicks)
            
            self._record_input(InputType.MOUSE, config.__dict__, True)
            return True
            
        except Exception as e:
            logger.error(f"Pynput mouse error: {str(e)}")
            return False
    
    def _mouse_backend(self, config: MouseInput) -> bool:
        """Mouse input using mouse library"""
        import mouse
        
        try:
            # Move to position
            mouse.move(config.x, config.y)
            time.sleep(self.safety_delay)
            
            # Handle drag
            if config.drag_to:
                mouse.drag(config.x, config.y, config.drag_to[0], config.drag_to[1], 
                          absolute=True, duration=0.5)
            
            # Handle scroll
            elif config.scroll_direction:
                scroll_amount = config.scroll_amount
                if config.scroll_direction in ["down", "left"]:
                    scroll_amount = -scroll_amount
                
                mouse.wheel(scroll_amount)
            
            # Handle clicks
            else:
                button = config.button.value
                for _ in range(config.clicks):
                    mouse.click(button)
                    if config.clicks > 1:
                        time.sleep(config.delay_between_clicks)
            
            self._record_input(InputType.MOUSE, config.__dict__, True)
            return True
            
        except Exception as e:
            logger.error(f"Mouse backend error: {str(e)}")
            return False
    
    def simulate_hotkey(self, keys: List[str], modifiers: List[str] = None) -> bool:
        """Simulate hotkey combination"""
        config = KeyboardInput(keys=keys, modifiers=modifiers or [])
        return self.simulate_keyboard_input(config)
    
    def simulate_text_input(self, text: str, typing_speed: float = 0.05) -> bool:
        """Simulate text typing"""
        config = KeyboardInput(text=text, delay_between_keys=typing_speed)
        return self.simulate_keyboard_input(config)
    
    def simulate_click(self, x: int, y: int, button: MouseButton = MouseButton.LEFT, clicks: int = 1) -> bool:
        """Simulate mouse click"""
        config = MouseInput(x=x, y=y, button=button, clicks=clicks)
        return self.simulate_mouse_input(config)
    
    def simulate_drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                     button: MouseButton = MouseButton.LEFT) -> bool:
        """Simulate drag operation"""
        config = MouseInput(x=start_x, y=start_y, button=button, drag_to=(end_x, end_y))
        return self.simulate_mouse_input(config)
    
    def simulate_scroll(self, x: int, y: int, direction: str, amount: int = 3) -> bool:
        """Simulate scroll operation"""
        config = MouseInput(x=x, y=y, scroll_direction=direction, scroll_amount=amount)
        return self.simulate_mouse_input(config)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get input simulation capabilities"""
        return {
            "available_backends": self.available_backends,
            "active_backend": self.active_backend,
            "supported_features": {
                "keyboard_input": self.active_backend != "none",
                "mouse_input": self.active_backend != "none",
                "hotkeys": self.active_backend != "none",
                "drag_drop": self.active_backend != "none",
                "scrolling": self.active_backend != "none",
                "text_typing": self.active_backend != "none"
            },
            "input_history_size": len(self.input_history),
            "max_history": self.max_history
        }
    
    def get_input_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent input history"""
        with self._lock:
            return self.input_history[-limit:] if limit > 0 else self.input_history.copy()
    
    def clear_input_history(self):
        """Clear input history"""
        with self._lock:
            self.input_history.clear()
        logger.info("Input history cleared")

# Global input simulator instance
_input_simulator: Optional[InputSimulator] = None

def get_input_simulator() -> InputSimulator:
    """Get global input simulator instance"""
    global _input_simulator
    if _input_simulator is None:
        _input_simulator = InputSimulator()
    return _input_simulator
