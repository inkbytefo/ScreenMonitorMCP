#!/usr/bin/env python3
"""
Screen Capture Module for ScreenMonitorMCP v2

This module provides screen capture functionality using the mss library.
It supports multi-monitor setups and various image formats.

Author: ScreenMonitorMCP Team
Version: 2.0.0
License: MIT
"""

import asyncio
import base64
import io
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import mss
from PIL import Image

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Screen capture functionality using mss library."""
    
    def __init__(self):
        """Initialize the screen capture system."""
        self.logger = logging.getLogger(__name__)
    
    async def capture_screen(self, monitor: int = 0, region: Optional[Dict[str, int]] = None, 
                           format: str = "png") -> bytes:
        """Capture screen and return image data.
        
        Args:
            monitor: Monitor number to capture (0 for primary)
            region: Optional region dict with x, y, width, height
            format: Image format (png, jpeg)
            
        Returns:
            Image data as bytes
        """
        try:
            # Run capture in executor to avoid blocking
            loop = asyncio.get_event_loop()
            image_data = await loop.run_in_executor(
                None, self._capture_screen_sync, monitor, region, format
            )
            return image_data
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            raise
    
    def _capture_screen_sync(self, monitor: int, region: Optional[Dict[str, int]], 
                           format: str) -> bytes:
        """Synchronous screen capture implementation."""
        with mss.mss() as sct:
            # Get monitor info
            if monitor >= len(sct.monitors):
                raise ValueError(f"Monitor {monitor} not found. Available: {len(sct.monitors) - 1}")
            
            # Use specific region or full monitor
            if region:
                capture_area = {
                    "left": region["x"],
                    "top": region["y"],
                    "width": region["width"],
                    "height": region["height"]
                }
            else:
                capture_area = sct.monitors[monitor]
            
            # Capture screenshot
            screenshot = sct.grab(capture_area)
            
            # Convert to PIL Image - handle different pixel formats safely
            try:
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            except Exception:
                # Fallback to RGBA format if BGRX fails
                img = Image.frombytes("RGBA", screenshot.size, screenshot.bgra, "raw", "BGRA")
                img = img.convert("RGB")
            
            # Save to bytes
            img_bytes = io.BytesIO()
            if format.lower() == "jpeg":
                img.save(img_bytes, format="JPEG", quality=85)
            else:
                img.save(img_bytes, format="PNG")
            
            return img_bytes.getvalue()
    
    async def get_monitors(self) -> list[Dict[str, Any]]:
        """Get information about available monitors."""
        try:
            loop = asyncio.get_event_loop()
            monitors = await loop.run_in_executor(None, self._get_monitors_sync)
            return monitors
        except Exception as e:
            self.logger.error(f"Failed to get monitors: {e}")
            raise
    
    def _get_monitors_sync(self) -> list[Dict[str, Any]]:
        """Synchronous monitor detection."""
        with mss.mss() as sct:
            monitors = []
            for i, monitor in enumerate(sct.monitors):
                monitors.append({
                    "id": i,
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                    "is_primary": i == 0
                })
            return monitors
    
    def is_available(self) -> bool:
        """Check if screen capture is available."""
        try:
            with mss.mss() as sct:
                return len(sct.monitors) > 0
        except Exception:
            return False