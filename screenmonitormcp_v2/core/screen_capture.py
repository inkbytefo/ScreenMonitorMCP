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
                           format: str = "png") -> Dict[str, Any]:
        """Capture screen and return image data.
        
        Args:
            monitor: Monitor number to capture (0 for primary)
            region: Optional region dict with x, y, width, height
            format: Image format (png, jpeg)
            
        Returns:
            Dict containing success status and image_data as base64 string
        """
        try:
            # Run capture in executor to avoid blocking
            loop = asyncio.get_event_loop()
            image_bytes = await loop.run_in_executor(
                None, self._capture_screen_sync, monitor, region, format
            )
            
            # Convert to base64 for MCP compatibility
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return {
                "success": True,
                "image_data": image_base64,
                "format": format,
                "size": len(image_bytes)
            }
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "image_data": None
            }
    
    async def capture_screen_raw(self, monitor: int = 0, region: Optional[Dict[str, int]] = None, 
                               format: str = "png") -> bytes:
        """Capture screen and return raw image bytes (for backward compatibility).
        
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
    
    async def capture_hq_frame(self, format: str = "png") -> Dict[str, Any]:
        """Capture high-quality frame for PNG high-quality captures.
        
        Args:
            format: Image format (png for high quality, jpeg also supported)
            
        Returns:
            Dict containing success status, image_bytes, dimensions, file_size, and format
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._capture_hq_frame_sync, format
            )
            return result
        except Exception as e:
            self.logger.error(f"HQ frame capture failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _capture_hq_frame_sync(self, format: str) -> Dict[str, Any]:
        """Synchronous high-quality frame capture implementation."""
        try:
            with mss.mss() as sct:
                # Capture primary monitor (monitor 0)
                monitor = sct.monitors[0]  # Primary monitor
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image - handle different pixel formats safely
                try:
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                except Exception:
                    # Fallback to RGBA format if BGRX fails
                    img = Image.frombytes("RGBA", screenshot.size, screenshot.bgra, "raw", "BGRA")
                    img = img.convert("RGB")
                
                # Save to bytes with high quality
                img_buffer = io.BytesIO()
                if format.lower() == "jpeg":
                    img.save(img_buffer, format="JPEG", quality=95, optimize=True)
                else:
                    img.save(img_buffer, format="PNG", optimize=True)
                
                image_bytes = img_buffer.getvalue()
                
                return {
                    "success": True,
                    "image_bytes": image_bytes,
                    "width": img.width,
                    "height": img.height,
                    "file_size": len(image_bytes),
                    "format": format.lower()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def capture_preview_frame(self, quality: int = 40, resolution: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Capture low-quality preview frame for JPEG low-quality captures.
        
        Args:
            quality: JPEG quality (1-100, default 40 for low quality)
            resolution: Optional tuple (width, height) for resizing
            
        Returns:
            Dict containing success status, image_bytes, dimensions, file_size, and format
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._capture_preview_frame_sync, quality, resolution
            )
            return result
        except Exception as e:
            self.logger.error(f"Preview frame capture failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _capture_preview_frame_sync(self, quality: int, resolution: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        """Synchronous preview frame capture implementation."""
        try:
            with mss.mss() as sct:
                # Capture primary monitor (monitor 0)
                monitor = sct.monitors[0]  # Primary monitor
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image - handle different pixel formats safely
                try:
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                except Exception:
                    # Fallback to RGBA format if BGRX fails
                    img = Image.frombytes("RGBA", screenshot.size, screenshot.bgra, "raw", "BGRA")
                    img = img.convert("RGB")
                
                # Resize if resolution specified
                if resolution:
                    img = img.resize(resolution, Image.Resampling.LANCZOS)
                
                # Save to bytes with specified quality (JPEG for preview)
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG", quality=quality, optimize=True)
                image_bytes = img_buffer.getvalue()
                
                return {
                    "success": True,
                    "image_bytes": image_bytes,
                    "width": img.width,
                    "height": img.height,
                    "file_size": len(image_bytes),
                    "format": "jpeg"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def is_available(self) -> bool:
        """Check if screen capture is available."""
        try:
            with mss.mss() as sct:
                return len(sct.monitors) > 0
        except Exception:
            return False