"""
Image Optimization Engine for ScreenMonitorMCP
Provides advanced image optimization with quality control and format conversion
"""

import os
import time
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Image optimization configuration"""
    target_size_kb: Optional[int] = None  # Target file size in KB
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    quality: int = 85  # JPEG quality (1-100)
    format: str = "PNG"  # Output format: PNG, JPEG, WEBP
    preserve_aspect_ratio: bool = True
    enable_compression: bool = True
    enable_enhancement: bool = False
    sharpness_factor: float = 1.0  # 0.0 = blur, 1.0 = original, 2.0 = sharp
    contrast_factor: float = 1.0   # 0.0 = gray, 1.0 = original, 2.0 = high contrast
    brightness_factor: float = 1.0 # 0.0 = black, 1.0 = original, 2.0 = bright

@dataclass
class OptimizationResult:
    """Result of image optimization"""
    success: bool
    optimized_data: Optional[bytes] = None
    original_size_bytes: int = 0
    optimized_size_bytes: int = 0
    compression_ratio: float = 0.0
    processing_time: float = 0.0
    format_changed: bool = False
    dimensions_changed: bool = False
    original_dimensions: Tuple[int, int] = (0, 0)
    optimized_dimensions: Tuple[int, int] = (0, 0)
    error: Optional[str] = None

class ImageOptimizer:
    """Advanced image optimization engine"""
    
    def __init__(self):
        self.supported_formats = {"PNG", "JPEG", "JPG", "WEBP", "BMP", "TIFF"}
        self.optimization_presets = {
            "web": OptimizationConfig(
                target_size_kb=500,
                max_width=1920,
                max_height=1080,
                quality=80,
                format="JPEG",
                enable_compression=True
            ),
            "thumbnail": OptimizationConfig(
                max_width=300,
                max_height=300,
                quality=75,
                format="JPEG",
                enable_compression=True
            ),
            "high_quality": OptimizationConfig(
                quality=95,
                format="PNG",
                enable_compression=False,
                enable_enhancement=True
            ),
            "minimal_size": OptimizationConfig(
                target_size_kb=100,
                quality=60,
                format="JPEG",
                enable_compression=True
            )
        }
        
        logger.info("Image optimizer initialized with presets: " + 
                   ", ".join(self.optimization_presets.keys()))
    
    def optimize_image(self, 
                      image_data: Union[bytes, str, Image.Image],
                      config: OptimizationConfig) -> OptimizationResult:
        """Optimize an image according to the given configuration"""
        start_time = time.time()
        
        try:
            # Load image
            if isinstance(image_data, str):
                # Base64 string
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                original_size = len(image_bytes)
            elif isinstance(image_data, bytes):
                # Raw bytes
                img = Image.open(io.BytesIO(image_data))
                original_size = len(image_data)
            elif isinstance(image_data, Image.Image):
                # PIL Image object
                img = image_data.copy()
                # Estimate original size
                buffer = io.BytesIO()
                img.save(buffer, format=img.format or "PNG")
                original_size = len(buffer.getvalue())
            else:
                raise ValueError("Unsupported image data type")
            
            original_dimensions = img.size
            
            # Convert to RGB if necessary (for JPEG output)
            if config.format.upper() == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            
            # Resize if needed
            dimensions_changed = False
            if config.max_width or config.max_height:
                new_size = self._calculate_new_size(
                    img.size, 
                    config.max_width, 
                    config.max_height, 
                    config.preserve_aspect_ratio
                )
                if new_size != img.size:
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    dimensions_changed = True
            
            # Apply enhancements if enabled
            if config.enable_enhancement:
                img = self._apply_enhancements(img, config)
            
            # Optimize and save
            optimized_data = self._save_optimized(img, config)
            
            # Check target size and adjust quality if needed
            if config.target_size_kb and config.format.upper() == "JPEG":
                optimized_data = self._adjust_for_target_size(
                    img, config, config.target_size_kb * 1024
                )
            
            processing_time = time.time() - start_time
            optimized_size = len(optimized_data)
            compression_ratio = (original_size - optimized_size) / original_size * 100
            
            return OptimizationResult(
                success=True,
                optimized_data=optimized_data,
                original_size_bytes=original_size,
                optimized_size_bytes=optimized_size,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                format_changed=True,  # Always true since we specify format
                dimensions_changed=dimensions_changed,
                original_dimensions=original_dimensions,
                optimized_dimensions=img.size
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Image optimization failed: {str(e)}")
            
            return OptimizationResult(
                success=False,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _calculate_new_size(self, 
                           current_size: Tuple[int, int],
                           max_width: Optional[int],
                           max_height: Optional[int],
                           preserve_aspect_ratio: bool) -> Tuple[int, int]:
        """Calculate new image size based on constraints"""
        width, height = current_size
        
        if not preserve_aspect_ratio:
            new_width = max_width if max_width and max_width < width else width
            new_height = max_height if max_height and max_height < height else height
            return (new_width, new_height)
        
        # Preserve aspect ratio
        scale_width = max_width / width if max_width and max_width < width else 1.0
        scale_height = max_height / height if max_height and max_height < height else 1.0
        scale = min(scale_width, scale_height)
        
        if scale < 1.0:
            return (int(width * scale), int(height * scale))
        
        return current_size
    
    def _apply_enhancements(self, img: Image.Image, config: OptimizationConfig) -> Image.Image:
        """Apply image enhancements"""
        try:
            # Sharpness
            if config.sharpness_factor != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(config.sharpness_factor)
            
            # Contrast
            if config.contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(config.contrast_factor)
            
            # Brightness
            if config.brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(config.brightness_factor)
            
            return img
            
        except Exception as e:
            logger.warning(f"Enhancement failed: {str(e)}")
            return img
    
    def _save_optimized(self, img: Image.Image, config: OptimizationConfig) -> bytes:
        """Save optimized image to bytes"""
        buffer = io.BytesIO()
        
        save_kwargs = {"format": config.format.upper()}
        
        if config.format.upper() == "JPEG":
            save_kwargs.update({
                "quality": config.quality,
                "optimize": config.enable_compression
            })
        elif config.format.upper() == "PNG":
            save_kwargs.update({
                "optimize": config.enable_compression,
                "compress_level": 6 if config.enable_compression else 1
            })
        elif config.format.upper() == "WEBP":
            save_kwargs.update({
                "quality": config.quality,
                "optimize": config.enable_compression
            })
        
        img.save(buffer, **save_kwargs)
        return buffer.getvalue()
    
    def _adjust_for_target_size(self, 
                               img: Image.Image, 
                               config: OptimizationConfig, 
                               target_bytes: int) -> bytes:
        """Adjust quality to meet target file size"""
        quality = config.quality
        min_quality = 10
        
        for _ in range(10):  # Max 10 iterations
            test_config = OptimizationConfig(
                quality=quality,
                format=config.format,
                enable_compression=config.enable_compression
            )
            
            data = self._save_optimized(img, test_config)
            
            if len(data) <= target_bytes or quality <= min_quality:
                return data
            
            # Reduce quality
            quality = max(min_quality, quality - 10)
        
        # Return best effort
        return data
    
    def optimize_with_preset(self, 
                           image_data: Union[bytes, str, Image.Image],
                           preset_name: str) -> OptimizationResult:
        """Optimize image using a predefined preset"""
        if preset_name not in self.optimization_presets:
            return OptimizationResult(
                success=False,
                error=f"Unknown preset: {preset_name}. Available: {list(self.optimization_presets.keys())}"
            )
        
        config = self.optimization_presets[preset_name]
        return self.optimize_image(image_data, config)
    
    def batch_optimize(self, 
                      images: Dict[str, Union[bytes, str, Image.Image]],
                      config: OptimizationConfig) -> Dict[str, OptimizationResult]:
        """Optimize multiple images"""
        results = {}
        
        for image_id, image_data in images.items():
            results[image_id] = self.optimize_image(image_data, config)
        
        return results
    
    def get_image_info(self, image_data: Union[bytes, str, Image.Image]) -> Dict[str, Any]:
        """Get information about an image"""
        try:
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                size_bytes = len(image_bytes)
            elif isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
                size_bytes = len(image_data)
            elif isinstance(image_data, Image.Image):
                img = image_data
                buffer = io.BytesIO()
                img.save(buffer, format=img.format or "PNG")
                size_bytes = len(buffer.getvalue())
            else:
                raise ValueError("Unsupported image data type")
            
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "size_bytes": size_bytes,
                "size_kb": round(size_bytes / 1024, 2),
                "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
            }
            
        except Exception as e:
            return {"error": str(e)}

# Global image optimizer instance
_image_optimizer: Optional[ImageOptimizer] = None

def get_image_optimizer() -> ImageOptimizer:
    """Get global image optimizer instance"""
    global _image_optimizer
    if _image_optimizer is None:
        _image_optimizer = ImageOptimizer()
    return _image_optimizer
