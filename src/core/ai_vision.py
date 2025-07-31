"""
AI Vision Analysis Module for ScreenMonitorMCP v2

This module provides AI-powered screen content analysis using vision models.
"""

import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

import openai
from openai import AsyncOpenAI

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ..server.config import config
    from ..models.responses import ScreenAnalysisResponse
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from server.config import config
    from models.responses import ScreenAnalysisResponse

logger = logging.getLogger(__name__)


class AIVisionAnalyzer:
    """AI Vision Analyzer for screen content analysis."""
    
    def __init__(self):
        """Initialize the AI Vision Analyzer."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client with configuration."""
        try:
            api_key = config.openai_api_key
            base_url = config.openai_base_url or "https://api.openai.com/v1"
            
            if not api_key:
                logger.error("AI API key not configured")
                return
                
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"AI Vision Analyzer initialized with base_url: {base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            self.client = None
    
    async def analyze_image(
        self,
        image_base64: str,
        prompt: str = "What's on this screen?",
        model: str = None,
        max_tokens: int = 300
    ) -> str:
        """
        Analyze a screen image using AI vision model.
        
        Args:
            image_base64: Base64 encoded image data
            prompt: Analysis prompt/question
            model: AI model to use (defaults to configured model)
            max_tokens: Maximum tokens in response
            
        Returns:
            AI analysis text response
            
        Raises:
            ValueError: If AI service is not configured
            Exception: For API errors
        """
        if not self.client:
            raise ValueError("AI service not configured or unavailable")
        
        if not model:
            model = config.openai_model
        
        try:
            logger.info(f"Analyzing image with model: {model}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"AI analysis completed. Tokens used: {usage}")
            return content
            
        except openai.AuthenticationError as e:
            logger.error(f"AI API authentication failed: {e}")
            raise ValueError("Invalid AI API credentials")
        except openai.RateLimitError as e:
            logger.error(f"AI API rate limit exceeded: {e}")
            raise Exception("AI service rate limit exceeded")
        except openai.APIError as e:
            logger.error(f"AI API error: {e}")
            raise Exception(f"AI service error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during AI analysis: {e}")
            raise Exception(f"Failed to analyze image: {e}")


async def stream_analysis_generator(
    stream_id: str,
    interval_seconds: int = 10,
    prompt: str = "Analyze this screen content and provide a detailed summary of what's happening.",
    model: str = None,
    max_tokens: int = 300
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate continuous AI analysis of screen content.
    
    Args:
        stream_id: Unique stream identifier
        interval_seconds: Analysis interval in seconds
        prompt: Analysis prompt for AI
        model: AI model to use
        max_tokens: Maximum tokens in response
        
    Yields:
        Analysis results as dictionaries
    """
    from ..core.streaming import ScreenStreamer
    
    analyzer = AIVisionAnalyzer()
    streamer = ScreenStreamer()
    
    if not analyzer.client:
        yield {
            "type": "error",
            "data": {"error": "AI service not configured"},
            "timestamp": datetime.now().isoformat(),
            "stream_id": stream_id
        }
        return
    
    sequence = 0
    
    try:
        while True:
            try:
                # Capture screen
                capture_result = await streamer.capture_screen(
                    monitor=1,
                    quality=80,
                    format="jpeg"
                )
                
                if not capture_result.get("image_data"):
                    yield {
                        "type": "error",
                        "data": {"error": "Failed to capture screen - no image data"},
                        "timestamp": datetime.now().isoformat(),
                        "stream_id": stream_id,
                        "sequence": sequence
                    }
                    await asyncio.sleep(interval_seconds)
                    continue
                
                # Analyze with AI
                image_base64 = capture_result["image_data"]
                analysis = await analyzer.analyze_image(
                    image_base64=image_base64,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens
                )
                
                # Create response
                response_data = {
                    "analysis": analysis,
                    "model": model or config.ai_provider.get("model", "gpt-4-vision-preview"),
                    "prompt": prompt,
                    "capture_info": {
                        "timestamp": capture_result["timestamp"],
                        "monitor": capture_result["monitor"],
                        "width": capture_result["width"],
                        "height": capture_result["height"],
                        "format": capture_result["format"],
                        "size": capture_result["size"]
                    },
                    "usage": {
                        "prompt_tokens": 0,  # Will be updated when OpenAI returns usage
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                
                yield {
                    "type": "analysis",
                    "data": response_data,
                    "timestamp": datetime.now().isoformat(),
                    "stream_id": stream_id,
                    "sequence": sequence
                }
                
                sequence += 1
                
            except Exception as e:
                logger.error(f"Error in stream analysis: {e}")
                yield {
                    "type": "error",
                    "data": {"error": str(e)},
                    "timestamp": datetime.now().isoformat(),
                    "stream_id": stream_id,
                    "sequence": sequence
                }
            
            await asyncio.sleep(interval_seconds)
            
    except asyncio.CancelledError:
        logger.info(f"Stream analysis cancelled for stream: {stream_id}")
        yield {
            "type": "status",
            "data": {"status": "stopped"},
            "timestamp": datetime.now().isoformat(),
            "stream_id": stream_id,
            "sequence": sequence
        }
    except Exception as e:
        logger.error(f"Fatal error in stream analysis: {e}")
        yield {
            "type": "error",
            "data": {"error": f"Fatal error: {str(e)}"},
            "timestamp": datetime.now().isoformat(),
            "stream_id": stream_id,
            "sequence": sequence
        }


# Utility functions
async def validate_ai_config() -> Dict[str, Any]:
    """Validate AI service configuration."""
    try:
        api_key = config.ai_provider.get("api_key")
        base_url = config.ai_provider.get("base_url", "https://api.openai.com/v1")
        model = config.ai_provider.get("model", "gpt-4-vision-preview")
        
        if not api_key:
            return {
                "service_available": False,
                "configured": False,
                "error": "API key not configured"
            }
        
        # Test API connectivity
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        # Try to list models (this will validate credentials)
        await client.models.list()
        
        return {
            "service_available": True,
            "configured": True,
            "provider": "openai",
            "base_url": base_url,
            "model": model,
            "models_available": 1  # Simplified for now
        }
        
    except Exception as e:
        return {
            "service_available": False,
            "configured": True,
            "error": str(e)
        }