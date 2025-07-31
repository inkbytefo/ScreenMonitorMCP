"""AI service for ScreenMonitorMCP v2 with OpenAI compatibility."""

from typing import Optional, Dict, Any, List
import logging
import sys
from openai import AsyncOpenAI
try:
    from ..server.config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from server.config import config

# Configure logger to use stderr only for MCP compatibility
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)  # Minimal logging for MCP mode


class AIService:
    """AI service supporting OpenAI and compatible APIs."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client with configuration."""
        if not config.openai_api_key:
            logger.warning("OpenAI API key not configured")
            return
        
        client_kwargs = {
            "api_key": config.openai_api_key,
            "timeout": config.openai_timeout,
        }
        
        # Add base URL if provided (for OpenAI compatible APIs)
        if config.openai_base_url:
            client_kwargs["base_url"] = config.openai_base_url
            logger.info(
                "Using OpenAI compatible API",
                base_url=config.openai_base_url,
                model=config.openai_model
            )
        else:
            logger.info(
                "Using OpenAI API",
                model=config.openai_model
            )
        
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def analyze_image(
        self,
        image_base64: str,
        prompt: str = "What's in this image?",
        model: Optional[str] = None,
        max_tokens: int = 300
    ) -> Dict[str, Any]:
        """
        Analyze an image using AI vision capabilities.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Prompt for the AI
            model: Model to use (defaults to config.openai_model)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with analysis results
        """
        if not self.client:
            return {
                "error": "AI service not configured",
                "success": False
            }
        
        try:
            model_to_use = model or config.openai_model
            
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model_to_use,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error("AI analysis failed", error=str(e))
            return {
                "error": str(e),
                "success": False
            }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate chat completion using any OpenAI compatible model.
        
        Args:
            messages: List of chat messages
            model: Model to use (defaults to config.openai_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Dict with completion results
        """
        if not self.client:
            return {
                "error": "AI service not configured",
                "success": False
            }
        
        try:
            model_to_use = model or config.openai_model
            
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model_to_use,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error("Chat completion failed", error=str(e))
            return {
                "error": str(e),
                "success": False
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models from the API.
        
        Returns:
            Dict with available models
        """
        if not self.client:
            return {
                "error": "AI service not configured",
                "success": False
            }
        
        try:
            models = await self.client.models.list()
            return {
                "success": True,
                "models": [model.id for model in models.data]
            }
            
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return {
                "error": str(e),
                "success": False
            }
    
    def is_configured(self) -> bool:
        """Check if AI service is properly configured."""
        return self.client is not None
    
    def is_available(self) -> bool:
        """Check if AI service is available."""
        return self.client is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI service status."""
        return {
            "configured": self.is_configured(),
            "available": self.is_available(),
            "model": config.openai_model if self.is_configured() else None,
            "base_url": config.openai_base_url if self.is_configured() else None
        }


# Global AI service instance
ai_service = AIService()