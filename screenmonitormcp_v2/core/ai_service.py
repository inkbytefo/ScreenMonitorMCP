"""AI Monitor Analysis Expert Service for ScreenMonitorMCP v2 with OpenAI compatibility and memory integration.

This module provides specialized AI services for monitor analysis with OpenAI-compatible interface,
integrated memory system, comprehensive error handling, and expert-level monitoring capabilities."""

from typing import Optional, Dict, Any, List
import logging
import sys
from openai import AsyncOpenAI
try:
    from ..server.config import config
    from .memory_system import memory_system
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from server.config import config
    from core.memory_system import memory_system

# Configure logger to use stderr only for MCP compatibility
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)  # Minimal logging for MCP mode


class AIService:
    """AI Monitor Analysis Expert Service with OpenAI compatibility and memory integration.
    
    Provides specialized AI analysis capabilities for monitor analysis with memory storage,
    retrieval, and expert-level monitoring insights. Optimized for screen monitoring,
    system performance analysis, and anomaly detection."""
    
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
        max_tokens: int = 1500,
        store_in_memory: bool = True,
        stream_id: Optional[str] = None,
        sequence: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using AI vision capabilities with memory integration.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Prompt for the AI
            model: Model to use (defaults to config.openai_model)
            max_tokens: Maximum tokens in response
            store_in_memory: Whether to store result in memory system
            stream_id: Optional stream identifier for memory storage
            sequence: Optional sequence number for memory storage
            tags: Optional tags for memory categorization
            
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
            
            result = {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model_to_use,
                "prompt": prompt,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Store in memory system if requested
            if store_in_memory:
                try:
                    memory_tags = tags or []
                    if "image_analysis" not in memory_tags:
                        memory_tags.append("image_analysis")
                    
                    memory_id = await memory_system.store_analysis(
                        analysis_result=result,
                        stream_id=stream_id,
                        sequence=sequence,
                        tags=memory_tags
                    )
                    result["memory_id"] = memory_id
                    logger.debug(f"Stored analysis in memory: {memory_id}")
                except Exception as memory_error:
                    logger.warning(f"Failed to store analysis in memory: {memory_error}")
                    # Don't fail the entire request if memory storage fails
            
            return result
            
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
        max_tokens: int = 2000,
        temperature: float = 0.2
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
            "base_url": config.openai_base_url if self.is_configured() else None,
            "memory_enabled": True
        }
    
    async def analyze_scene_from_memory(self, 
                                      query: str,
                                      stream_id: Optional[str] = None,
                                      time_range_hours: int = 1) -> Dict[str, Any]:
        """Analyze scene based on memory data.
        
        Args:
            query: Scene analysis query
            stream_id: Optional stream ID to filter by
            time_range_hours: Hours to look back in memory
            
        Returns:
            Scene analysis result
        """
        try:
            from datetime import timedelta
            
            # Get relevant memory entries
            memory_entries = await memory_system.query_memory(
                query=query,
                stream_id=stream_id,
                limit=20,
                time_range=timedelta(hours=time_range_hours)
            )
            
            if not memory_entries:
                return {
                    "success": False,
                    "error": "No relevant memory data found",
                    "query": query
                }
            
            # Prepare context from memory
            context_data = []
            for entry in memory_entries:
                if entry.entry_type == "analysis":
                    context_data.append({
                        "timestamp": entry.timestamp,
                        "analysis": entry.content.get("response", ""),
                        "type": "analysis"
                    })
                elif entry.entry_type == "scene":
                    context_data.append({
                        "timestamp": entry.timestamp,
                        "description": entry.content.get("description", ""),
                        "objects": entry.content.get("objects", []),
                        "activities": entry.content.get("activities", []),
                        "type": "scene"
                    })
            
            # Create analysis prompt with context
            context_summary = "\n".join([
                f"[{item['timestamp']}] {item.get('analysis', item.get('description', ''))}" 
                for item in context_data[:10]  # Limit context to avoid token limits
            ])
            
            analysis_prompt = f"""
            Based on the following recent screen analysis history, please answer this query: "{query}"
            
            Recent Analysis History:
            {context_summary}
            
            Please provide a comprehensive answer based on the available context data.
            """
            
            # Use chat completion for analysis
            result = await self.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing screen content based on historical data. Provide accurate and helpful responses based on the given context."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=500
            )
            
            if result.get("success"):
                # Store the scene analysis result
                analysis_result = {
                    "query": query,
                    "response": result["response"],
                    "context_entries": len(memory_entries),
                    "time_range_hours": time_range_hours,
                    "model": result["model"]
                }
                
                try:
                    memory_id = await memory_system.store_analysis(
                        analysis_result=analysis_result,
                        stream_id=stream_id,
                        tags=["scene_query", "memory_analysis"]
                    )
                    analysis_result["memory_id"] = memory_id
                except Exception as memory_error:
                    logger.warning(f"Failed to store scene analysis in memory: {memory_error}")
                
                return {
                    "success": True,
                    **analysis_result
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Scene analysis from memory failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics.
        
        Returns:
            Memory system statistics
        """
        try:
            stats = await memory_system.get_statistics()
            return {
                "success": True,
                "statistics": stats
            }
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def query_memory_direct(self, 
                                query: str,
                                entry_type: Optional[str] = None,
                                stream_id: Optional[str] = None,
                                limit: int = 10) -> Dict[str, Any]:
        """Direct memory query interface.
        
        Args:
            query: Search query
            entry_type: Filter by entry type
            stream_id: Filter by stream ID
            limit: Maximum results
            
        Returns:
            Memory query results
        """
        try:
            entries = await memory_system.query_memory(
                query=query,
                entry_type=entry_type,
                stream_id=stream_id,
                limit=limit
            )
            
            # Convert entries to serializable format
            results = []
            for entry in entries:
                results.append({
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "entry_type": entry.entry_type,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "tags": entry.tags,
                    "stream_id": entry.stream_id,
                    "sequence": entry.sequence
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Memory query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }


# Global AI service instance
ai_service = AIService()