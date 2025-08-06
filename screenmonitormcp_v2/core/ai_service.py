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
    
    # Specialized AI methods consolidated from ai_analyzer.py
    
    async def detect_ui_elements(self, image_base64: str) -> Dict[str, Any]:
        """Detect and classify UI elements in the screen.
        
        Args:
            image_base64: Base64 encoded image data
            
        Returns:
            UI elements detection results
        """
        prompt = """
        You are a UI/UX Analysis Expert specializing in interface element detection.
        
        Analyze this screenshot and identify all UI elements:
        
        1. INTERACTIVE ELEMENTS:
           - Buttons (primary, secondary, action buttons)
           - Links and clickable text
           - Input fields and forms
           - Dropdown menus and selectors
           - Checkboxes and radio buttons
        
        2. NAVIGATION ELEMENTS:
           - Menu bars and navigation panels
           - Breadcrumbs and page indicators
           - Tabs and accordion sections
           - Search bars and filters
        
        3. CONTENT ELEMENTS:
           - Headers and titles
           - Text blocks and paragraphs
           - Images and media content
           - Tables and data grids
           - Cards and containers
        
        4. FEEDBACK ELEMENTS:
           - Progress bars and loading indicators
           - Notifications and alerts
           - Tooltips and help text
           - Status indicators
        
        Provide precise locations and descriptions for monitoring and automation purposes.
        """
        
        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=1500,
            store_in_memory=True,
            tags=["ui_elements", "interface_analysis"]
        )
    
    async def assess_system_performance(self, image_base64: str) -> Dict[str, Any]:
        """Assess system performance indicators visible on screen.
        
        Args:
            image_base64: Base64 encoded image data
            
        Returns:
            Performance assessment results
        """
        prompt = """
        You are a System Performance Monitoring Expert. Analyze this screenshot for performance indicators.
        
        Focus on identifying:
        
        1. PERFORMANCE METRICS:
           - CPU, memory, disk usage indicators
           - Network activity and bandwidth usage
           - Response times and latency metrics
           - Throughput and processing rates
        
        2. SYSTEM STATUS:
           - Application responsiveness
           - Loading states and progress
           - Error conditions and warnings
           - Resource availability
        
        3. MONITORING DASHBOARDS:
           - Charts, graphs, and visualizations
           - Real-time data displays
           - Alert panels and notifications
           - Trend indicators
        
        4. HEALTH INDICATORS:
           - Green/yellow/red status lights
           - Uptime and availability metrics
           - Service status indicators
           - Connection states
        
        Provide specific observations about system health and performance for monitoring purposes.
        """
        
        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=1500,
            store_in_memory=True,
            tags=["performance", "system_monitoring"]
        )
    
    async def detect_anomalies(self, image_base64: str, baseline: str = "") -> Dict[str, Any]:
        """Detect visual anomalies and unusual patterns in the screen.
        
        Args:
            image_base64: Base64 encoded image data
            baseline: Optional description of normal state
            
        Returns:
            Anomaly detection results
        """
        baseline_context = f"\nBASELINE REFERENCE: {baseline}" if baseline else ""
        
        prompt = f"""
        You are an Anomaly Detection Expert specializing in visual system monitoring.
        
        Analyze this screenshot for anomalies, irregularities, and unusual patterns:{baseline_context}
        
        Look for:
        
        1. VISUAL ANOMALIES:
           - Unexpected UI elements or layouts
           - Distorted or corrupted displays
           - Missing or misplaced components
           - Unusual color patterns or artifacts
        
        2. FUNCTIONAL ANOMALIES:
           - Error messages and warnings
           - Frozen or unresponsive interfaces
           - Unexpected application states
           - Performance degradation indicators
        
        3. SECURITY CONCERNS:
           - Suspicious pop-ups or dialogs
           - Unauthorized access attempts
           - Unusual network activity indicators
           - Security warnings or alerts
        
        4. SYSTEM IRREGULARITIES:
           - Resource usage spikes
           - Unexpected process behavior
           - Configuration changes
           - Service disruptions
        
        Rate the severity of any detected anomalies (LOW/MEDIUM/HIGH/CRITICAL) and provide specific recommendations.
        """
        
        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=1500,
            store_in_memory=True,
            tags=["anomaly_detection", "security_monitoring"]
        )
    
    async def generate_monitoring_report(self, image_base64: str, context: str = "") -> Dict[str, Any]:
        """Generate comprehensive monitoring report from screen analysis.
        
        Args:
            image_base64: Base64 encoded image data
            context: Additional context for the report
            
        Returns:
            Comprehensive monitoring report
        """
        context_info = f"\nCONTEXT: {context}" if context else ""
        
        prompt = f"""
        You are a Senior System Monitoring Analyst. Generate a comprehensive monitoring report from this screenshot.{context_info}
        
        Structure your report as follows:
        
        ## EXECUTIVE SUMMARY
        - Overall system status
        - Key findings and observations
        - Critical issues requiring attention
        
        ## DETAILED ANALYSIS
        
        ### System State
        - Application status and responsiveness
        - Resource utilization
        - Performance indicators
        
        ### User Interface
        - UI element functionality
        - Layout and accessibility
        - User experience factors
        
        ### Security & Compliance
        - Security status indicators
        - Access control elements
        - Compliance-related observations
        
        ### Performance Metrics
        - Response times and latency
        - Throughput and capacity
        - Resource efficiency
        
        ## RECOMMENDATIONS
        - Immediate actions required
        - Optimization opportunities
        - Preventive measures
        - Next monitoring steps
        
        ## RISK ASSESSMENT
        - Identified risks and their severity
        - Mitigation strategies
        - Monitoring priorities
        
        Provide actionable insights suitable for technical teams and management.
        """
        
        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=2000,
            store_in_memory=True,
            tags=["monitoring_report", "comprehensive_analysis"]
        )
    
    async def extract_text(self, image_base64: str) -> Dict[str, Any]:
        """Extract text from screen with monitoring-focused analysis.
        
        Args:
            image_base64: Base64 encoded image data
            
        Returns:
            Extracted text and metadata with monitoring context
        """
        prompt = """
        You are a Screen Monitor Analysis Expert specializing in text extraction and analysis.
        
        Extract and analyze all visible text from this screen capture:
        
        1. TEXT CONTENT:
           - All readable text (exact transcription)
           - Error messages and alerts
           - Status indicators and labels
           - Menu items and button text
        
        2. TEXT ORGANIZATION:
           - Hierarchical structure (headings, subheadings)
           - Lists, tables, and data structures
           - Navigation elements
           - Form fields and input areas
        
        3. MONITORING RELEVANCE:
           - System status messages
           - Performance metrics (if visible)
           - Warning or error indicators
           - Version numbers and timestamps
        
        4. CONTEXTUAL ANALYSIS:
           - Application or system being monitored
           - User interface state
           - Potential issues indicated by text
           - Critical information requiring attention
        
        Prioritize accuracy and focus on text that provides monitoring insights.
        """
        
        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=1500,
            store_in_memory=True,
            tags=["text_extraction", "ocr_analysis"]
        )
    
    async def analyze_screen_for_task(self, image_base64: str, task: str) -> Dict[str, Any]:
        """Analyze screen for a specific monitoring task with expert-level precision.
        
        Args:
            image_base64: Base64 encoded image data
            task: Task description
            
        Returns:
            Task-specific analysis with detailed monitoring insights
        """
        prompt = f"""
        You are a specialized Screen Monitor Analysis Expert. Analyze this screenshot with precision and expertise.
        
        TASK: {task}
        
        Provide a comprehensive analysis including:
        
        1. VISUAL ELEMENTS DETECTED:
           - UI components, buttons, menus, dialogs
           - Text content and readability
           - Icons, images, and graphical elements
           - Layout structure and organization
        
        2. SYSTEM STATE ASSESSMENT:
           - Application status and responsiveness
           - Error messages or warnings
           - Loading states or progress indicators
           - Resource usage indicators (if visible)
        
        3. MONITORING INSIGHTS:
           - Performance indicators
           - User interaction opportunities
           - Potential issues or anomalies
           - Security-related observations
        
        4. ACTIONABLE RECOMMENDATIONS:
           - Immediate actions required
           - Optimization opportunities
           - Risk mitigation steps
           - Next monitoring checkpoints
        
        Focus on technical accuracy and provide specific, actionable insights for system monitoring purposes.
        """
        
        return await self.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=1500,
            store_in_memory=True,
            tags=["task_analysis", "monitoring_task"]
        )


# Global AI service instance
ai_service = AIService()