#!/usr/bin/env python3
"""
AI Monitor Analysis Expert for ScreenMonitorMCP v2

This module provides specialized AI-powered screen monitoring and analysis using OpenAI's vision models.
Designed specifically for computer screen analysis, UI element detection, and system monitoring tasks.

Author: inkbytefo
Version: 2.1.0
License: MIT
"""

import asyncio
import base64
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import openai
from .config import Config

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Specialized AI Monitor Analysis Expert for screen monitoring and computer vision tasks.
    
    This class is optimized for:
    - Screen content analysis and monitoring
    - UI element detection and classification
    - System state assessment
    - Application interface analysis
    - Performance monitoring visualization
    """
    
    def __init__(self):
        """Initialize the AI analyzer."""
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client only if API key is available
        if self.config.openai_api_key:
            self.client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)
        else:
            self.client = None
            self.logger.warning("OpenAI API key not found. AI analysis features will be disabled.")
    
    async def analyze_image(self, image_data: bytes, prompt: str, 
                          detail_level: str = "high") -> Dict[str, Any]:
        """Analyze an image using OpenAI's vision model.
        
        Args:
            image_data: Image data as bytes
            prompt: Analysis prompt
            detail_level: Level of detail (low, high)
            
        Returns:
            Analysis result dictionary
        """
        # Check if OpenAI client is available
        if not self.client:
            return {
                "error": "OpenAI API key not configured. AI analysis is disabled.",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
        try:
            # Convert image data to base64
            if isinstance(image_data, bytes):
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            else:
                image_base64 = image_data
            
            # Prepare the message
            messages = [
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
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": detail_level
                            }
                        }
                    ]
                }
            ]
            
            # Call OpenAI API with optimized token limits for screen analysis
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                max_tokens=1500,  # Increased for detailed screen analysis
                temperature=0.1   # Low temperature for consistent monitoring results
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "model": self.config.openai_model,
                "detail_level": detail_level
            }
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_screen_for_task(self, image_data: bytes, task: str) -> Dict[str, Any]:
        """Analyze screen for a specific monitoring task with expert-level precision.
        
        Args:
            image_data: Screenshot data
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
        
        return await self.analyze_image(image_data, prompt, "high")
    
    async def detect_ui_elements(self, image_data: bytes) -> Dict[str, Any]:
        """Detect and classify UI elements in the screen.
        
        Args:
            image_data: Screenshot data as bytes
            
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
        
        return await self.analyze_image(image_data, prompt, "high")
    
    async def assess_system_performance(self, image_data: bytes) -> Dict[str, Any]:
        """Assess system performance indicators visible on screen.
        
        Args:
            image_data: Screenshot data as bytes
            
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
        
        return await self.analyze_image(image_data, prompt, "high")
    
    async def detect_anomalies(self, image_data: bytes, baseline_description: str = "") -> Dict[str, Any]:
        """Detect visual anomalies and unusual patterns in the screen.
        
        Args:
            image_data: Screenshot data as bytes
            baseline_description: Optional description of normal state
            
        Returns:
            Anomaly detection results
        """
        baseline_context = f"\nBASELINE REFERENCE: {baseline_description}" if baseline_description else ""
        
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
        
        return await self.analyze_image(image_data, prompt, "high")
    
    async def generate_monitoring_report(self, image_data: bytes, context: str = "") -> Dict[str, Any]:
        """Generate comprehensive monitoring report from screen analysis.
        
        Args:
            image_data: Screenshot data as bytes
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
        
        return await self.analyze_image(image_data, prompt, "high")
    
    async def analyze_screen(self, image_data: bytes, query: str, detail_level: str = "high") -> str:
        """Analyze screen content using specialized monitoring expertise.
        
        Args:
            image_data: Screenshot data as bytes
            query: What to analyze or look for
            detail_level: Level of detail for analysis
            
        Returns:
            Expert analysis result as text
        """
        # Enhance query with monitoring context
        enhanced_query = f"""
        You are a Screen Monitor Analysis Expert. Analyze this screenshot with professional expertise.
        
        USER QUERY: {query}
        
        Provide analysis as a monitoring specialist, focusing on:
        - Technical accuracy and precision
        - System state and performance indicators
        - UI/UX elements and their functionality
        - Potential issues or anomalies
        - Actionable insights for system monitoring
        
        Deliver concise, expert-level observations that would be valuable for system monitoring and analysis.
        """
        
        result = await self.analyze_image(image_data, enhanced_query, detail_level)
        
        if result.get("success"):
            return result.get("analysis", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    
    async def extract_text(self, image_data: bytes) -> Dict[str, Any]:
        """Extract text from screen with monitoring-focused analysis.
        
        Args:
            image_data: Image data as bytes
            
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
        
        return await self.analyze_image(image_data, prompt, "high")