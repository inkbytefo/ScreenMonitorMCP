#!/usr/bin/env python3
"""
AI Analyzer Module for ScreenMonitorMCP v2

This module provides AI-powered image analysis using OpenAI's vision models.
It can analyze screenshots and provide detailed descriptions.

Author: ScreenMonitorMCP Team
Version: 2.0.0
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
    """AI-powered image analysis using OpenAI vision models."""
    
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
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                max_tokens=500
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
        """Analyze screen for a specific task.
        
        Args:
            image_data: Screenshot data
            task: Task description
            
        Returns:
            Task-specific analysis
        """
        prompt = f"""
        Analyze this screenshot to help accomplish the following task: {task}
        
        Please provide:
        1. Relevant UI elements visible on screen
        2. Current state of the application/interface
        3. Specific steps or actions needed to accomplish the task
        4. Any obstacles or issues that might prevent task completion
        5. Recommendations for next steps
        
        Task: {task}
        """
        
        return await self.analyze_image(image_data, prompt, "high")
    
    async def analyze_screen(self, image_data: bytes, query: str, detail_level: str = "high") -> str:
        """Analyze screen content using AI vision.
        
        Args:
            image_data: Screenshot data as bytes
            query: What to analyze or look for
            detail_level: Level of detail for analysis
            
        Returns:
            Analysis result as text
        """
        result = await self.analyze_image(image_data, query, detail_level)
        
        if result.get("success"):
            return result.get("analysis", "No analysis available")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
    
    async def extract_text(self, image_data: bytes) -> Dict[str, Any]:
        """Extract text from an image.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Extracted text and metadata
        """
        prompt = """
        Extract all visible text from this image. Please provide:
        1. All readable text content
        2. Text organization (headings, paragraphs, lists, etc.)
        3. Any important formatting or styling
        4. Location/context of text elements
        
        Focus on accuracy and completeness of text extraction.
        """
        
        return await self.analyze_image(image_data, prompt, "high")