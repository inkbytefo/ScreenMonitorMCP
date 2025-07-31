#!/usr/bin/env python3
"""
Configuration Module for ScreenMonitorMCP v2

This module handles configuration management for the MCP server.
It loads settings from environment variables and provides defaults.

Author: ScreenMonitorMCP Team
Version: 2.0.0
License: MIT
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration management for ScreenMonitorMCP v2."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4-vision-preview")
        
        # Server Configuration
        self.server_host = os.getenv("SERVER_HOST", "localhost")
        self.server_port = int(os.getenv("SERVER_PORT", "8000"))
        
        # Screen Capture Configuration
        self.default_image_format = os.getenv("DEFAULT_IMAGE_FORMAT", "png")
        self.default_image_quality = int(os.getenv("DEFAULT_IMAGE_QUALITY", "85"))
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # MCP Configuration
        self.mcp_server_name = "screenmonitormcp-v2"
        self.mcp_server_version = "2.0.0"
        self.mcp_protocol_version = "2025-06-18"
    
    def validate(self) -> bool:
        """Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not set. AI analysis will not work.", file=sys.stderr)
            return False
        
        return True
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration.
        
        Returns:
            Dictionary with OpenAI settings
        """
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model
        }
    
    def get_server_config(self) -> dict:
        """Get server configuration.
        
        Returns:
            Dictionary with server settings
        """
        return {
            "host": self.server_host,
            "port": self.server_port
        }
    
    def get_mcp_info(self) -> dict:
        """Get MCP server information.
        
        Returns:
            Dictionary with MCP server info
        """
        return {
            "name": self.mcp_server_name,
            "version": self.mcp_server_version,
            "protocol_version": self.mcp_protocol_version
        }