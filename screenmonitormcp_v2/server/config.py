"""Configuration management for ScreenMonitorMCP v2."""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Server configuration settings."""
    
    # Server settings
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, description="Server port")
    workers: int = Field(1, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload")
    
    # Security settings
    api_key: Optional[str] = Field(None, description="API key for authentication")
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    cors_credentials: bool = Field(True, description="Allow CORS credentials")
    
    # Streaming settings - Optimized for client stability
    max_stream_fps: int = Field(10, description="Maximum streaming FPS - Reduced for client stability")
    default_stream_fps: int = Field(1, description="Default streaming FPS - Conservative default")
    max_stream_quality: int = Field(75, description="Maximum streaming quality - Reduced for bandwidth")
    default_stream_quality: int = Field(60, description="Default streaming quality - Optimized for performance")
    max_concurrent_streams: int = Field(25, description="Maximum concurrent streams - Reduced for stability")
    stream_buffer_size: int = Field(10, description="Stream buffer size - Reduced to prevent memory issues")
    
    # Connection settings - Enhanced for stability
    max_connections: int = Field(250, description="Maximum concurrent connections - Reduced for stability")
    connection_timeout: int = Field(20, description="Connection timeout in seconds - Faster cleanup")
    keep_alive_timeout: int = Field(3, description="Keep-alive timeout in seconds - Faster detection")
    client_buffer_limit: int = Field(5 * 1024 * 1024, description="Client buffer limit in bytes (5MB)")
    max_frame_size: int = Field(2 * 1024 * 1024, description="Maximum frame size in bytes (2MB)")
    
    # Performance settings
    request_timeout: int = Field(30, description="Request timeout in seconds")
    max_request_size: int = Field(100 * 1024 * 1024, description="Maximum request size in bytes")
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute")
    
    # Logging settings
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "json",
        description="Log format: json, plain"
    )
    
    # AI Provider settings
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(None, description="OpenAI API base URL (for compatible APIs)")
    openai_model: str = Field("gpt-4o", description="OpenAI model name - Optimized for monitor analysis")
    openai_timeout: int = Field(30, description="OpenAI timeout in seconds")
    
    # Cache settings
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(1000, description="Maximum cache entries")
    
    # Monitoring settings
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(9090, description="Metrics server port")
    health_check_interval: int = Field(30, description="Health check interval in seconds")
    
    # MCP Server settings (from core/config.py)
    mcp_server_name: str = Field("screenmonitormcp-v2", description="MCP server name")
    mcp_server_version: str = Field("2.0.0", description="MCP server version")
    mcp_protocol_version: str = Field("2025-06-18", description="MCP protocol version")
    
    # Screen capture settings (from core/config.py)
    default_image_format: str = Field("png", description="Default image format for screen captures")
    default_image_quality: int = Field(85, description="Default image quality for screen captures")
    
    class Config:
        """Pydantic configuration."""
        
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


# Global configuration instance
config = ServerConfig()
