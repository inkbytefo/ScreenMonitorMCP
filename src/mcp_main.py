#!/usr/bin/env python3
"""
ScreenMonitorMCP v2 - MCP Server Entry Point

This script runs the MCP server in stdio mode for integration with MCP clients.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# CRITICAL: Disable all logging to stdout for MCP protocol
# MCP requires clean JSON-RPC communication over stdout
logging.basicConfig(
    level=logging.CRITICAL,  # Only critical errors
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr,  # All logs go to stderr
    force=True
)

# Suppress ALL loggers that might interfere with stdout
for logger_name in ["openai", "httpx", "uvicorn", "fastapi", "structlog", "mss", "PIL"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).disabled = True

# Disable structlog completely for MCP mode
import structlog
structlog.configure(
    processors=[],
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

from .core.mcp_server import run_mcp_server

def main():
    """Main entry point for MCP server."""
    # Set up environment for MCP
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    
    # Ensure stdout is line buffered for JSON-RPC
    if sys.stdout.isatty():
        sys.stdout.reconfigure(line_buffering=True)
    
    # Run the MCP server
    try:
        run_mcp_server()
    except KeyboardInterrupt:
        print("MCP Server stopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"MCP Server error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()