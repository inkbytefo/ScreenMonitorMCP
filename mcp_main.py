#!/usr/bin/env python3
"""
ScreenMonitorMCP v2 - MCP Server Entry Point

This is the main entry point for the ScreenMonitorMCP v2 MCP server.
It runs using the official MCP Python SDK with stdio transport.

Usage:
    python mcp_main.py

The server will:
1. Initialize the MCP server using official SDK
2. Listen for JSON-RPC requests on stdin
3. Send responses on stdout
4. Log errors to stderr

Author: ScreenMonitorMCP Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import asyncio
import logging

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure logging for MCP mode - minimal logging to stderr only
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(message)s',
    stream=sys.stderr
)

# Ensure no stdout logging interferes with MCP protocol
for handler in logging.root.handlers[:]:
    if hasattr(handler, 'stream') and handler.stream == sys.stdout:
        logging.root.removeHandler(handler)

# Disable all loggers that might interfere with stdio
logging.getLogger('mss').setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.getLogger('openai').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)


def main():
    """Main entry point for MCP server."""
    # Set environment variables for optimal MCP operation
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Import and run the MCP server
        from src.core.mcp_server import run_mcp_server
        
        # Run the async MCP server
        asyncio.run(run_mcp_server())
        
    except KeyboardInterrupt:
        print("\nMCP Server interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()