"""Main entry point for ScreenMonitorMCP v2."""

import argparse
import asyncio
import sys
from pathlib import Path

from .server.app import app
from .server.config import config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ScreenMonitorMCP v2 - Streamable HTTP/SSE MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  screenmonitormcp-v2                    # Start server on default port 8000
  screenmonitormcp-v2 --host 0.0.0.0    # Bind to all interfaces
  screenmonitormcp-v2 --port 8080       # Use custom port
  screenmonitormcp-v2 --reload          # Enable auto-reload for development

API Endpoints:
  GET  /health                          # Health check
  POST /api/v2/capture                  # Capture screen
  POST /api/v2/streams                  # Create stream
  GET  /api/v2/streams/{id}/sse         # SSE streaming
  GET  /api/v2/streams/{id}             # Stream info
  DELETE /api/v2/streams/{id}           # Stop stream
  GET  /docs                            # Interactive API documentation
        """
    )

    parser.add_argument(
        "--host",
        default=config.host,
        help=f"Host to bind to (default: {config.host})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.port,
        help=f"Port to bind to (default: {config.port})"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=config.log_level.lower(),
        help=f"Log level (default: {config.log_level.lower()})"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="ScreenMonitorMCP v2.0.0"
    )

    args = parser.parse_args()

    # Update config with CLI arguments
    config.host = args.host
    config.port = args.port
    config.reload = args.reload
    config.log_level = args.log_level.upper()

    # Print banner
    print("\n" + "="*60)
    print("ScreenMonitorMCP v2 - Streamable HTTP/SSE MCP Server")
    print("   Modern streaming architecture with HTTP/SSE support")
    print("="*60)
    print()
    print(f"Starting server on http://{args.host}:{args.port}")
    print("API Documentation: http://{}:{}/docs".format(args.host, args.port))
    print("WebSocket: ws://{}:{}/ws".format(args.host, args.port))
    print("Health Check: http://{}:{}/health".format(args.host, args.port))
    print()

    # Import uvicorn here to avoid import issues
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install uvicorn[standard]")
        sys.exit(1)

    # Run the server
    uvicorn.run(
        "screenmonitormcp_v2.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()
