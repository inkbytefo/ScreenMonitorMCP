"""CLI interface for ScreenMonitorMCP v2."""

import asyncio
import json
import sys
from typing import Optional

import aiohttp
import click
from datetime import datetime

from .server.config import config


@click.group()
@click.version_option(version="2.0.0", prog_name="ScreenMonitorMCP v2")
def cli():
    """ScreenMonitorMCP v2 CLI - Streamable HTTP/SSE MCP Server."""
    pass


@cli.command()
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--log-level", default="info", type=click.Choice(["debug", "info", "warning", "error"]))
def serve(host, port, reload, log_level):
    """Start the ScreenMonitorMCP v2 server."""
    from .__main__ import main
    
    # Override config
    config.host = host
    config.port = port
    config.reload = reload
    config.log_level = log_level.upper()
    
    main()


@cli.command()
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
@click.option("--monitor", default=0, type=int, help="Monitor number")
@click.option("--quality", default=85, type=int, help="Image quality (1-100)")
@click.option("--format", default="jpeg", type=click.Choice(["jpeg", "png", "webp"]))
def capture(host, port, monitor, quality, format):
    """Capture a screenshot."""
    async def _capture():
        url = f"http://{host}:{port}/api/v2/capture"
        
        payload = {
            "monitor_number": monitor,
            "quality": quality,
            "output_format": format
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(f"Error: {response.status}", err=True)
                    sys.exit(1)
    
    asyncio.run(_capture())


@cli.command()
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
@click.option("--fps", default=10, type=int, help="Frames per second")
@click.option("--quality", default=75, type=int, help="Stream quality (1-100)")
@click.option("--format", default="jpeg", type=click.Choice(["jpeg", "png", "webp"]))
def create_stream(host, port, fps, quality, format):
    """Create a new screen stream."""
    async def _create():
        url = f"http://{host}:{port}/api/v2/streams"
        
        payload = {
            "stream_type": "screen",
            "fps": fps,
            "quality": quality,
            "format": format
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(f"Error: {response.status}", err=True)
                    sys.exit(1)
    
    asyncio.run(_create())


@cli.command()
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
def status(host, port):
    """Get server status."""
    async def _status():
        url = f"http://{host}:{port}/health"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(f"Error: {response.status}", err=True)
                    sys.exit(1)
    
    asyncio.run(_status())


@cli.command()
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
def list_streams(host, port):
    """List active streams."""
    async def _list():
        url = f"http://{host}:{port}/api/v2/streams"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(f"Error: {response.status}", err=True)
                    sys.exit(1)
    
    asyncio.run(_list())


@cli.command()
@click.argument("stream_id")
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
def stop_stream(stream_id, host, port):
    """Stop a stream."""
    async def _stop():
        url = f"http://{host}:{port}/api/v2/streams/{stream_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url) as response:
                if response.status == 200:
                    data = await response.json()
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(f"Error: {response.status}", err=True)
                    sys.exit(1)
    
    asyncio.run(_stop())


@cli.command()
@click.option("--host", default=config.host, help="Server host")
@click.option("--port", default=config.port, type=int, help="Server port")
def connections(host, port):
    """List active connections."""
    async def _connections():
        url = f"http://{host}:{port}/api/v2/connections"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(f"Error: {response.status}", err=True)
                    sys.exit(1)
    
    asyncio.run(_connections())


if __name__ == "__main__":
    cli()
