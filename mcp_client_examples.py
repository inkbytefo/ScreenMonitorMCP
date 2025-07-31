#!/usr/bin/env python3
"""
MCP Client Connection Examples for ScreenMonitorMCP-v2
Author: inkbytefo

This file demonstrates how to connect to the ScreenMonitorMCP-v2 server
using different MCP client implementations.
"""

import asyncio
import json
from typing import Dict, Any, List

# Example 1: Direct MCP Client Connection
async def connect_direct_mcp():
    """
    Direct connection to MCP server using mcp library
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        # Server parameters
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "screenmonitormcp_v2.mcp_main"]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                tools_result = await session.list_tools()
                print(f"📋 Available tools: {len(tools_result.tools)}")
                
                for tool in tools_result.tools:
                    print(f"  🔧 {tool.name}: {tool.description}")
                
                # Example: Capture screen
                capture_result = await session.call_tool(
                    "capture_screen",
                    arguments={"monitor": 0, "format": "png"}
                )
                print(f"📸 Screen capture result: {capture_result}")
                
    except ImportError:
        print("❌ MCP library not installed. Install with: pip install mcp")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

# Example 2: JSON-RPC Client Connection
class MCPJSONRPCClient:
    """
    JSON-RPC based MCP client implementation
    """
    
    def __init__(self):
        self.request_id = 0
        
    def create_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 request"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            request["params"] = params
        return request
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send request to MCP server (placeholder implementation)"""
        request = self.create_request(method, params)
        print(f"📤 Sending request: {json.dumps(request, indent=2)}")
        
        # This is a placeholder - in real implementation, you would:
        # 1. Send the request via stdio/websocket/http
        # 2. Receive and parse the response
        # 3. Handle errors appropriately
        
        return {"jsonrpc": "2.0", "id": request["id"], "result": "placeholder"}

async def example_jsonrpc_client():
    """
    Example using JSON-RPC client
    """
    client = MCPJSONRPCClient()
    
    # Initialize connection
    init_response = await client.send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "example-client",
                "version": "1.0.0"
            }
        }
    )
    
    # List tools
    tools_response = await client.send_request("tools/list")
    
    # Call a tool
    capture_response = await client.send_request(
        "tools/call",
        {
            "name": "capture_screen",
            "arguments": {
                "monitor": 0,
                "format": "png"
            }
        }
    )

# Example 3: Claude Desktop Configuration
def generate_claude_desktop_config() -> Dict[str, Any]:
    """
    Generate Claude Desktop compatible configuration
    """
    config = {
        "mcpServers": {
            "screenmonitormcp-v2": {
                "command": "python",
                "args": [
                    "-m",
                    "screenmonitormcp_v2.mcp_main"
                ],
                "env": {
                    "OPENAI_API_KEY": "your-openai-api-key-here",
                    "ANTHROPIC_API_KEY": "your-anthropic-api-key-here",
                    "GOOGLE_API_KEY": "your-google-api-key-here"
                }
            }
        }
    }
    
    print("🔧 Claude Desktop Configuration:")
    print(json.dumps(config, indent=2))
    return config

# Example 4: Available Tools Reference
def list_available_tools() -> List[Dict[str, Any]]:
    """
    Reference list of all available tools in ScreenMonitorMCP-v2
    """
    tools = [
        {
            "name": "capture_screen",
            "description": "Capture a screenshot of the current screen",
            "parameters": {
                "monitor": "Monitor number to capture (0 for primary)",
                "region": "Specific region to capture (optional)",
                "format": "Image format (png/jpeg)"
            }
        },
        {
            "name": "analyze_screen",
            "description": "Analyze the current screen content using AI vision",
            "parameters": {
                "query": "What to analyze or look for in the screen",
                "monitor": "Monitor number to analyze (0 for primary)",
                "detail_level": "Level of detail for analysis (low/high)"
            }
        },
        {
            "name": "analyze_image",
            "description": "Analyze an image file using AI vision",
            "parameters": {
                "image_path": "Path to the image file to analyze",
                "query": "What to analyze or look for in the image",
                "detail_level": "Level of detail for analysis (low/high)"
            }
        },
        {
            "name": "chat_completion",
            "description": "Generate AI chat completion responses",
            "parameters": {
                "messages": "List of chat messages",
                "model": "AI model to use (optional)",
                "max_tokens": "Maximum tokens in response (optional)",
                "temperature": "Response creativity (0.0-1.0, optional)"
            }
        },
        {
            "name": "list_ai_models",
            "description": "List all available AI models",
            "parameters": {}
        },
        {
            "name": "get_ai_status",
            "description": "Get current AI service status and configuration",
            "parameters": {}
        },
        {
            "name": "get_performance_metrics",
            "description": "Get system performance metrics",
            "parameters": {
                "include_history": "Include historical data (optional)"
            }
        },
        {
            "name": "get_system_status",
            "description": "Get overall system status",
            "parameters": {}
        },
        {
            "name": "create_stream",
            "description": "Create a new video stream",
            "parameters": {
                "name": "Stream name",
                "monitor": "Monitor to stream (optional)",
                "quality": "Stream quality (optional)",
                "fps": "Frames per second (optional)"
            }
        },
        {
            "name": "list_streams",
            "description": "List all active streams",
            "parameters": {}
        },
        {
            "name": "get_stream_info",
            "description": "Get information about a specific stream",
            "parameters": {
                "stream_id": "ID of the stream to get info for"
            }
        },
        {
            "name": "stop_stream",
            "description": "Stop a running stream",
            "parameters": {
                "stream_id": "ID of the stream to stop"
            }
        }
    ]
    
    print("🔧 Available Tools:")
    for i, tool in enumerate(tools, 1):
        print(f"  {i:2d}. {tool['name']}: {tool['description']}")
    
    return tools

# Main execution
async def main():
    """
    Main function to run all examples
    """
    print("🚀 ScreenMonitorMCP-v2 Client Examples")
    print("=" * 50)
    
    print("\n1. Available Tools Reference:")
    list_available_tools()
    
    print("\n2. Claude Desktop Configuration:")
    generate_claude_desktop_config()
    
    print("\n3. JSON-RPC Client Example:")
    await example_jsonrpc_client()
    
    print("\n4. Direct MCP Client Example:")
    await connect_direct_mcp()
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())