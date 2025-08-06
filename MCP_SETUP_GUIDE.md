# ScreenMonitorMCP v2 - MCP Client Setup Guide

This guide will help you set up ScreenMonitorMCP v2 with various MCP clients including Claude Desktop.

## Installation

### 1. Install the Package

```bash
# Install from PyPI (when published)
pip install screenmonitormcp-v2

# Or install from source
pip install -e .
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```env
# AI Service Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# Alternative: Use OpenRouter
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# OPENAI_MODEL=qwen/qwen2.5-vl-32b-instruct:free

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
DEBUG=false
```

## Claude Desktop Setup

### 1. Locate Claude Desktop Config

Find your Claude Desktop configuration file:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### 2. Add MCP Server Configuration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "screenmonitormcp-v2": {
      "command": "python",
      "args": ["-m", "screenmonitormcp_v2.mcp_main"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "gpt-4o"
      }
    }
  }
}
```

### 3. Alternative: Using Installed Package

If you installed via pip, you can use:

```json
{
  "mcpServers": {
    "screenmonitormcp-v2": {
      "command": "screenmonitormcp-v2-mcp",
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "gpt-4o"
      }
    }
  }
}
```

### 4. Restart Claude Desktop

After updating the configuration, restart Claude Desktop to load the MCP server.

## Other MCP Clients

### Generic MCP Client

For other MCP clients, use the following command:

```bash
python -m screenmonitormcp_v2.mcp_main
```

Or if installed via pip:

```bash
screenmonitormcp-v2-mcp
```

### MCP Inspector (for testing)

```bash
npx @modelcontextprotocol/inspector python -m screenmonitormcp_v2.mcp_main
```

## Available Tools

Once configured, you'll have access to these tools:

### Available Tools

### 1. `analyze_screen`
- **Description**: Analyze the current screen content using AI vision
- **Parameters**:
  - `query`: What to analyze or look for in the screen
  - `monitor` (optional): Monitor number to analyze (0 for primary)
  - `detail_level` (optional): Level of detail for analysis (low or high, default: high)

### 2. `chat_completion`
- **Description**: Generate chat completion using AI models
- **Parameters**:
  - `messages`: Array of chat messages with role and content (required)
  - `model` (optional): AI model to use
  - `max_tokens` (optional): Maximum tokens for response (default: 1000)
  - `temperature` (optional): Temperature for response generation (default: 0.7)

### 3. `list_ai_models`
- **Description**: List available AI models from the configured provider
- **Parameters**: None

### 4. `get_ai_status`
- **Description**: Get AI service configuration status
- **Parameters**: None

### 5. `get_performance_metrics`
- **Description**: Get detailed performance metrics and system health
- **Parameters**: None

### 6. `get_system_status`
- **Description**: Get overall system status and health information
- **Parameters**: None

### 7. `create_stream`
- **Description**: Create a new screen streaming session
- **Parameters**:
  - `monitor` (optional): Monitor number to stream (0 for primary, default: 0)
  - `fps` (optional): Frames per second for streaming (default: 10)
  - `quality` (optional): Image quality 1-100 (default: 80)
  - `format` (optional): Image format (jpeg or png, default: jpeg)

### 8. `list_streams`
- **Description**: List all active streaming sessions
- **Parameters**: None

### 9. `get_stream_info`
- **Description**: Get information about a specific stream
- **Parameters**:
  - `stream_id`: Stream ID to get information for (required)

### 10. `stop_stream`
- **Description**: Stop a specific streaming session
- **Parameters**:
  - `stream_id`: Stream ID to stop (required)

### 11. `analyze_scene_from_memory`
- **Description**: Analyze scene based on stored memory data
- **Parameters**:
  - `query`: What to analyze or look for in the stored scenes (required)
  - `stream_id` (optional): Specific stream to analyze
  - `time_range_minutes` (optional): Time range to search in minutes (default: 30)
  - `limit` (optional): Maximum number of results to analyze (default: 10)

### 12. `query_memory`
- **Description**: Query the memory system for stored analysis data
- **Parameters**:
  - `query`: Search query for memory entries (required)
  - `stream_id` (optional): Filter by specific stream ID
  - `time_range_minutes` (optional): Time range to search in minutes (default: 60)
  - `limit` (optional): Maximum number of results (default: 20)

### 13. `get_memory_statistics`
- **Description**: Get memory system statistics and health information
- **Parameters**: None

## Usage Examples

### In Claude Desktop

Once configured, you can ask Claude:

- "Can you take a screenshot of my screen?"
- "What's currently displayed on my screen?"
- "Analyze my screen and tell me what applications are open"
- "Take a screenshot of the top-left corner of my screen"

### Testing the Setup

To test if everything is working:

1. Open Claude Desktop
2. Start a new conversation
3. Ask: "Can you take a screenshot?"
4. Claude should be able to capture and analyze your screen

## Troubleshooting

### Common Issues

1. **"Tool not found" error**:
   - Check that the MCP server is properly configured in `claude_desktop_config.json`
   - Restart Claude Desktop after configuration changes

2. **"Permission denied" error**:
   - Ensure Python has screen capture permissions on macOS
   - Run as administrator on Windows if needed

3. **"AI service error"**:
   - Check your API key is valid
   - Verify the base URL and model are correct
   - Check your internet connection

### Debug Mode

To enable debug logging, set `DEBUG=true` in your `.env` file or environment variables.

### Logs

Check the Claude Desktop logs for MCP server errors:
- **Windows**: `%APPDATA%\Claude\logs\`
- **macOS**: `~/Library/Logs/Claude/`
- **Linux**: `~/.local/share/Claude/logs/`

## Advanced Configuration

### Custom AI Models

You can use different AI models by updating the environment variables:

```json
{
  "mcpServers": {
    "screenmonitormcp-v2": {
      "command": "python",
      "args": ["-m", "screenmonitormcp_v2.mcp_main"],
      "env": {
        "OPENAI_API_KEY": "your-api-key",
        "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENAI_MODEL": "anthropic/claude-3.5-sonnet"
      }
    }
  }
}
```

### Multiple Monitors

The server supports multiple monitors. Use the `monitor` parameter to specify which monitor to capture (0 for primary, 1 for secondary, etc.).

### Region Capture

You can capture specific regions of the screen:

```json
{
  "region": {
    "x": 100,
    "y": 100,
    "width": 800,
    "height": 600
  }
}
```

## Support

For issues and support:
- GitHub Issues: https://github.com/inkbytefo/ScreenMonitorMCP/issues
- Documentation: https://github.com/inkbytefo/ScreenMonitorMCP#readme