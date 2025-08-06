# ScreenMonitorMCP v2

[![Version](https://img.shields.io/badge/version-2.0.7-blue.svg)](https://github.com/inkbytefo/ScreenMonitorMCP/releases/tag/v2.0.7)
[![PyPI](https://img.shields.io/pypi/v/screenmonitormcp-v2.svg)](https://pypi.org/project/screenmonitormcp-v2/)
[![Python](https://img.shields.io/pypi/pyversions/screenmonitormcp-v2.svg)](https://pypi.org/project/screenmonitormcp-v2/)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/a2dbda0f-f46d-40e1-9c13-0b47eff9df3a)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/inkbytefo-screenmonitormcp-badge.png)](https://mseep.ai/app/inkbytefo-screenmonitormcp)
A powerful Model Context Protocol (MCP) server that gives AI real-time vision capabilities and enhanced UI intelligence. Transform your AI assistant into a visual powerhouse that can see, analyze, and interact with your screen content.

## What is ScreenMonitorMCP?

ScreenMonitorMCP v2 is a revolutionary MCP server that bridges the gap between AI and visual computing. It enables AI assistants to capture screenshots, analyze screen content, and provide intelligent insights about what's happening on your display.

## Key Features

- **Real-time Screen Capture**: Instant screenshot capabilities across multiple monitors
- **AI-Powered Analysis**: Advanced screen content analysis using state-of-the-art vision models
- **Streaming Support**: Live screen streaming for continuous monitoring
- **Performance Monitoring**: Built-in system health and performance metrics
- **Multi-Platform**: Works seamlessly on Windows, macOS, and Linux
- **Easy Integration**: Simple setup with Claude Desktop and other MCP clients

## Quick Start

### Installation

```bash
# Install from PyPI
pip install screenmonitormcp

# Or install from source
git clone https://github.com/inkbytefo/screenmonitormcp.git
cd screenmonitormcp
pip install -e .
```

### Configuration

1. Create a `.env` file with your AI service credentials:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
```

2. Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "screenmonitormcp-v2": {
      "command": "python",
      "args": ["-m", "screenmonitormcp_v2.mcp_main"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENAI_MODEL": "qwen/qwen2.5-vl-32b-instruct:free"
      }
    }
  }
}
```

3. Restart Claude Desktop and start capturing!

## Available Tools

- `capture_screen` - Take screenshots of any monitor
- `analyze_screen` - AI-powered screen content analysis
- `analyze_image` - Analyze any image with AI vision
- `create_stream` - Start live screen streaming
- `get_performance_metrics` - System health monitoring

## Use Cases

- **UI/UX Analysis**: Get AI insights on interface design and usability
- **Debugging Assistance**: Visual debugging with AI-powered error detection
- **Content Creation**: Automated screenshot documentation and analysis
- **Accessibility Testing**: Screen reader and accessibility compliance checking
- **System Monitoring**: Visual system health and performance tracking

## Documentation

For detailed setup instructions and advanced configuration, see our [MCP Setup Guide](MCP_SETUP_GUIDE.md).

## Requirements

- Python 3.8+
- OpenAI API key (or compatible service)
- MCP-compatible client (Claude Desktop, etc.)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Previous Version

Looking for v1? Check the [v1 branch](https://github.com/inkbytefo/ScreenMonitorMCP/tree/v1) for the previous version.

---

**Built with ❤️ by [inkbytefo](https://github.com/inkbytefo)**
