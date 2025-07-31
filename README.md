# ScreenMonitorMCP v2

🚀 **A REVOLUTIONARY Model Context Protocol (MCP) server!** Gives AI real-time vision capabilities and enhanced UI intelligence power. This isn't just screen capture - it gives AI the power to truly "see" and understand your digital world!

## 🌟 Features

### Core Capabilities
- **Real-time Screen Capture**: High-performance screen monitoring with multi-monitor support
- **AI Vision Integration**: Advanced computer vision with OCR, object detection, and scene analysis
- **Streaming Support**: Live screen streaming with configurable quality and frame rates
- **Performance Monitoring**: Real-time system metrics and health monitoring
- **Multi-AI Provider Support**: Compatible with OpenAI, Anthropic, and other AI services

### Enhanced v2 Features
- **Improved Performance**: Optimized capture algorithms and reduced latency
- **Better AI Integration**: Enhanced vision capabilities with multiple AI models
- **Advanced Streaming**: Real-time screen streaming with WebSocket support
- **Comprehensive Monitoring**: Detailed performance metrics and system health
- **Modular Architecture**: Clean, maintainable codebase with proper separation of concerns

## 🚀 Quick Start

### Installation

```bash
pip install screenmonitormcp-v2
```

### Basic Usage

```python
from screenmonitormcp_v2 import ScreenMonitorMCP

# Initialize the MCP server
server = ScreenMonitorMCP()

# Start the server
server.run()
```

### MCP Client Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "screenmonitormcp-v2": {
      "command": "screenmonitormcp-v2-mcp",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## 📖 Documentation

For detailed setup instructions, see [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md)

## 🛠️ Development

### Requirements
- Python 3.8+
- Windows/macOS/Linux
- OpenAI API key (optional, for AI features)

### Installation from Source

```bash
git clone https://github.com/inkbytefo/ScreenMonitorMCP.git
cd ScreenMonitorMCP
pip install -e .
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

Developed by **inkbytefo**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📊 Version History

### v2.0.0
- Complete rewrite with improved architecture
- Enhanced AI vision capabilities
- Real-time streaming support
- Better performance monitoring
- Modular design

### v1.x
- Initial release (see [v1 branch](https://github.com/inkbytefo/ScreenMonitorMCP/tree/v1))
