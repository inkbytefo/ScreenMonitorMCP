# ScreenMonitorMCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A Model Context Protocol (MCP) server that provides AI assistants with intelligent screen monitoring, UI interaction, and real-time visual analysis capabilities.

## Features

### 🧠 Smart Monitoring System
- **Intelligent Triggers**: Event-driven monitoring with 6 smart triggers (significant_change, error_detected, new_window, application_change, code_change, text_appears)
- **Adaptive Performance**: Automatically adjusts FPS based on screen activity
- **Event Classification**: Intelligent categorization of screen events

### 🎯 UI Intelligence & Interaction
- **Smart Click**: Natural language UI interaction ("Click the save button")
- **OCR Text Extraction**: Extract text from screen regions with coordinates
- **Element Detection**: Identify and interact with UI elements
- **75% Success Rate**: Enhanced fuzzy matching and position-based scoring

### 📊 Application Monitoring
- **Context Awareness**: Track active applications and window changes
- **Event Broadcasting**: Real-time application event notifications
- **Multi-Application Support**: Works with any desktop application

### 🎬 Video Recording & Analysis
- **Screen Recording**: Capture screen activity with configurable FPS
- **AI Analysis**: Multiple analysis types (summary, frame-by-frame, key moments)
- **Format Support**: Save recordings in various formats

## Available Tools

### Smart Monitoring
- `start_smart_monitoring()` - Begin intelligent trigger-based monitoring
- `stop_smart_monitoring()` - Stop smart monitoring
- `get_monitoring_insights()` - Get AI-powered analysis insights
- `get_recent_events()` - Retrieve recent smart events with details
- `get_monitoring_summary()` - Get comprehensive monitoring session report

### UI Intelligence
- `smart_click()` - Natural language UI interaction ("Click the save button")
- `extract_text_from_screen()` - OCR text extraction with coordinates
- `analyze_ui_elements()` - Detect and analyze UI elements

### Core Functionality
- `capture_and_analyze()` - Screen capture with AI analysis
- `record_and_analyze()` - Video recording with AI analysis
- `get_active_application()` - Get current application context
- `list_tools()` - List all available tools

## Quick Start

### Installation

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/inkbytefo/ScreenMonitorMCP.git
   cd ScreenMonitorMCP
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the server**
   ```bash
   python main.py
   ```

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": ["/path/to/ScreenMonitorMCP/main.py"],
      "cwd": "/path/to/ScreenMonitorMCP"
    }
  }
}
```

### Usage Examples

**Smart Monitoring**
```python
# Start intelligent monitoring
await start_smart_monitoring(
    triggers=["significant_change", "error_detected", "new_window"],
    analysis_prompt="What changed on screen and why is it important?",
    fps=2,
    sensitivity="medium"
)

# Get insights
insights = await get_monitoring_insights()
```

**UI Interaction**
```python
# Natural language clicking
await smart_click("Save button")
await smart_click("File menu")

# Extract text from regions
text_data = await extract_text_from_screen(
    region={"x": 100, "y": 100, "width": 300, "height": 200}
)
```

**Video Analysis**
```python
# Record and analyze screen activity
video_result = await record_and_analyze(
    duration=15,
    fps=2,
    analysis_type="summary",
    analysis_prompt="What happened in this recording?",
    save_video=True
)
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Server Configuration
HOST=127.0.0.1
PORT=7777
API_KEY=your_secret_key

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_OPENAI_MODEL=gpt-4o
DEFAULT_MAX_TOKENS=1000
```

### Advanced MCP Client Configurations

**With Environment Variables**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": ["/path/to/ScreenMonitorMCP/main.py"],
      "cwd": "/path/to/ScreenMonitorMCP",
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

**With API Key Security**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": [
        "/path/to/ScreenMonitorMCP/main.py",
        "--api-key", "your-secret-key"
      ],
      "cwd": "/path/to/ScreenMonitorMCP"
    }
  }
}
```

**Windows Configuration**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "C:/Python311/python.exe",
      "args": ["C:/path/to/ScreenMonitorMCP/main.py"],
      "cwd": "C:/path/to/ScreenMonitorMCP"
    }
  }
}
```

## Use Cases

### Real-time Application Monitoring
Monitor application changes and events in real-time, perfect for development workflows and debugging.

### AI-Powered Screen Analysis
Analyze screen content with AI to understand context, detect errors, and provide intelligent insights.

### Natural UI Interaction
Enable AI assistants to interact with desktop applications using natural language commands.

### Automated Testing & QA
Record and analyze user interactions for automated testing and quality assurance workflows.

## Requirements

- Python 3.9+
- OpenAI API key (for AI analysis)
- Windows/macOS/Linux support

## Dependencies

Key dependencies include:
- `fastmcp` - MCP server framework
- `pillow` - Image processing
- `easyocr` - Text extraction
- `opencv-python` - Video recording
- `openai` - AI analysis
- `psutil` - System monitoring

## Performance

- **Smart Triggers**: Only analyzes when meaningful events occur
- **Adaptive FPS**: Automatically adjusts monitoring speed (1-5 FPS)
- **75% Success Rate**: Enhanced UI element detection and interaction
- **Memory Efficient**: Event-driven architecture minimizes resource usage

## Troubleshooting

### Common Issues

**Unicode/Encoding Error (Windows)**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Solution:** Fixed automatically - server uses UTF-8 encoding.

**JSON Configuration Error**
```json
// ❌ Wrong - trailing comma
{
  "command": "python",
  "args": ["path/to/main.py",]
}

// ✅ Correct
{
  "command": "python",
  "args": ["path/to/main.py"]
}
```

**Python Path Issue**
Use full Python path if needed:
```json
{
  "command": "C:/Python311/python.exe",
  "args": ["C:/path/to/ScreenMonitorMCP/main.py"]
}
```

**Missing Dependencies**
```bash
cd ScreenMonitorMCP
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**ScreenMonitorMCP** - Giving AI assistants intelligent vision and interaction capabilities through the Model Context Protocol.
