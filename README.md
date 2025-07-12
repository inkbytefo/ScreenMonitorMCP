# ScreenMonitorMCP - Revolutionary AI Vision Server

**Give AI real-time sight and screen interaction capabilities**

ScreenMonitorMCP is a revolutionary MCP (Model Context Protocol) server that provides Claude and other AI assistants with real-time screen monitoring, visual analysis, and intelligent interaction capabilities. This project enables AI to see, understand, and interact with your screen in ways never before possible.

## Why ScreenMonitorMCP?

Transform your AI assistant from text-only to a visual powerhouse that can:
- Monitor your screen in real-time and detect important changes
- Click UI elements using natural language commands
- Extract text from any part of your screen
- Analyze screenshots and videos with AI
- Provide intelligent insights about screen activity

## Core Features

### Smart Monitoring System
- **start_smart_monitoring()** - Enable intelligent monitoring with configurable triggers
- **get_monitoring_insights()** - AI-powered analysis of screen activity
- **get_recent_events()** - History of detected screen changes
- **stop_smart_monitoring()** - Stop monitoring with preserved insights

### Natural Language UI Interaction
- **smart_click()** - Click elements using descriptions like "Save button"
- **extract_text_from_screen()** - OCR text extraction from screen regions
- **get_active_application()** - Get current application context

### Visual Analysis Tools
- **capture_and_analyze()** - Screenshot capture with AI analysis
- **record_and_analyze()** - Video recording with AI analysis
- **query_vision_about_current_view()** - Ask AI questions about current screen

### System Performance
- **get_system_metrics()** - Comprehensive system health dashboard
- **get_cache_stats()** - Cache performance statistics
- **optimize_image()** - Advanced image optimization
- **simulate_input()** - Keyboard and mouse simulation

## Quick Setup

### 1. Installation
```bash
git clone https://github.com/inkbytefo/ScreenMonitorMCP.git
cd ScreenMonitorMCP
pip install -r requirements.txt
```

### 2. Configuration
```bash
cp .env.example .env
# Edit .env file with your OpenAI API key
```

### 3. Run Server
```bash
python main.py
```

### 4. Claude Desktop Integration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": [
        "/path/to/ScreenMonitorMCP/main.py"
      ]
    }
  }
}
```

## Usage Examples

```python
# Start intelligent monitoring
await start_smart_monitoring(triggers=['significant_change', 'error_detected'])

# Natural language UI interaction
await smart_click('Save button')
await smart_click('Email input field')

# Ask AI about current screen
await query_vision_about_current_view('What errors are visible on this page?')

# Extract text from screen
await extract_text_from_screen()
```

## Available Tools (21 Total)

**Smart Monitoring (6 tools)**: Real-time screen monitoring with AI analysis
**UI Interaction (2 tools)**: Natural language screen control
**Visual Analysis (3 tools)**: AI-powered image and video analysis
**System Performance (7 tools)**: Performance monitoring and optimization
**Input Simulation (2 tools)**: Keyboard and mouse automation
**Utility (1 tool)**: Tool documentation and listing

## Technical Features

- **21 Revolutionary Tools** - Comprehensive AI vision capabilities
- **Real-time Monitoring** - Adaptive FPS with smart triggers
- **Multi-AI Support** - OpenAI, OpenRouter, and custom endpoints
- **Advanced OCR** - Tesseract and EasyOCR integration
- **Cross-platform** - Windows, macOS, Linux support
- **Smart Caching** - Performance optimization
- **Security Focused** - API key management

## Vision & Mission

**Vision**: Enable AI assistants to see and interact with the visual world, breaking down the barrier between text-based AI and real-world interfaces.

**Mission**: Provide the foundational technology for AI-human visual interaction, making AI assistants truly helpful in visual tasks and screen-based workflows.

## Contributing

We welcome contributions to this revolutionary project:
- Bug reports and feature requests
- Code contributions and improvements
- Documentation enhancements

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Ready to give your AI real sight?**

*ScreenMonitorMCP transforms AI assistants from text-only to visually intelligent companions.*