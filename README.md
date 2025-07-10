# 🚀 ScreenMonitorMCP - Revolutionary AI Smart Monitoring System

[![CI](https://github.com/inkbytefo/ScreenMonitorMCP/workflows/ScreenMonitorMCP%20CI/badge.svg)](https://github.com/inkbytefo/ScreenMonitorMCP/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

🌍 **GLOBAL BREAKTHROUGH**: The world's first Model Context Protocol (MCP) server with **intelligent trigger-based monitoring**! This revolutionary system gives AI models **smart real-time vision** and **intelligent interaction capabilities**.

**🔥 BREAKTHROUGH in v4.0.0:** Revolutionary Smart Monitoring System - AI now has intelligent, trigger-based vision with enhanced performance!

> **"This is not just screen monitoring - this is giving AI intelligent sight with smart triggers, context, and interaction."**

## 🌟 **WHY ScreenMonitorMCP?**

- 🌍 **World's First**: Intelligent trigger-based monitoring system
- 🔥 **Revolutionary**: Smart AI vision with intelligent event detection
- 🧠 **Intelligent**: AI that truly understands and interacts with your screen
- ⚡ **Efficient**: Smart trigger system - only analyzes when needed
- 🎯 **Natural**: AI that responds to commands like "Click the save button"
- 🚀 **Global Impact**: Setting new standards for AI-human interaction

## 🔥 **REVOLUTIONARY FEATURES**

### 🧠 **SMART MONITORING SYSTEM** ⭐ **WORLD'S FIRST**
- **Intelligent Trigger System**: Only analyzes when meaningful events occur
- **6 Smart Triggers**: significant_change, error_detected, new_window, application_change, code_change, text_appears
- **Adaptive Performance**: Automatically adjusts FPS based on activity
- **Event Categorization**: Intelligent classification of screen events
- **Global Breakthrough**: First MCP server with trigger-based smart monitoring

### 🎯 **UI Intelligence & Interaction**
- **Smart Click System**: Natural language commands like "Click the save button"
- **OCR Text Extraction**: Reads text from anywhere on the screen with coordinates
- **Intelligent Interaction**: AI knows exactly where and how to interact
- **Context Awareness**: Understands current application and screen state

### 📊 **Application Monitoring**
- **Context Awareness**: Detects which application is currently active
- **Change Detection**: Monitors application-specific changes and events
- **Event Broadcasting**: Relays application events to AI clients
- **Multi-Application Support**: Works with any application (Blender, VSCode, browsers, etc.)

### ⚡ **Performance & Efficiency**
- **Smart Resource Usage**: Only processes when triggers are activated
- **Adaptive FPS**: Automatically adjusts monitoring speed based on activity
- **Event History**: Comprehensive tracking with configurable limits
- **Intelligent Analysis**: AI analysis only when needed, not continuously

## 🛠️ **REVOLUTIONARY MCP TOOLS**

### 🧠 **SMART MONITORING TOOLS** ⭐ **NEW BREAKTHROUGH**
- `start_smart_monitoring()` - 🔥 **THE ULTIMATE TOOL** - Intelligent trigger-based monitoring!
- `stop_smart_monitoring()` - Stop smart monitoring
- `get_monitoring_insights()` - AI-powered insights and analysis results
- `get_recent_events()` - Recently detected smart events with details
- `get_monitoring_summary()` - Comprehensive monitoring session report

### 🎯 **UI Intelligence Tools**
- `smart_click()` - Smart clicking with natural language commands ("Click the save button")
- `extract_text_from_screen()` - OCR text extraction from screen with coordinates

### 📊 **Core Tools**
- `capture_and_analyze()` - Enhanced screen capture and AI analysis
- `record_and_analyze()` - **🎬 NEW!** Video recording and AI analysis with multiple analysis types
- `get_active_application()` - Get currently active application context
- `list_tools()` - **MCP standard compliant** lists all tools with detailed information

## 🎯 **USAGE SCENARIOS**

### 🧠 **SMART MONITORING** ⭐ **BREAKTHROUGH**
```python
# 🔥 THE ULTIMATE EXPERIENCE - Intelligent trigger-based monitoring
await start_smart_monitoring(
    triggers=["significant_change", "error_detected", "new_window"],
    analysis_prompt="Ekranda ne değişti ve neden önemli?",
    fps=2,                  # 2 FPS monitoring
    sensitivity="medium"    # Sensitivity level
)

# Get intelligent insights
insights = await get_monitoring_insights()

# View recent smart events
events = await get_recent_events(limit=5)

# Get comprehensive summary
summary = await get_monitoring_summary()
```

### 🎯 **Smart UI Interaction**
```python
# Smart clicking with natural language
await smart_click("Save button")  # ✅ Works perfectly!
await smart_click("File menu")    # ✅ Enhanced matching!

# Extract text from specific regions
text_data = await extract_text_from_screen(
    region={"x": 100, "y": 100, "width": 300, "height": 200}
)
```

### 📊 **Application Context & Analysis**
```python
# Get active application context
app_context = await get_active_application()

# Enhanced screen capture and analysis
result = await capture_and_analyze(
    analysis_prompt="What's happening on this screen?",
    max_tokens=1000
)

# 🎬 NEW: Video recording and analysis
video_result = await record_and_analyze(
    duration=15,  # Record for 15 seconds
    fps=2,        # 2 frames per second
    analysis_type="summary",  # or "frame_by_frame", "key_moments"
    analysis_prompt="Bu video kaydında ne oldu?",
    save_video=True  # Save video file
)
```

## 🚀 **INSTALLATION**

### **1. Prepare Project Files**
```bash
# Navigate to project directory
cd ScreenMonitorMCP

# Install required libraries
pip install -r requirements.txt
```

### **2. Configure Environment Variables**
Edit the `.env` file:
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

### **3. Standalone Testing (Optional)**
```bash
# Test the server
python main.py

# Test revolutionary features
python test_revolutionary_features.py
```

## 🔧 **MCP CLIENT SETUP**

### **Claude Desktop / MCP Client Configuration**

Add the following JSON to your MCP client's configuration file:

#### **🎯 Simple Configuration (Recommended)**
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

#### **🔧 Advanced Configuration**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": [
        "/path/to/ScreenMonitorMCP/main.py"
      ],
      "cwd": "/path/to/ScreenMonitorMCP",
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

#### **🛡️ Secure Configuration**
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

#### **🪟 Windows Example**
```json
{
  "mcpServers": {
    "screenMonitorMCP": {
      "command": "python",
      "args": ["C:/path/to/ScreenMonitorMCP/main.py"],
      "cwd": "C:/path/to/ScreenMonitorMCP"
    }
  }
}
```

### **⚠️ Important Notes**

1. **File Path**: Update `/path/to/ScreenMonitorMCP/main.py` path according to your project directory
2. **Python Path**: Make sure Python is in PATH or use full path: `"C:/Python311/python.exe"`
3. **Working Directory**: `cwd` parameter is important for proper `.env` file reading
4. **API Keys**: All settings are automatically read from `.env` file

## 🧪 **USAGE EXAMPLES**

### **🎬 Video Recording and Analysis** ⭐ **NEW FEATURE**
```python
# Basic video recording and analysis
result = await record_and_analyze(
    duration=10,  # Record for 10 seconds
    fps=2,        # 2 frames per second
    analysis_type="summary",  # Get overall summary
    analysis_prompt="Bu video kaydında ne oldu? Detaylı analiz yap."
)

# Advanced video recording with key moments detection
result = await record_and_analyze(
    duration=30,
    fps=3,
    capture_mode="monitor",  # Specific monitor
    monitor_number=1,
    analysis_type="key_moments",  # Only analyze significant changes
    analysis_prompt="Önemli değişiklikleri analiz et",
    save_video=True,  # Save video file
    output_format="mp4"
)

# Frame-by-frame detailed analysis
result = await record_and_analyze(
    duration=15,
    fps=1,  # 1 frame per second for detailed analysis
    analysis_type="frame_by_frame",  # Analyze each frame separately
    analysis_prompt="Her kareyi detaylı analiz et",
    max_tokens=500  # Limit tokens per frame
)

# Region-specific video recording
result = await record_and_analyze(
    duration=20,
    fps=2,
    capture_mode="region",
    region={"x": 100, "y": 100, "width": 800, "height": 600},
    analysis_type="summary",
    analysis_prompt="Bu bölgedeki aktiviteyi analiz et"
)
```

### **🔄 Starting Real-Time Monitoring**
```python
# Start AI's continuous vision capability
result = await start_continuous_monitoring(
    fps=3,
    change_threshold=0.1,
    smart_detection=True
)

# Check monitoring status
status = await get_monitoring_status()

# View recent changes
changes = await get_recent_changes(limit=10)

# Stop monitoring
await stop_continuous_monitoring()
```

### **🎯 Using UI Intelligence**
```python
# Analyze all UI elements on screen
ui_elements = await analyze_ui_elements(
    detect_buttons=True,
    extract_text=True,
    confidence_threshold=0.7
)

# Smart clicking with natural language
await smart_click("Click the save button", dry_run=False)

# Extract text from specific region
text_data = await extract_text_from_screen(
    region={"x": 100, "y": 100, "width": 500, "height": 300}
)
```

### **📊 Application Monitoring**
```python
# Start application monitoring
await start_application_monitoring()

# Get active application context
app_context = await get_active_application()

# Register Blender for monitoring
await register_application_events(
    app_name="Blender",
    event_types=["scene_change", "object_modification"]
)

# Monitor application changes
changes = await get_recent_application_events(limit=10)

# Broadcast Blender scene change
await broadcast_application_change(
    app_name="Blender",
    event_type="scene_change",
    event_data={"objects_modified": ["Cube", "Camera"]}
)
```

## 🎯 **BLENDER INTEGRATION EXAMPLE**

With this system, you can relay real-time changes from Blender to your AI client (like Claude Desktop):

### **Step 1: Start ScreenMonitorMCP**
```bash
# Add ScreenMonitorMCP to your Claude Desktop config
python main.py
```

### **Step 2: Activate Application Monitoring**
```python
# Run these commands in Claude Desktop:
await start_application_monitoring()
await register_application_events("Blender")
```

### **Step 3: Work in Blender**
- Open Blender and make changes to your scene
- ScreenMonitorMCP automatically detects window focus changes
- Your AI client knows you're working in Blender

### **Step 4: Send Custom Events (Future Feature)**
```python
# From within your Blender script:
await broadcast_application_change(
    app_name="Blender",
    event_type="object_added",
    event_data={"object_name": "Suzanne", "object_type": "MESH"}
)
```


### **📸 Screen Capture and Analysis**
```python
# Enhanced screen capture and analysis
result = await capture_and_analyze(
    capture_mode="all",
    analysis_prompt="What do you see on this screen?",
    max_tokens=1500  # AI models can now use more tokens for detailed analysis
)
```

### **🎬 Video Recording and Analysis** ⭐ **NEW!**
```python
# Record screen activity and analyze
video_analysis = await record_and_analyze(
    duration=15,  # 15 seconds recording
    fps=2,        # 2 frames per second
    analysis_type="summary",  # Get overall summary
    analysis_prompt="Analyze what happened in this recording",
    save_video=True  # Save video file for later reference
)

# List all tools
tools = await list_tools()
```

## 🚀 **REVOLUTIONARY CAPABILITIES**

This MCP server gives AI the following capabilities:

- 👁️ **Continuous Vision**: AI can monitor the screen non-stop
- 🧠 **Enhanced Smart Understanding**: Recognizes UI elements and interacts with them (75% success rate!)
- 🎯 **Advanced Natural Interaction**: Understands commands like "File", "Save button" with fuzzy matching
- 📍 **Menu Intelligence**: Detects menu bars, menu items, and UI hierarchies
- 🔍 **Multi-Strategy Matching**: Fuzzy text matching, position-based scoring, and semantic understanding
- ⚡ **Proactive Help**: Offers help before you need it
- 📊 **Application Awareness**: Monitors and broadcasts application events

## 🔧 **TROUBLESHOOTING**

### **Common Issues and Solutions**

1. **Unicode/Encoding Error (Windows)**
   ```
   UnicodeEncodeError: 'charmap' codec can't encode character
   ```
   **Solution:** ✅ This error is fixed! Server automatically uses UTF-8 encoding.

2. **JSON Configuration Error**
   ```json
   // ❌ Wrong
   {
     "command": "python",
     "args": ["path/to/main.py",]  // Trailing comma is wrong
   }

   // ✅ Correct
   {
     "command": "python",
     "args": ["path/to/main.py"]
   }
   ```

3. **Python Path Issue**
   ```json
   {
     "command": "C:/Python311/python.exe",  // Use full path
     "args": ["C:/path/to/ScreenMonitorMCP/main.py"]
   }
   ```

4. **Missing Dependencies**
   ```bash
   cd ScreenMonitorMCP
   pip install -r requirements.txt
   ```

5. **OCR Issues**
   ```bash
   # Install Tesseract (optional)
   # EasyOCR installs automatically
   ```

6. **MCP Connection Closed Error**
   ```
   MCP error -32000: Connection closed
   ```
   **Solution:** Check file paths and add `cwd` parameter.

## 📝 **LICENSE**

This project is licensed under the MIT License.

---

**🚀 Revolutionary MCP server that gives AI real "eyes"!**
**🔥 Next-generation AI-human interaction starts here!**
