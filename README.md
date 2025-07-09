# 🚀 Revolutionary Screen Monitor MCP Server

[![CI](https://github.com/inkbytefo/ScreenMonitorMCP/workflows/ScreenMonitorMCP%20CI/badge.svg)](https://github.com/inkbytefo/ScreenMonitorMCP/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A **REVOLUTIONARY** Model Context Protocol (MCP) server! Gives AI **real-time vision capabilities** and **enhanced UI intelligence** power. This isn't just screen capture - it gives AI the power to truly "see" and understand your digital world!

**🎯 NEW in v2.1.0:** Enhanced Smart Click with 75% success rate, menu detection, and fuzzy matching!

## 🌟 **WHY ScreenMonitorMCP?**

- 🔥 **First & Only**: Real-time continuous screen monitoring feature
- 🧠 **AI Intelligence**: AI that understands UI elements and can interact with them
- ⚡ **Intelligent**: Smart UI detection and interaction capabilities
- 🎯 **Natural**: AI that understands commands like "Click the save button"

## 🔥 **REVOLUTIONARY FEATURES**

### 🔄 **Real-Time Continuous Monitoring**
- **AI's Eyes Never Close**: 2-5 FPS continuous screen monitoring
- **Smart Change Detection**: Distinguishes between small, major, and critical changes
- **Proactive Analysis**: AI automatically analyzes important changes
- **Adaptive Performance**: Smart frame rate adjustment

### 🎯 **UI Element Intelligence**
- **Computer Vision UI Detection**: Automatically recognizes buttons, forms, menus
- **OCR Text Extraction**: Reads text from anywhere on the screen
- **Smart Click System**: Natural language commands like "Click the save button"
- **Interaction Mapping**: AI knows exactly where and how to interact

### 📊 **Application Monitoring**
- **Context Awareness**: Detects which application is currently active
- **Change Detection**: Monitors application-specific changes and events
- **Event Broadcasting**: Relays application events to AI clients
- **Multi-Application Support**: Works with any application (Blender, VSCode, browsers, etc.)

## 🛠️ **REVOLUTIONARY MCP TOOLS**

### 🔄 **Real-Time Monitoring Tools**
- `start_continuous_monitoring()` - Starts AI's continuous vision capability
- `stop_continuous_monitoring()` - Stops continuous monitoring
- `get_monitoring_status()` - Real-time status information and statistics
- `get_recent_changes()` - Recently detected screen changes

### 🎯 **UI Intelligence Tools**
- `analyze_ui_elements()` - Recognizes and maps all UI elements on screen
- `smart_click()` - Smart clicking with natural language commands ("Click the save button")
- `extract_text_from_screen()` - OCR text extraction from screen

### 📊 **Application Monitoring Tools**
- `get_active_application()` - Get currently active application context
- `register_application_events()` - Register for application-specific events
- `broadcast_application_change()` - Broadcast application changes to AI clients

### 📸 **Traditional Tools**
- `capture_and_analyze()` - Screen capture and AI analysis (enhanced)
- `list_tools()` - **MCP standard compliant** lists all tools (categorized, detailed information)

## 🎯 **USAGE SCENARIOS**

### 🔍 **Real-Time Monitoring**
```python
# Start AI's continuous vision capability
await start_continuous_monitoring(fps=3, change_threshold=0.1)

# Check monitoring status
status = await get_monitoring_status()

# View recent changes
changes = await get_recent_changes(limit=5)
```

### 🎯 **Enhanced UI Intelligence** ⭐ NEW
```python
# Analyze all UI elements on screen (now with menu detection!)
ui_analysis = await analyze_ui_elements()

# Smart clicking with natural language (75% success rate!)
await smart_click("File")  # ✅ Works!
await smart_click("Save button")  # ✅ Enhanced matching!

# Extract text from screen with OCR
text_data = await extract_text_from_screen()
```

### 📊 **Application Monitoring**
```python
# Get active application context
app_context = await get_active_application()

# Register for application events
await register_application_events(app_name="Blender")

# Monitor application changes
changes = await get_recent_changes(limit=5)
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


### **📸 Traditional Screen Capture**
```python
# Enhanced screen capture and analysis
result = await capture_and_analyze(
    capture_mode="all",
    analysis_prompt="What do you see on this screen?",
    max_tokens=500
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
