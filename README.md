# 🚀 Revolutionary Screen Monitor MCP Server

[![CI](https://github.com/yourusername/ScreenMonitorMCP/workflows/ScreenMonitorMCP%20CI/badge.svg)](https://github.com/yourusername/ScreenMonitorMCP/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A **REVOLUTIONARY** Model Context Protocol (MCP) server! Gives AI **real-time vision capabilities**, **UI intelligence**, and **predictive behavior learning** power. This isn't just screen capture - it gives AI the power to truly "see" and understand your digital world!

## 🌟 **WHY ScreenMonitorMCP?**

- 🔥 **First & Only**: Real-time continuous screen monitoring feature
- 🧠 **AI Intelligence**: AI that understands UI elements and can interact with them
- 🔮 **Predictive**: System that learns and predicts user behaviors
- ⚡ **Proactive**: Assistant that offers help before you need it
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

### 🧠 **Predictive Intelligence**
- **Behavior Learning**: AI learns your usage patterns and habits
- **Intent Prediction**: Predicts what you'll do next based on context
- **Proactive Help**: Offers help before you ask
- **Workflow Optimization**: Suggests improvements in your work patterns

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

### 🧠 **Predictive AI Tools**
- `learn_user_patterns()` - Learns and analyzes user behavior patterns
- `predict_user_intent()` - Predicts user intent based on current context
- `proactive_assistance()` - Offers proactive help before user requests
- `record_user_action()` - Records user actions and feeds learning system

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

### 🎯 **UI Intelligence**
```python
# Analyze all UI elements on screen
ui_analysis = await analyze_ui_elements()

# Smart clicking with natural language
await smart_click("Click the save button")

# Extract text from screen
text_data = await extract_text_from_screen()
```

### 🧠 **Predictive AI**
```python
# Learn user behavior patterns
patterns = await learn_user_patterns()

# Predict user intent
intent = await predict_user_intent()

# Get proactive assistance
assistance = await proactive_assistance()
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

### **🧠 Predictive Intelligence**
```python
# Learn user behavior patterns
patterns = await learn_user_patterns()

# Predict user intent
intent = await predict_user_intent(
    current_context={"current_app": "VSCode"}
)

# Get proactive assistance
assistance = await proactive_assistance()

# Record user action
await record_user_action(
    action_type="click",
    target="save_button",
    app_context="VSCode"
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
- 🧠 **Smart Understanding**: Recognizes UI elements and interacts with them
- 🔮 **Future Prediction**: Learns and predicts user behaviors
- ⚡ **Proactive Help**: Offers help before you need it
- 🎯 **Natural Interaction**: Understands commands like "Click the save button"

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
