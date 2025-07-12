import argparse
import os
from dotenv import load_dotenv
import mss
import base64
from io import BytesIO
import pygetwindow as gw
from PIL import Image
from typing import Any, Literal, Optional, Dict, List
import asyncio
import structlog
from datetime import datetime
import numpy as np
import time

from mcp.server.fastmcp import FastMCP
from ai_providers import OpenAIProvider
from ui_detection import get_ui_detector, get_smart_clicker
from application_monitor import ApplicationMonitor, ApplicationEvent, get_global_app_monitor, set_global_app_monitor
from smart_monitoring import SmartMonitor, SmartMonitoringConfig, SmartEvent, get_global_smart_monitor, set_global_smart_monitor
from video_recorder import VideoRecorder, VideoAnalyzer, VideoRecordingConfig, VideoAnalysisResult
from cache_manager import get_cache_manager, shutdown_cache
from conversation_context import get_conversation_manager
from system_metrics import get_metrics_manager
from batch_processor import get_batch_processor, BatchPriority
from image_optimizer import get_image_optimizer, OptimizationConfig
from error_recovery import get_error_recovery_manager, with_recovery
from platform_support import get_platform_manager
from input_simulator import get_input_simulator, KeyboardInput, MouseInput, MouseButton


# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Command-line arguments setup
parser = argparse.ArgumentParser(description="Screen Monitor MCP Server")
parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="Host IP to run the server on")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 7777)), help="Port to run the server on")
parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="API Key for securing the endpoints")
parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key for vision analysis")
parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="Custom OpenAI API Base URL")
parser.add_argument("--default-openai-model", default=os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o"), help="Default OpenAI model for analysis")
parser.add_argument("--default-max-tokens", type=int, default=int(os.getenv("DEFAULT_MAX_TOKENS", 1000)), help="Default max tokens for AI analysis")
args = parser.parse_args()

API_KEY = args.api_key
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url
DEFAULT_OPENAI_MODEL = args.default_openai_model
DEFAULT_MAX_TOKENS = args.default_max_tokens

# Initialize AI Provider
openai_provider = None

if OPENAI_API_KEY:
    try:
        openai_provider = OpenAIProvider(OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
        logger.info("OpenAI Provider ENABLED")
        if OPENAI_BASE_URL:
            logger.info("Using custom OpenAI Base URL", base_url=OPENAI_BASE_URL)
    except ValueError as e:
        logger.warning("OpenAI Provider could not be initialized", error=str(e))

if not API_KEY:
    logger.warning("No API_KEY found. The server will be unsecured.")

if not OPENAI_API_KEY:
    logger.warning("No OpenAI API Key found. Vision analysis will not be available.")

# Initialize FastMCP server
mcp = FastMCP("screen-monitor")

# Initialize Smart Monitor
smart_config = SmartMonitoringConfig(
    triggers=["significant_change"],
    analysis_prompt="Ekranda ne değişti ve neden önemli?",
    fps=2,
    sensitivity="medium",
    adaptive_fps=True
)
smart_monitor = SmartMonitor(smart_config, openai_provider)
set_global_smart_monitor(smart_monitor)

# Initialize Application Monitor
app_monitor = ApplicationMonitor()
set_global_app_monitor(app_monitor)

# Setup smart event callback for AI analysis
def on_smart_event(event: SmartEvent):
    """Smart event algılandığında callback"""
    logger.info("Smart event detected",
               event_type=event.event_type,
               description=event.description,
               confidence=event.confidence)

smart_monitor.add_event_callback(on_smart_event)

# Application monitoring callback
def on_application_event(app_event: ApplicationEvent):
    """Handle application events"""
    try:
        logger.info("Application event detected",
                   event_type=app_event.event_type.value,
                   app_name=app_event.application_name,
                   window_title=app_event.window_title)

        # If we have an AI provider, analyze the event
        if openai_provider:
            try:
                event_description = f"Application event: {app_event.event_type.value} in {app_event.application_name}"
                if app_event.window_title:
                    event_description += f" (Window: {app_event.window_title})"

                # For now, just log the event. In the future, we could capture screenshots
                # and send them to AI for analysis
                logger.info("Application event logged for AI analysis", description=event_description)

            except Exception as e:
                logger.error("AI analysis failed for application event", error=str(e))
    except Exception as e:
        logger.error("Application event callback error", error=str(e))

app_monitor.add_event_callback(on_application_event)

# Helper function to capture screenshot
def _capture_screenshot_to_base64(capture_mode: str, monitor_number: int, capture_active_window: bool, region: Optional[Dict[str, int]], output_format: str) -> tuple[str, Dict[str, Any]]:
    with mss.mss() as sct:
        capture_area = None
        
        if capture_mode == "window" and capture_active_window:
            active_window = gw.getActiveWindow()
            if active_window:
                capture_area = {
                    "top": active_window.top,
                    "left": active_window.left,
                    "width": active_window.width,
                    "height": active_window.height
                }
            else:
                raise ValueError("No active window found.")
        elif capture_mode == "monitor":
            if monitor_number < 1 or monitor_number >= len(sct.monitors):
                raise ValueError(f"Invalid monitor number. Available monitors: 1 to {len(sct.monitors) - 1}")
            capture_area = sct.monitors[monitor_number]
        elif capture_mode == "region":
            if not region:
                raise ValueError("Region must be specified for 'region' capture mode.")
            capture_area = region
        else:  # "all"
            capture_area = sct.monitors[0]  # Capture all screens combined

        if not capture_area:
            raise ValueError("Could not determine the capture area.")

        sct_img = sct.grab(capture_area)
        
        img_buffer = BytesIO()
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        if output_format == "png":
            img.save(img_buffer, "PNG")
        else:  # jpeg
            img.save(img_buffer, "JPEG")

        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_base64, capture_area

# MCP Tools

@mcp.tool()
async def list_tools() -> Dict[str, Any]:
    """
    Lists all available MCP tools with revolutionary features categorization.

    This function provides a comprehensive overview of all available tools in the MCP server,
    categorized by their revolutionary capabilities and standard features.

    Returns:
        Dict containing:
        - total_tools: Total number of available tools
        - revolutionary_features: AI vision and intelligence tools
        - standard_features: Traditional screen capture tools
        - capabilities: Server capabilities and status
        - tool_categories: Organized tool categories
    """
    tools = []
    revolutionary_tools = []
    standard_tools = []

    # Define all available tools manually (FastMCP doesn't expose _tools)
    all_available_tools = [
        {
            "name": "list_tools",
            "description": "Lists all available MCP tools with revolutionary features categorization",
            "category": "standard",
            "parameters": []
        },
        {
            "name": "capture_and_analyze",
            "description": "Captures screenshot and analyzes it using AI (enhanced version)",
            "category": "standard",
            "parameters": ["capture_mode", "monitor_number", "analysis_prompt", "max_tokens"]
        },
        {
            "name": "record_and_analyze",
            "description": "REVOLUTIONARY: Records screen video and analyzes it using AI",
            "category": "revolutionary",
            "parameters": ["duration", "fps", "capture_mode", "analysis_type", "analysis_prompt", "save_video"]
        },


        {
            "name": "smart_click",
            "description": "REVOLUTIONARY: AI clicks elements using natural language commands",
            "category": "revolutionary",
            "parameters": ["element_description", "confidence_threshold", "dry_run"]
        },
        {
            "name": "extract_text_from_screen",
            "description": "REVOLUTIONARY: Extracts text from screen using OCR technology",
            "category": "revolutionary",
            "parameters": ["region", "ocr_engine"]
        },
        {
            "name": "get_active_application",
            "description": "Get currently active application context",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "start_smart_monitoring",
            "description": "REVOLUTIONARY: Starts intelligent monitoring with trigger-based analysis",
            "category": "revolutionary",
            "parameters": ["triggers", "analysis_prompt", "fps", "sensitivity"]
        },
        {
            "name": "stop_smart_monitoring",
            "description": "Stops smart monitoring",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "get_monitoring_insights",
            "description": "Gets intelligent monitoring insights and analysis results",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "get_recent_events",
            "description": "Gets recent smart monitoring events",
            "category": "revolutionary",
            "parameters": ["limit"]
        },
        {
            "name": "get_monitoring_summary",
            "description": "Gets comprehensive monitoring summary report",
            "category": "revolutionary",
            "parameters": []
        },

    ]

    # Categorize tools
    for tool_data in all_available_tools:
        if tool_data["category"] == "revolutionary":
            revolutionary_tools.append(tool_data)
        else:
            standard_tools.append(tool_data)
        tools.append(tool_data)

    # Get server status
    smart_monitor = get_global_smart_monitor()
    app_monitor = get_global_app_monitor()
    server_status = {
        "ai_provider": "OpenAI" if openai_provider else "None",
        "smart_monitoring": smart_monitor.is_monitoring if smart_monitor else False,
        "application_monitoring": app_monitor.is_monitoring if app_monitor else False,
        "total_smart_events": len(smart_monitor.event_history) if smart_monitor else 0,
        "total_app_events": len(app_monitor.event_history) if app_monitor else 0,
        "server_version": "2.1.0-smart-click-enhanced",
        "capabilities": [
            "Real-time screen monitoring",
            "Application context detection",
            "UI element detection",
            "Smart click automation",
            "OCR text extraction",
            "Application event broadcasting",
            "Multi-application support"
        ]
    }

    return {
        "mcp_server": "Revolutionary Screen Monitor",
        "version": "2.1.0",
        "total_tools": len(tools),
        "revolutionary_features": {
            "count": len(revolutionary_tools),
            "description": "AI vision and intelligence tools that give AI real-time sight and interaction capabilities",
            "tools": revolutionary_tools,
            "categories": {
                "real_time_monitoring": [t for t in revolutionary_tools if "monitoring" in t["name"] or "changes" in t["name"]],
                "ui_intelligence": [t for t in revolutionary_tools if "ui" in t["name"] or "click" in t["name"] or "text" in t["name"]],
                "predictive_ai": [t for t in revolutionary_tools if "learn" in t["name"] or "predict" in t["name"] or "proactive" in t["name"]]
            }
        },
        "standard_features": {
            "count": len(standard_tools),
            "description": "Traditional screen capture and analysis tools",
            "tools": standard_tools
        },
        "server_status": server_status,
        "usage_examples": {
            "smart_monitoring": "await start_smart_monitoring(triggers=['significant_change', 'error_detected'])",
            "smart_interaction": "await smart_click('Save button')",
            "text_extraction": "await extract_text_from_screen()",
            "monitoring_insights": "await get_monitoring_insights()"
        },
        "documentation": {
            "quick_start": "See QUICK_START.md for setup instructions",
            "full_docs": "See README.md for complete documentation",
            "revolutionary_features": "This server gives AI real-time vision and predictive intelligence"
        }
    }



# === REVOLUTIONARY FEATURE 2: SMART CLICK ===

@mcp.tool()
async def smart_click(
    element_description: str,
    confidence_threshold: float = 0.8,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    REVOLUTIONARY FEATURE: AI finds and clicks elements described in natural language.

    Example usage:
    - "Click the save button"
    - "Click the email field"
    - "Find and click the login button"

    Args:
        element_description: Natural language description of element to click
        confidence_threshold: Minimum confidence threshold
        dry_run: Only find, don't click (test mode)

    Returns:
        Click result and element information
    """
    try:
        # Get metrics manager for enhanced functionality
        metrics_manager = get_metrics_manager()

        # Increment counter
        metrics_manager.increment_counter("smart_clicks")

        clicker = get_smart_clicker()

        if dry_run:
            # Only find element, don't click
            element = clicker.find_element_by_text(element_description, confidence_threshold)

            if element:
                return {
                    "success": True,
                    "dry_run": True,
                    "element_found": True,
                    "element_type": element.element_type,
                    "coordinates": element.center_point,
                    "text_content": element.text_content,
                    "confidence": element.confidence,
                    "description": element.description,
                    "message": f"Element bulundu ama tıklanmadı (dry_run=True)"
                }
            else:
                return {
                    "success": False,
                    "dry_run": True,
                    "element_found": False,
                    "message": f"Element bulunamadı: '{element_description}'"
                }
        else:
            # Gerçek tıklama
            result = clicker.smart_click(element_description)
            result["revolutionary_feature"] = "AI Smart Click"
            result["description"] = "AI doğal dil komutlarını anlayıp UI ile etkileşim kurabiliyor!"

            logger.info("Smart click executed",
                       description=element_description,
                       success=result.get("success", False))

            return result

    except Exception as e:
        logger.error("Smart click failed", error=str(e))
        return {"error": f"Smart click failed: {str(e)}"}

@mcp.tool()
async def extract_text_from_screen(
    region: Optional[Dict[str, int]] = None,
    ocr_engine: str = "auto"
) -> Dict[str, Any]:
    """
    REVOLUTIONARY FEATURE: Extract text from screen or specific region.

    Args:
        region: Specific region {'x': int, 'y': int, 'width': int, 'height': int}
        ocr_engine: OCR engine ('auto', 'tesseract', 'easyocr')

    Returns:
        Extracted texts and their coordinates
    """
    try:
        import mss
        import cv2

        # Get metrics manager for enhanced functionality
        metrics_manager = get_metrics_manager()

        # Increment counter
        metrics_manager.increment_counter("ui_detections")

        # Take screenshot
        with mss.mss() as sct:
            if region:
                # Capture specific region
                capture_area = {
                    "top": region["y"],
                    "left": region["x"],
                    "width": region["width"],
                    "height": region["height"]
                }
            else:
                # Capture entire screen
                capture_area = sct.monitors[0]

            sct_img = sct.grab(capture_area)
            screenshot = np.array(sct_img)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Apply OCR
        detector = get_ui_detector()

        # Update OCR engine setting
        if ocr_engine in ["tesseract", "easyocr"]:
            detector.ocr_engine.preferred_engine = ocr_engine

        logger.info("Starting OCR extraction", engine=ocr_engine, region=region)

        ocr_results = detector.ocr_engine.extract_text(screenshot)

        # Sonuçları formatla
        extracted_texts = []
        for result in ocr_results:
            extracted_texts.append({
                "text": result["text"],
                "coordinates": result["coordinates"],
                "confidence": round(result["confidence"], 3),
                "method": result["method"]
            })

        # OCR engine durumunu kontrol et
        ocr_engine_used = "none"
        if ocr_results:
            ocr_engine_used = ocr_results[0]["method"]
        elif not detector.ocr_engine.easyocr_reader and ocr_engine == "easyocr":
            ocr_engine_used = "easyocr_not_available"
        elif ocr_engine == "tesseract":
            from ui_detection import TESSERACT_AVAILABLE
            if not TESSERACT_AVAILABLE:
                ocr_engine_used = "tesseract_not_available"

        logger.info("Text extraction completed",
                   texts_found=len(extracted_texts),
                   ocr_engine=ocr_engine,
                   engine_used=ocr_engine_used)

        return {
            "revolutionary_feature": "AI Text Extraction",
            "description": "AI can read texts from screen and knows their locations!",
            "total_texts_found": len(extracted_texts),
            "ocr_engine_used": ocr_engine_used,
            "region_analyzed": region or "full_screen",
            "extracted_texts": extracted_texts,
            "debug_info": {
                "requested_engine": ocr_engine,
                "screenshot_size": screenshot.shape if 'screenshot' in locals() else None,
                "available_engines": {
                    "tesseract": getattr(__import__('ui_detection'), 'TESSERACT_AVAILABLE', False),
                    "easyocr": getattr(__import__('ui_detection'), 'EASYOCR_AVAILABLE', False)
                }
            },
            "capabilities": [
                "OCR text extraction",
                "Multi-language support",
                "Coordinate mapping",
                "Confidence scoring"
            ]
        }

    except Exception as e:
        logger.error("Text extraction failed", error=str(e))
        return {
            "error": f"Text extraction failed: {str(e)}",
            "total_texts_found": 0,
            "ocr_engine_used": "error",
            "extracted_texts": []
        }

# === APPLICATION MONITORING SYSTEM ===

@mcp.tool()
async def get_active_application() -> Dict[str, Any]:
    """
    Get currently active application context.

    Returns:
        Information about the currently active application and window
    """
    try:
        app_monitor = get_global_app_monitor()
        if not app_monitor:
            return {"error": "Application monitor not initialized"}

        result = app_monitor.get_active_application()

        return {
            "status": "success",
            "active_application": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get active application", error=str(e))
        return {"error": f"Failed to get active application: {str(e)}"}

# === SMART MONITORING SYSTEM ===

@mcp.tool()
async def start_smart_monitoring(
    triggers: List[str] = ["significant_change"],
    analysis_prompt: str = "Ekranda ne değişti ve neden önemli?",
    fps: int = 2,
    sensitivity: Literal["low", "medium", "high"] = "medium"
) -> Dict[str, Any]:
    """
    REVOLUTIONARY FEATURE: Starts intelligent monitoring with trigger-based analysis.

    This combines real-time monitoring with smart analysis, only analyzing when
    meaningful events occur based on configured triggers.

    Args:
        triggers: List of triggers to monitor for:
                 - "significant_change": Major visual changes (>20%)
                 - "error_detected": Red text/error messages
                 - "new_window": New windows opening
                 - "application_change": App switching
                 - "code_change": Changes in code editors
                 - "text_appears": Specific text appearing
        analysis_prompt: Custom prompt for AI analysis
        fps: Frames per second for monitoring (1-5 recommended)
        sensitivity: Monitoring sensitivity level

    Returns:
        Smart monitoring startup status and configuration
    """
    try:
        smart_monitor = get_global_smart_monitor()
        if not smart_monitor:
            return {"error": "Smart monitor not initialized"}

        # Update configuration
        smart_monitor.config.triggers = triggers
        smart_monitor.config.analysis_prompt = analysis_prompt
        smart_monitor.config.fps = max(1, min(5, fps))

        # Set sensitivity thresholds
        if sensitivity == "low":
            smart_monitor.config.significant_change_threshold = 0.3
        elif sensitivity == "medium":
            smart_monitor.config.significant_change_threshold = 0.2
        else:  # high
            smart_monitor.config.significant_change_threshold = 0.1

        result = smart_monitor.start_monitoring()

        logger.info("Smart monitoring started via MCP",
                   triggers=triggers, fps=fps, sensitivity=sensitivity)

        return {
            **result,
            "revolutionary_feature": "Smart Monitoring with AI Analysis",
            "description": "AI now intelligently monitors screen with trigger-based analysis!",
            "active_triggers": triggers,
            "capabilities": [
                "Intelligent change detection",
                "Trigger-based analysis",
                "Adaptive FPS control",
                "Event categorization",
                "Real-time insights"
            ]
        }

    except Exception as e:
        logger.error("Failed to start smart monitoring", error=str(e))
        return {"error": f"Smart monitoring could not be started: {str(e)}"}

@mcp.tool()
async def stop_smart_monitoring() -> Dict[str, Any]:
    """
    Stops smart monitoring.

    Returns:
        Smart monitoring stop status and statistics
    """
    try:
        smart_monitor = get_global_smart_monitor()
        if not smart_monitor:
            return {"error": "Smart monitor not found"}

        result = smart_monitor.stop_monitoring()
        logger.info("Smart monitoring stopped via MCP", stats=result.get("stats", {}))

        return {
            **result,
            "message": "Smart monitoring stopped successfully",
            "note": "Event history and insights preserved"
        }

    except Exception as e:
        logger.error("Failed to stop smart monitoring", error=str(e))
        return {"error": f"Smart monitoring could not be stopped: {str(e)}"}

@mcp.tool()
async def get_monitoring_insights() -> Dict[str, Any]:
    """
    Gets intelligent monitoring insights and analysis results.

    Returns:
        Comprehensive insights from smart monitoring system
    """
    try:
        smart_monitor = get_global_smart_monitor()
        if not smart_monitor:
            return {"error": "Smart monitor not found"}

        insights = smart_monitor.get_monitoring_insights()

        return {
            **insights,
            "revolutionary_feature": "AI Monitoring Insights",
            "description": "Intelligent analysis of screen activity patterns and events"
        }

    except Exception as e:
        logger.error("Failed to get monitoring insights", error=str(e))
        return {"error": f"Monitoring insights could not be retrieved: {str(e)}"}

@mcp.tool()
async def get_recent_events(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Gets recent smart monitoring events.

    Args:
        limit: Maximum number of recent events to retrieve

    Returns:
        List of recent smart monitoring events with details
    """
    try:
        smart_monitor = get_global_smart_monitor()
        if not smart_monitor:
            return [{"error": "Smart monitor not found"}]

        events = smart_monitor.get_recent_events(limit)

        return events

    except Exception as e:
        logger.error("Failed to get recent events", error=str(e))
        return [{"error": f"Recent events could not be retrieved: {str(e)}"}]

@mcp.tool()
async def get_monitoring_summary() -> Dict[str, Any]:
    """
    Gets comprehensive monitoring summary report.

    Returns:
        Detailed summary of monitoring session and activity
    """
    try:
        smart_monitor = get_global_smart_monitor()
        if not smart_monitor:
            return {"error": "Smart monitor not found"}

        summary = smart_monitor.get_monitoring_summary()

        return {
            **summary,
            "revolutionary_feature": "Smart Monitoring Summary",
            "description": "Comprehensive overview of AI monitoring session"
        }

    except Exception as e:
        logger.error("Failed to get monitoring summary", error=str(e))
        return {"error": f"Monitoring summary could not be retrieved: {str(e)}"}







@mcp.tool()
async def capture_and_analyze(capture_mode: Literal["all", "monitor", "window", "region"] = "all", monitor_number: int = 1, capture_active_window: bool = False, region: Optional[Dict[str, int]] = None, output_format: Literal["png", "jpeg"] = "png", analysis_prompt: str = "Please analyze this screenshot and provide information about its content.", max_tokens: Optional[int] = None) -> str:
    """
    Captures a screenshot and directly sends it to an AI model for analysis, returning the result.
    
    Args:
        capture_mode: Specify what to capture. Options are 'all', 'monitor', 'window', 'region'.
        monitor_number: The monitor number to capture (1-based index). Only used in 'monitor' mode.
        capture_active_window: If true, captures the currently active window. Overrides other settings if in 'window' mode.
        region: Specify a region to capture: {'top': int, 'left': int, 'width': int, 'height': int}. Only used in 'region' mode.
        output_format: The output image format. Options are 'png' or 'jpeg'.
        analysis_prompt: The prompt for the AI model to analyze the screenshot.
        max_tokens: The maximum number of tokens for the AI model's response.
    
    Returns:
        A string containing the analysis result from the AI model.
    """
    if not openai_provider:
        return "OpenAI sağlayıcısı yapılandırılmamış veya API Anahtarı eksik. Lütfen API anahtarını ayarlayın."
    
    try:
        # Get managers for enhanced functionality
        cache_manager = get_cache_manager()
        metrics_manager = get_metrics_manager()

        # Increment counters
        metrics_manager.increment_counter("screenshots")
        metrics_manager.increment_counter("analyses")

        # Generate cache key for analysis
        import hashlib
        cache_key_data = f"{capture_mode}_{monitor_number}_{capture_active_window}_{region}_{analysis_prompt}"
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()

        # Check cache for recent analysis (1 minute TTL for same parameters)
        cached_result = cache_manager.get("analyses", cache_key)
        if cached_result:
            logger.info("Analysis served from cache", cache_key=cache_key[:8])
            return f"📋 [CACHED] {cached_result}"

        img_base64, capture_details = _capture_screenshot_to_base64(capture_mode, monitor_number, capture_active_window, region, output_format)

        # Cache the screenshot
        screenshot_id = f"screenshot_{int(time.time())}"
        cache_manager.set("screenshots", screenshot_id, img_base64, ttl=300)  # 5 minutes

        model_to_use = DEFAULT_OPENAI_MODEL
        tokens_to_use = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

        # Track AI provider performance
        start_time = time.time()
        ai_analysis = await openai_provider.analyze_image(
            image_base64=img_base64,
            prompt=analysis_prompt,
            model=model_to_use,
            output_format=output_format,
            max_tokens=tokens_to_use
        )
        response_time = time.time() - start_time

        # Update provider metrics
        metrics_manager.update_provider_status("openai", "active", response_time)

        # Cache the analysis result
        result = f"Screenshot successfully captured and analyzed. Analysis: {ai_analysis}. Capture details: {capture_details}. Model used: {model_to_use}. Provider used: openai"
        cache_manager.set("analyses", cache_key, result, ttl=60)  # 1 minute

        logger.info("Screenshot analysis completed",
                   response_time=response_time,
                   model=model_to_use,
                   cached=False)

        return result

    except Exception as e:
        # Update provider metrics on error
        metrics_manager = get_metrics_manager()
        metrics_manager.update_provider_status("openai", "error", error=str(e))

        logger.error("Screenshot capture and analysis failed", error=str(e))
        return f"Screenshot capture and analysis failed. Error: {str(e)}. Please check parameters or try again later."

@mcp.tool()
async def record_and_analyze(
    duration: int = 10,
    fps: int = 2,
    capture_mode: Literal["all", "monitor", "window", "region"] = "all",
    monitor_number: int = 1,
    region: Optional[Dict[str, int]] = None,
    analysis_type: Literal["summary", "frame_by_frame", "key_moments"] = "summary",
    analysis_prompt: str = "Bu video kaydında ne olduğunu detaylıca analiz et",
    max_tokens: Optional[int] = None,
    save_video: bool = False,
    output_format: Literal["mp4", "avi"] = "mp4"
) -> str:
    """
    Records screen for specified duration and analyzes the video using AI.

    This is the video version of capture_and_analyze - records screen activity
    and provides detailed AI analysis of what happened during the recording.

    Args:
        duration: Recording duration in seconds (default: 10)
        fps: Frames per second for recording (default: 2)
        capture_mode: What to capture - 'all', 'monitor', 'window', or 'region'
        monitor_number: Monitor number to capture (1-based index)
        region: Region to capture: {'x': int, 'y': int, 'width': int, 'height': int}
        analysis_type: Type of analysis - 'summary', 'frame_by_frame', or 'key_moments'
        analysis_prompt: Custom prompt for AI analysis
        max_tokens: Maximum tokens for AI response
        save_video: Whether to save the video file
        output_format: Video format - 'mp4' or 'avi'

    Returns:
        Detailed analysis of the recorded video content
    """
    if not openai_provider:
        return "OpenAI sağlayıcısı yapılandırılmamış veya API Anahtarı eksik. Lütfen API anahtarını ayarlayın."

    try:
        # Create video recording configuration
        config = VideoRecordingConfig(
            duration=duration,
            fps=fps,
            capture_mode=capture_mode,
            monitor_number=monitor_number,
            region=region,
            analysis_type=analysis_type,
            analysis_prompt=analysis_prompt,
            max_tokens=max_tokens,
            save_video=save_video,
            output_format=output_format
        )

        # Create recorder and analyzer
        recorder = VideoRecorder(config)
        analyzer = VideoAnalyzer(openai_provider)

        logger.info("Starting video recording and analysis",
                   duration=duration,
                   fps=fps,
                   analysis_type=analysis_type,
                   capture_mode=capture_mode)

        # Record video
        recording_result = recorder.start_recording()

        if recording_result["status"] != "completed":
            return f"Video kaydı başarısız oldu: {recording_result.get('message', 'Bilinmeyen hata')}"

        # Analyze video
        analysis_result = await analyzer.analyze_video(recorder)

        # Cleanup
        recorder.cleanup()

        # Format response
        response = f"## Video Kaydı ve Analizi Tamamlandı\n\n"
        response += f"**Kayıt Detayları:**\n"
        response += f"- Süre: {analysis_result.duration:.1f} saniye\n"
        response += f"- Toplam Kare: {analysis_result.total_frames}\n"
        response += f"- Önemli Anlar: {len(analysis_result.key_moments)}\n"
        response += f"- Analiz Türü: {analysis_result.analysis_type}\n"
        response += f"- İşlem Süresi: {analysis_result.processing_time:.1f} saniye\n"

        if analysis_result.video_path:
            response += f"- Video Dosyası: {analysis_result.video_path}\n"

        response += f"\n**AI Analizi:**\n{analysis_result.analysis_text}\n"

        if analysis_result.key_moments:
            response += f"\n**Önemli Anlar:**\n"
            for moment in analysis_result.key_moments:
                response += f"- Kare {moment['frame_number']}: {moment['timestamp']} (Değişim: %{moment['change_percentage']*100:.1f})\n"

        logger.info("Video recording and analysis completed",
                   total_frames=analysis_result.total_frames,
                   key_moments=len(analysis_result.key_moments),
                   processing_time=analysis_result.processing_time)

        return response

    except Exception as e:
        logger.error("Video recording and analysis failed", error=str(e))
        return f"Video kaydı ve analizi başarısız oldu: {str(e)}. Lütfen parametreleri kontrol edin ve tekrar deneyin."

# === NEW ENHANCED FEATURES ===

@mcp.tool()
async def query_vision_about_current_view(
    question: str,
    context: Optional[str] = None,
    use_cache: bool = True,
    conversation_id: Optional[str] = None
) -> str:
    """
    REVOLUTIONARY FEATURE: Query AI about current screen view with conversation context.

    This tool captures the current screen and asks the AI a specific question about it,
    while maintaining conversation context and history.

    Args:
        question: The question to ask about the current screen view
        context: Additional context to provide to the AI
        use_cache: Whether to use cached screenshots (default: True)
        conversation_id: Optional conversation ID to maintain context

    Returns:
        AI's response to the question about the current screen view
    """
    try:
        # Get managers
        cache_manager = get_cache_manager()
        conversation_manager = get_conversation_manager()
        metrics_manager = get_metrics_manager()

        # Increment counter
        metrics_manager.increment_counter("analyses")

        # Generate cache key for screenshot
        screenshot_cache_key = f"current_view_{int(time.time() // 60)}"  # Cache for 1 minute

        screenshot_base64 = None
        if use_cache:
            screenshot_base64 = cache_manager.get("screenshots", screenshot_cache_key)

        if not screenshot_base64:
            # Capture new screenshot
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[0])
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                # Convert to base64
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                screenshot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Cache the screenshot
                if use_cache:
                    cache_manager.set("screenshots", screenshot_cache_key, screenshot_base64, ttl=300)  # 5 minutes

        # Get conversation context if conversation_id provided
        conversation_context = ""
        if conversation_id:
            context_data = conversation_manager.get_conversation_context(conversation_id)
            if "conversation_summary" in context_data:
                conversation_context = f"\n\nConversation Context:\n{context_data['conversation_summary']}"

        # Build prompt
        full_prompt = f"Question about current screen view: {question}"
        if context:
            full_prompt += f"\n\nAdditional context: {context}"
        full_prompt += conversation_context

        # Check OpenAI provider
        if not openai_provider:
            return "OpenAI sağlayıcısı yapılandırılmamış veya API Anahtarı eksik. Lütfen API anahtarını ayarlayın."

        # Analyze with AI
        start_time = time.time()
        analysis = await openai_provider.analyze_image(
            screenshot_base64,
            full_prompt,
            model=DEFAULT_OPENAI_MODEL,
            max_tokens=1000
        )
        response_time = time.time() - start_time

        # Update provider status
        metrics_manager.update_provider_status("openai", "active", response_time)

        # Add to conversation if conversation_id provided
        if conversation_id:
            conversation_manager.add_message(
                conversation_id,
                "user",
                question,
                context_data={"context": context},
                screenshot_id=screenshot_cache_key
            )
            conversation_manager.add_message(
                conversation_id,
                "assistant",
                analysis,
                context_data={"response_time": response_time}
            )

        logger.info("Vision query completed",
                   question=question[:50] + "..." if len(question) > 50 else question,
                   conversation_id=conversation_id,
                   response_time=response_time,
                   use_cache=use_cache)

        return analysis

    except Exception as e:
        logger.error("Vision query failed", error=str(e))
        return f"Vision query failed: {str(e)}"

@mcp.tool()
async def get_system_metrics() -> str:
    """
    REVOLUTIONARY FEATURE: Get comprehensive system metrics and health status.

    Returns real-time system health, performance metrics, cache statistics,
    provider status, and performance insights.

    Returns:
        Comprehensive system metrics report in formatted text
    """
    try:
        metrics_manager = get_metrics_manager()

        # Get individual components to avoid errors
        try:
            system_health = metrics_manager.get_system_health()
        except Exception:
            system_health = {"status": "no_data", "message": "System health data unavailable"}

        try:
            performance_metrics = metrics_manager.get_performance_metrics()
        except Exception:
            performance_metrics = {"error": "Performance metrics unavailable"}

        try:
            cache_stats = get_cache_manager().get_stats()
        except Exception:
            cache_stats = {"error": "Cache stats unavailable"}

        # Format response
        response = "## 📊 System Metrics Dashboard\n\n"

        # System Health
        health = system_health
        status_emoji = "🟢" if health["status"] == "healthy" else "🟡" if health["status"] == "warning" else "🔴"
        response += f"### {status_emoji} System Health: {health['status'].upper()}\n"

        if health.get("warnings"):
            response += "**Warnings:**\n"
            for warning in health["warnings"]:
                response += f"- ⚠️ {warning}\n"

        # Current metrics
        if "current_metrics" in health:
            metrics = health["current_metrics"]
            response += f"\n**Current Performance:**\n"
            response += f"- CPU Usage: {metrics['cpu_percent']:.1f}%\n"
            response += f"- Memory Usage: {metrics['memory_percent']:.1f}% ({metrics['memory_used_mb']:.0f}MB used)\n"
            response += f"- Disk Usage: {metrics['disk_usage_percent']:.1f}%\n"

        response += f"- Uptime: {health['uptime_seconds']:.0f} seconds\n"
        response += f"- Monitoring Active: {'✅' if health['monitoring_active'] else '❌'}\n\n"

        # Performance Metrics
        perf = performance_metrics
        response += "### 🚀 Performance Metrics\n"
        response += f"- Uptime: {perf['uptime_hours']} hours\n"

        counters = perf["performance_counters"]
        response += f"- Screenshots: {counters['total_screenshots']} ({counters['screenshots_per_hour']:.1f}/hour)\n"
        response += f"- AI Analyses: {counters['total_analyses']} ({counters['analyses_per_hour']:.1f}/hour)\n"
        response += f"- UI Detections: {counters['total_ui_detections']}\n"
        response += f"- Smart Clicks: {counters['total_smart_clicks']}\n\n"

        # Cache Performance
        cache = perf["cache_performance"]
        response += "### 💾 Cache Performance\n"
        response += f"- Hit Rate: {cache['hit_rate']}% (Target: >80%)\n"
        response += f"- Total Hits: {cache['hits']}\n"
        response += f"- Total Misses: {cache['misses']}\n"
        response += f"- Memory Usage: {cache['memory_usage_mb']}MB\n"
        response += f"- Disk Usage: {cache['disk_usage_mb']}MB\n"
        response += f"- Active Entries: {cache['memory_entries']}\n\n"

        # Provider Status
        if perf["provider_status"]:
            response += "### 🤖 AI Provider Status\n"
            for name, status in perf["provider_status"].items():
                status_emoji = "🟢" if status["status"] == "active" else "🔴"
                response += f"- {status_emoji} {name.upper()}: {status['total_requests']} requests\n"
                response += f"  - Error Rate: {status['error_rate']:.1f}%\n"
                response += f"  - Avg Response: {status['avg_response_time']:.2f}s\n"

        # Cache Statistics
        response += f"\n### 💾 Cache Performance\n"
        if "error" not in cache_stats:
            response += f"- Hit Rate: {cache_stats.get('hit_rate', 0)}%\n"
            response += f"- Memory Usage: {cache_stats.get('memory_usage_mb', 0)} MB\n"
            response += f"- Total Entries: {cache_stats.get('total_entries', 0)}\n"
        else:
            response += f"- Status: {cache_stats['error']}\n"

        response += f"\n*Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

        logger.info("System metrics report generated")
        return response

    except Exception as e:
        logger.error("System metrics retrieval failed", error=str(e))
        return f"System metrics retrieval failed: {str(e)}"

@mcp.tool()
async def get_cache_stats() -> str:
    """
    Get detailed cache performance statistics.

    Returns:
        Detailed cache statistics and performance metrics
    """
    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_stats()

        response = "## 💾 Cache Statistics\n\n"
        response += f"**Performance:**\n"
        response += f"- Hit Rate: {stats['hit_rate']}% (Target: >80%)\n"
        response += f"- Total Hits: {stats['hits']}\n"
        response += f"- Total Misses: {stats['misses']}\n"
        response += f"- Evictions: {stats['evictions']}\n\n"

        response += f"**Memory Usage:**\n"
        response += f"- Active Entries: {stats['memory_entries']}\n"
        response += f"- Memory Usage: {stats['memory_usage_mb']} MB\n"
        response += f"- Disk Usage: {stats['disk_usage_mb']} MB\n\n"

        # Performance assessment
        if stats['hit_rate'] >= 80:
            response += "✅ **Cache Performance: EXCELLENT**\n"
        elif stats['hit_rate'] >= 60:
            response += "🟡 **Cache Performance: GOOD**\n"
        else:
            response += "🔴 **Cache Performance: NEEDS IMPROVEMENT**\n"

        return response

    except Exception as e:
        logger.error("Cache stats retrieval failed", error=str(e))
        return f"Cache stats retrieval failed: {str(e)}"

@mcp.tool()
async def clear_cache(namespace: Optional[str] = None) -> str:
    """
    Clear cache entries.

    Args:
        namespace: Optional namespace to clear (if None, clears all cache)

    Returns:
        Cache clearing result
    """
    try:
        cache_manager = get_cache_manager()

        if namespace:
            cache_manager.clear(namespace)
            message = f"Cache cleared for namespace: {namespace}"
        else:
            cache_manager.clear()
            message = "All cache entries cleared"

        logger.info("Cache cleared", namespace=namespace)
        return f"✅ {message}"

    except Exception as e:
        logger.error("Cache clearing failed", error=str(e))
        return f"Cache clearing failed: {str(e)}"

# === PHASE 2 & 3 ENHANCED FEATURES ===

@mcp.tool()
async def get_batch_processor_stats() -> str:
    """
    Get batch processor statistics and performance metrics.

    Returns:
        Detailed batch processing statistics
    """
    try:
        batch_processor = get_batch_processor()
        stats = batch_processor.get_statistics()

        response = "## 🔄 Batch Processor Statistics\n\n"
        response += f"**Performance:**\n"
        response += f"- Total Requests: {stats['total_requests']}\n"
        response += f"- Total Batches: {stats['total_batches']}\n"
        response += f"- Average Batch Size: {stats['average_batch_size']}\n"
        response += f"- Average Processing Time: {stats['average_processing_time']}s\n\n"

        response += f"**Current Status:**\n"
        response += f"- Pending Requests: {stats['pending_requests']}\n"
        response += f"- Processing Batches: {stats['processing_batches']}\n"
        response += f"- Completed Results: {stats['completed_results']}\n\n"

        response += f"**Configuration:**\n"
        config = stats['configuration']
        response += f"- Max Batch Size: {config['max_batch_size']}\n"
        response += f"- Max Wait Time: {config['max_wait_time']}s\n"
        response += f"- Max Concurrent Batches: {config['max_concurrent_batches']}\n"
        response += f"- Priority Queue: {'✅ Enabled' if config['priority_queue_enabled'] else '❌ Disabled'}\n"

        return response

    except Exception as e:
        logger.error(f"Batch processor stats failed: {str(e)}")
        return f"❌ Batch processor stats failed: {str(e)}"

@mcp.tool()
async def optimize_image(
    image_data: str,
    preset: str = "web",
    target_size_kb: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    quality: int = 85
) -> str:
    """
    Optimize an image with advanced compression and quality control.

    Args:
        image_data: Base64 encoded image data
        preset: Optimization preset (web, thumbnail, high_quality, minimal_size)
        target_size_kb: Target file size in KB
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        quality: JPEG quality (1-100)

    Returns:
        Optimization result with statistics
    """
    try:
        image_optimizer = get_image_optimizer()

        if preset in image_optimizer.optimization_presets:
            result = image_optimizer.optimize_with_preset(image_data, preset)
        else:
            config = OptimizationConfig(
                target_size_kb=target_size_kb,
                max_width=max_width,
                max_height=max_height,
                quality=quality
            )
            result = image_optimizer.optimize_image(image_data, config)

        if result.success:
            response = "## 🖼️ Image Optimization Complete\n\n"
            response += f"**Results:**\n"
            response += f"- Original Size: {result.original_size_bytes:,} bytes ({result.original_size_bytes/1024:.1f} KB)\n"
            response += f"- Optimized Size: {result.optimized_size_bytes:,} bytes ({result.optimized_size_bytes/1024:.1f} KB)\n"
            response += f"- Compression Ratio: {result.compression_ratio:.1f}%\n"
            response += f"- Processing Time: {result.processing_time:.2f}s\n\n"

            response += f"**Changes:**\n"
            response += f"- Original Dimensions: {result.original_dimensions[0]}x{result.original_dimensions[1]}\n"
            response += f"- Optimized Dimensions: {result.optimized_dimensions[0]}x{result.optimized_dimensions[1]}\n"
            response += f"- Format Changed: {'✅ Yes' if result.format_changed else '❌ No'}\n"
            response += f"- Dimensions Changed: {'✅ Yes' if result.dimensions_changed else '❌ No'}\n"

            return response
        else:
            return f"❌ Image optimization failed: {result.error}"

    except Exception as e:
        logger.error(f"Image optimization failed: {str(e)}")
        return f"❌ Image optimization failed: {str(e)}"

@mcp.tool()
async def get_error_recovery_stats() -> str:
    """
    Get error recovery system statistics and health status.

    Returns:
        Comprehensive error recovery statistics
    """
    try:
        error_manager = get_error_recovery_manager()
        stats = error_manager.get_error_statistics()

        response = "## 🛡️ Error Recovery Statistics\n\n"
        response += f"**Overall Performance:**\n"
        response += f"- Total Errors: {stats['total_errors']}\n"
        response += f"- Recovered Errors: {stats['recovered_errors']}\n"
        response += f"- Critical Errors: {stats['critical_errors']}\n"
        response += f"- Emergency Stops: {stats['emergency_stops']}\n"
        response += f"- Recovery Rate: {stats['recovery_rate']:.1f}%\n\n"

        response += f"**Recent Activity:**\n"
        response += f"- Recent Errors (1h): {stats['recent_errors_count']}\n\n"

        response += f"**Component Statistics:**\n"
        for component, comp_stats in stats['component_statistics'].items():
            response += f"- **{component.title()}:**\n"
            response += f"  - Total: {comp_stats['total_errors']}\n"
            response += f"  - Recovered: {comp_stats['recovered_errors']}\n"
            response += f"  - Critical: {comp_stats['critical_errors']}\n"

        response += f"\n**System Health:**\n"
        response += f"- Registered Fallbacks: {len(stats['registered_fallbacks'])}\n"
        response += f"- Health Checks: {len(stats['registered_health_checks'])}\n"

        if stats['recovery_rate'] >= 80:
            response += "\n🟢 **System Health: EXCELLENT**"
        elif stats['recovery_rate'] >= 60:
            response += "\n🟡 **System Health: GOOD**"
        else:
            response += "\n🔴 **System Health: NEEDS ATTENTION**"

        return response

    except Exception as e:
        logger.error(f"Error recovery stats failed: {str(e)}")
        return f"❌ Error recovery stats failed: {str(e)}"

@mcp.tool()
async def get_platform_info() -> str:
    """
    Get comprehensive platform and system information.

    Returns:
        Detailed platform information and capabilities
    """
    try:
        platform_manager = get_platform_manager()
        platform_info = platform_manager.get_platform_info()

        response = "## 🖥️ Platform Information\n\n"
        response += f"**System Details:**\n"
        response += f"- Platform: {platform_info['detected_platform'].title()}\n"
        response += f"- System: {platform_info['system']}\n"
        response += f"- Release: {platform_info['release']}\n"
        response += f"- Machine: {platform_info['machine']}\n"
        response += f"- Processor: {platform_info['processor']}\n"
        response += f"- Node: {platform_info['node']}\n\n"

        response += f"**Python Environment:**\n"
        response += f"- Python Version: {platform_info['python_version']}\n"
        response += f"- Architecture: {platform_info['architecture'][0]} ({platform_info['architecture'][1]})\n\n"

        response += f"**Available Features:**\n"
        features = platform_info.get('available_features', [])
        if features:
            for feature in features:
                response += f"- ✅ {feature.replace('_', ' ').title()}\n"
        else:
            response += "- ❌ No platform-specific features available\n"

        response += f"\n**Feature Availability:**\n"
        feature_checks = [
            ("window_management", "Window Management"),
            ("screen_capture", "Screen Capture"),
            ("mouse_control", "Mouse Control"),
            ("keyboard_input", "Keyboard Input")
        ]

        for feature_key, feature_name in feature_checks:
            available = platform_manager.is_feature_available(feature_key)
            status = "✅ Available" if available else "❌ Not Available"
            response += f"- {feature_name}: {status}\n"

        return response

    except Exception as e:
        logger.error(f"Platform info failed: {str(e)}")
        return f"❌ Platform info failed: {str(e)}"

@mcp.tool()
async def simulate_input(
    input_type: str,
    text: Optional[str] = None,
    keys: Optional[List[str]] = None,
    modifiers: Optional[List[str]] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    button: str = "left",
    clicks: int = 1
) -> str:
    """
    Simulate keyboard or mouse input with advanced capabilities.

    Args:
        input_type: Type of input ('keyboard', 'mouse', 'hotkey', 'text')
        text: Text to type (for text input)
        keys: List of keys to press (for keyboard/hotkey input)
        modifiers: List of modifier keys (ctrl, alt, shift, cmd)
        x: X coordinate (for mouse input)
        y: Y coordinate (for mouse input)
        button: Mouse button (left, right, middle)
        clicks: Number of clicks (for mouse input)

    Returns:
        Input simulation result
    """
    try:
        input_simulator = get_input_simulator()

        if input_type == "text" and text:
            success = input_simulator.simulate_text_input(text)
            return f"✅ Text input simulated: '{text}'" if success else "❌ Text input failed"

        elif input_type == "keyboard" and keys:
            config = KeyboardInput(keys=keys, modifiers=modifiers or [])
            success = input_simulator.simulate_keyboard_input(config)
            return f"✅ Keyboard input simulated: {'+'.join((modifiers or []) + keys)}" if success else "❌ Keyboard input failed"

        elif input_type == "hotkey" and keys:
            success = input_simulator.simulate_hotkey(keys, modifiers or [])
            return f"✅ Hotkey simulated: {'+'.join((modifiers or []) + keys)}" if success else "❌ Hotkey failed"

        elif input_type == "mouse" and x is not None and y is not None:
            mouse_button = MouseButton.LEFT
            if button.lower() == "right":
                mouse_button = MouseButton.RIGHT
            elif button.lower() == "middle":
                mouse_button = MouseButton.MIDDLE

            success = input_simulator.simulate_click(x, y, mouse_button, clicks)
            return f"✅ Mouse click simulated at ({x}, {y})" if success else "❌ Mouse click failed"

        else:
            return "❌ Invalid input parameters. Please specify valid input_type and required parameters."

    except Exception as e:
        logger.error(f"Input simulation failed: {str(e)}")
        return f"❌ Input simulation failed: {str(e)}"

@mcp.tool()
async def get_input_capabilities() -> str:
    """
    Get input simulation capabilities and backend information.

    Returns:
        Input simulation capabilities and status
    """
    try:
        input_simulator = get_input_simulator()
        capabilities = input_simulator.get_capabilities()

        response = "## ⌨️ Input Simulation Capabilities\n\n"
        response += f"**Active Backend:** {capabilities['active_backend']}\n\n"

        response += f"**Available Backends:**\n"
        for backend, available in capabilities['available_backends'].items():
            status = "✅ Available" if available else "❌ Not Available"
            response += f"- {backend}: {status}\n"

        response += f"\n**Supported Features:**\n"
        features = capabilities['supported_features']
        for feature, supported in features.items():
            status = "✅ Supported" if supported else "❌ Not Supported"
            feature_name = feature.replace('_', ' ').title()
            response += f"- {feature_name}: {status}\n"

        response += f"\n**Statistics:**\n"
        response += f"- Input History Size: {capabilities['input_history_size']}\n"
        response += f"- Max History: {capabilities['max_history']}\n"

        return response

    except Exception as e:
        logger.error(f"Input capabilities failed: {str(e)}")
        return f"❌ Input capabilities failed: {str(e)}"

# Running the server
if __name__ == "__main__":
    # Windows Unicode encoding fix
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("=== REVOLUTIONARY SCREEN MONITOR MCP SERVER ===")
    print("Revolutionary features that give AI real-time vision capabilities!")
    print()

    if API_KEY:
        logger.info("Server is SECURED", api_key_preview=f"{API_KEY[:4]}...{API_KEY[-4:]}")
    else:
        logger.warning("Server is UNSECURED. No API Key provided.")

    print("Revolutionary Features:")
    print("   - Smart Monitoring - Intelligent trigger-based monitoring")
    print("   - Event Detection - Automatic change and error detection")
    print("   - Smart Click System - Natural language clicking")
    print("   - OCR Text Extraction - Screen text reading")
    print("   - Application Monitoring - Multi-application context awareness")
    print("   - AI Analysis - Intelligent event analysis")
    print("   - 🆕 Advanced Cache System - TTL-based caching with metrics")
    print("   - 🆕 Conversation Context - Chat history and context preservation")
    print("   - 🆕 System Metrics - Real-time health monitoring")
    print()

    print("🚀 REVOLUTIONARY MCP TOOLS:")
    print("   🧠 Smart Monitoring:")
    print("      * start_smart_monitoring() - Intelligent monitoring with triggers")
    print("      * get_monitoring_insights() - AI-powered insights")
    print("      * get_recent_events() - Event history")
    print("      * get_monitoring_summary() - Comprehensive reports")
    print()
    print("   🎯 UI Intelligence:")
    print("      * smart_click() - Click with natural language")
    print("      * extract_text_from_screen() - Extract text from screen")
    print()
    print("   📊 Core Features:")
    print("      * capture_and_analyze() - AI-powered screenshot analysis (with cache)")
    print("      * record_and_analyze() - AI-powered video recording and analysis")
    print("      * get_active_application() - Get active app context")
    print("      * list_tools() - Complete tool documentation")
    print()
    print("   🆕 Enhanced Features:")
    print("      * query_vision_about_current_view() - Ask AI about current screen")
    print("      * get_system_metrics() - Comprehensive system health dashboard")
    print("      * get_cache_stats() - Cache performance statistics")
    print("      * clear_cache() - Cache management")
    print()
    print("   🚀 Phase 2 & 3 Features:")
    print("      * get_batch_processor_stats() - Batch processing statistics")
    print("      * optimize_image() - Advanced image optimization")
    print("      * get_error_recovery_stats() - Error recovery system status")
    print("      * get_platform_info() - Cross-platform system information")
    print("      * simulate_input() - Advanced input simulation")
    print("      * get_input_capabilities() - Input system capabilities")
    print()

    logger.info("Starting Revolutionary MCP Server")
    print("🔥 Server starting with Smart Monitoring capability...")
    print("🎯 AI now has enhanced vision and smart interaction!")
    mcp.run(transport='stdio')
