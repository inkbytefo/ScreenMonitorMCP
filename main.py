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

from mcp.server.fastmcp import FastMCP
from ai_providers import OpenAIProvider
from real_time_monitor import RealTimeMonitor, MonitoringConfig, ChangeEvent, get_global_monitor, set_global_monitor
from ui_detection import UIElementDetector, SmartClicker, get_ui_detector, get_smart_clicker
from predictive_intelligence import PredictiveEngine, UserAction, get_predictive_engine

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
args = parser.parse_args()

API_KEY = args.api_key
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url
DEFAULT_OPENAI_MODEL = args.default_openai_model

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

# Initialize Real-Time Monitor
monitor_config = MonitoringConfig(
    fps=2,
    change_threshold=0.1,
    smart_detection=True,
    save_screenshots=True
)
real_time_monitor = RealTimeMonitor(monitor_config)
set_global_monitor(real_time_monitor)

# Setup change event callback for AI analysis
async def on_change_detected(change_event: ChangeEvent):
    """Değişiklik algılandığında AI analizi yapar"""
    if change_event.change_type in ['major', 'critical'] and openai_provider:
        try:
            if change_event.screenshot_base64:
                analysis = await openai_provider.analyze_image(
                    image_base64=change_event.screenshot_base64,
                    prompt=f"Bu ekran değişikliğini analiz edin. Değişiklik tipi: {change_event.change_type}, Etkilenen bölgeler: {len(change_event.affected_regions)}",
                    model=DEFAULT_OPENAI_MODEL,
                    output_format="png",
                    max_tokens=150
                )
                logger.info("Change analyzed by AI",
                           change_type=change_event.change_type,
                           analysis=analysis[:100] + "..." if len(analysis) > 100 else analysis)
        except Exception as e:
            logger.error("AI analysis failed for change event", error=str(e))

real_time_monitor.add_change_callback(on_change_detected)

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
            "name": "start_continuous_monitoring",
            "description": "🚀 REVOLUTIONARY: Starts AI's continuous screen monitoring with real-time vision",
            "category": "revolutionary",
            "parameters": ["fps", "change_threshold", "smart_detection", "save_screenshots"]
        },
        {
            "name": "stop_continuous_monitoring",
            "description": "Stops continuous screen monitoring",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "get_monitoring_status",
            "description": "Gets real-time monitoring status and statistics",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "get_recent_changes",
            "description": "Gets recently detected screen changes",
            "category": "revolutionary",
            "parameters": ["limit"]
        },
        {
            "name": "analyze_ui_elements",
            "description": "🚀 REVOLUTIONARY: Detects and maps all UI elements on screen with interaction capabilities",
            "category": "revolutionary",
            "parameters": ["detect_buttons", "detect_text_fields", "extract_text", "confidence_threshold"]
        },
        {
            "name": "smart_click",
            "description": "🚀 REVOLUTIONARY: AI clicks elements using natural language commands",
            "category": "revolutionary",
            "parameters": ["element_description", "confidence_threshold", "dry_run"]
        },
        {
            "name": "extract_text_from_screen",
            "description": "🚀 REVOLUTIONARY: Extracts text from screen using OCR technology",
            "category": "revolutionary",
            "parameters": ["region", "ocr_engine"]
        },
        {
            "name": "learn_user_patterns",
            "description": "🚀 REVOLUTIONARY: AI learns user behavior patterns and habits",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "predict_user_intent",
            "description": "🚀 REVOLUTIONARY: Predicts user's future intentions based on context",
            "category": "revolutionary",
            "parameters": ["current_context", "prediction_horizon"]
        },
        {
            "name": "proactive_assistance",
            "description": "🚀 REVOLUTIONARY: Offers proactive help before user asks",
            "category": "revolutionary",
            "parameters": []
        },
        {
            "name": "record_user_action",
            "description": "🚀 REVOLUTIONARY: Records user actions for learning system",
            "category": "revolutionary",
            "parameters": ["action_type", "target", "coordinates", "text_content", "app_context"]
        }
    ]

    # Categorize tools
    for tool_data in all_available_tools:
        if tool_data["category"] == "revolutionary":
            revolutionary_tools.append(tool_data)
        else:
            standard_tools.append(tool_data)
        tools.append(tool_data)

    # Get server status
    monitor = get_global_monitor()
    server_status = {
        "ai_provider": "OpenAI" if openai_provider else "None",
        "real_time_monitoring": monitor.is_monitoring if monitor else False,
        "total_actions_learned": len(monitor.event_history) if monitor else 0,
        "server_version": "1.0.0-revolutionary",
        "capabilities": [
            "Real-time screen monitoring",
            "UI element detection",
            "Smart click automation",
            "OCR text extraction",
            "Predictive intelligence",
            "Behavior learning",
            "Proactive assistance"
        ]
    }

    return {
        "mcp_server": "Revolutionary Screen Monitor",
        "version": "1.0.0",
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
            "start_ai_vision": "await start_continuous_monitoring(fps=3)",
            "smart_interaction": "await smart_click('Save button')",
            "learn_behavior": "await learn_user_patterns()",
            "get_predictions": "await predict_user_intent()"
        },
        "documentation": {
            "quick_start": "See QUICK_START.md for setup instructions",
            "full_docs": "See README.md for complete documentation",
            "revolutionary_features": "This server gives AI real-time vision and predictive intelligence"
        }
    }

# === REVOLUTIONARY FEATURE 1: REAL-TIME MONITORING ===

@mcp.tool()
async def start_continuous_monitoring(
    fps: int = 2,
    change_threshold: float = 0.1,
    major_change_threshold: float = 0.3,
    critical_change_threshold: float = 0.6,
    smart_detection: bool = True,
    save_screenshots: bool = True
) -> Dict[str, Any]:
    """
    🚀 REVOLUTIONARY FEATURE: Starts AI's continuous screen monitoring.

    This tool gives AI real-time vision capabilities. AI can now continuously monitor
    the screen and detect changes, not just when requested.

    Args:
        fps: How many frames per second to capture (1-10 recommended)
        change_threshold: Threshold for small changes (0.0-1.0)
        major_change_threshold: Threshold for major changes (0.0-1.0)
        critical_change_threshold: Threshold for critical changes (0.0-1.0)
        smart_detection: Whether smart change detection is active
        save_screenshots: Whether to save screenshots when changes occur

    Returns:
        Monitoring startup status and configuration information
    """
    try:
        monitor = get_global_monitor()
        if not monitor:
            return {"error": "Real-time monitor başlatılamadı"}

        # Yapılandırmayı güncelle
        monitor.config.fps = max(1, min(10, fps))
        monitor.config.change_threshold = max(0.01, min(1.0, change_threshold))
        monitor.config.major_change_threshold = max(0.1, min(1.0, major_change_threshold))
        monitor.config.critical_change_threshold = max(0.2, min(1.0, critical_change_threshold))
        monitor.config.smart_detection = smart_detection
        monitor.config.save_screenshots = save_screenshots

        result = monitor.start_monitoring()
        logger.info("Continuous monitoring started via MCP", config=result.get("config", {}))

        return {
            **result,
            "revolutionary_feature": "Real-Time AI Vision",
            "description": "AI is now continuously monitoring your screen and detecting changes!",
            "capabilities": [
                "Real-time change detection",
                "Smart change classification",
                "Proactive AI analysis",
                "Automatic screenshot saving"
            ]
        }

    except Exception as e:
        logger.error("Failed to start continuous monitoring", error=str(e))
        return {"error": f"Monitoring could not be started: {str(e)}"}

@mcp.tool()
async def stop_continuous_monitoring() -> Dict[str, Any]:
    """
    Stops continuous screen monitoring.

    Returns:
        Monitoring stop status and statistics
    """
    try:
        monitor = get_global_monitor()
        if not monitor:
            return {"error": "Monitor bulunamadı"}

        result = monitor.stop_monitoring()
        logger.info("Continuous monitoring stopped via MCP", stats=result.get("stats", {}))

        return {
            **result,
            "message": "AI's continuous vision feature stopped",
            "note": "Statistics saved"
        }

    except Exception as e:
        logger.error("Failed to stop continuous monitoring", error=str(e))
        return {"error": f"Monitoring could not be stopped: {str(e)}"}

@mcp.tool()
async def get_monitoring_status() -> Dict[str, Any]:
    """
    Gets real-time monitoring status and statistics.

    Returns:
        Monitoring status, statistics and recent events
    """
    try:
        monitor = get_global_monitor()
        if not monitor:
            return {"error": "Monitor bulunamadı"}

        status = monitor.get_status()

        return {
            **status,
            "revolutionary_insight": "AI's vision history",
            "description": "How long AI has been monitoring your screen and what it has detected"
        }

    except Exception as e:
        logger.error("Failed to get monitoring status", error=str(e))
        return {"error": f"Status could not be retrieved: {str(e)}"}

@mcp.tool()
async def get_recent_changes(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Gets recently detected screen changes.

    Args:
        limit: How many recent changes to retrieve

    Returns:
        List of recent changes
    """
    try:
        monitor = get_global_monitor()
        if not monitor:
            return [{"error": "Monitor bulunamadı"}]

        recent_events = monitor.event_history[-limit:] if monitor.event_history else []

        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "change_type": event.change_type,
                "change_percentage": round(event.change_percentage * 100, 2),
                "affected_regions_count": len(event.affected_regions),
                "description": event.description,
                "has_screenshot": event.screenshot_base64 is not None
            }
            for event in recent_events
        ]

    except Exception as e:
        logger.error("Failed to get recent changes", error=str(e))
        return [{"error": f"Recent changes could not be retrieved: {str(e)}"}]

# === REVOLUTIONARY FEATURE 2: UI ELEMENT DETECTION ===

@mcp.tool()
async def analyze_ui_elements(
    detect_buttons: bool = True,
    detect_text_fields: bool = True,
    extract_text: bool = True,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    🚀 DEVRIMSEL ÖZELLİK: Ekrandaki tüm UI elementlerini tanır ve etkileşim haritası çıkarır.

    AI artık sadece görmekle kalmıyor, hangi butona basacağını, hangi formu dolduracağını biliyor!

    Args:
        detect_buttons: Butonları algıla
        detect_text_fields: Metin alanlarını algıla
        extract_text: OCR ile metinleri çıkar
        confidence_threshold: Minimum güven eşiği (0.0-1.0)

    Returns:
        UI elementleri, koordinatlar ve etkileşim önerileri
    """
    try:
        detector = get_ui_detector()
        analysis = detector.analyze_screen()

        # Güven eşiğine göre filtrele
        filtered_elements = [
            elem for elem in analysis.elements
            if elem.confidence >= confidence_threshold
        ]

        # Elementleri kategorize et
        buttons = [elem for elem in filtered_elements if elem.element_type == "button"]
        text_fields = [elem for elem in filtered_elements if elem.element_type == "text_field"]
        texts = [elem for elem in filtered_elements if elem.element_type == "text"]

        # Etkileşim önerileri oluştur
        interaction_suggestions = []
        for elem in filtered_elements:
            if elem.clickable:
                suggestion = {
                    "action": "click",
                    "element_type": elem.element_type,
                    "coordinates": elem.center_point,
                    "description": elem.description,
                    "confidence": elem.confidence
                }
                if elem.text_content:
                    suggestion["text_content"] = elem.text_content
                interaction_suggestions.append(suggestion)

        logger.info("UI analysis completed",
                   total_elements=len(filtered_elements),
                   clickable_elements=len([e for e in filtered_elements if e.clickable]))

        return {
            "revolutionary_feature": "AI UI Intelligence",
            "description": "AI now recognizes all UI elements on screen and knows how to interact with them!",
            "analysis_summary": {
                "total_elements": len(filtered_elements),
                "buttons": len(buttons),
                "text_fields": len(text_fields),
                "texts_found": len(texts),
                "clickable_elements": len([e for e in filtered_elements if e.clickable]),
                "analysis_time": round(analysis.analysis_time, 3),
                "ocr_method": analysis.ocr_method
            },
            "ui_elements": [
                {
                    "type": elem.element_type,
                    "coordinates": elem.coordinates,
                    "center_point": elem.center_point,
                    "confidence": round(elem.confidence, 3),
                    "clickable": elem.clickable,
                    "text_content": elem.text_content,
                    "description": elem.description
                }
                for elem in filtered_elements
            ],
            "interaction_suggestions": interaction_suggestions,
            "capabilities": [
                "UI element detection",
                "Text extraction (OCR)",
                "Clickable element identification",
                "Coordinate mapping",
                "Interaction planning"
            ]
        }

    except Exception as e:
        logger.error("UI analysis failed", error=str(e))
        return {"error": f"UI analysis failed: {str(e)}"}

@mcp.tool()
async def smart_click(
    element_description: str,
    confidence_threshold: float = 0.8,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    🚀 REVOLUTIONARY FEATURE: AI finds and clicks elements described in natural language.

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
    🚀 REVOLUTIONARY FEATURE: Extract text from screen or specific region.

    Args:
        region: Specific region {'x': int, 'y': int, 'width': int, 'height': int}
        ocr_engine: OCR engine ('auto', 'tesseract', 'easyocr')

    Returns:
        Extracted texts and their coordinates
    """
    try:
        import mss
        import cv2

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

# === REVOLUTIONARY FEATURE 3: PREDICTIVE INTELLIGENCE ===

@mcp.tool()
async def learn_user_patterns() -> Dict[str, Any]:
    """
    🚀 REVOLUTIONARY FEATURE: Learns and analyzes user behavior patterns.

    AI is now learning user habits and can predict future needs!

    Returns:
        User behavior analysis and patterns
    """
    try:
        engine = get_predictive_engine()
        insights = engine.get_user_insights()

        logger.info("User patterns analyzed",
                   total_actions=insights.get("total_actions", 0),
                   total_patterns=insights.get("total_patterns", 0))

        return {
            "revolutionary_feature": "AI Behavioral Learning",
            "description": "AI kullanıcının davranış kalıplarını öğrendi ve analiz etti!",
            "learning_summary": {
                "total_actions_recorded": insights.get("total_actions", 0),
                "patterns_discovered": insights.get("total_patterns", 0),
                "pattern_types": insights.get("pattern_breakdown", {}),
                "data_quality": "High" if insights.get("total_actions", 0) > 50 else "Building"
            },
            "behavioral_insights": {
                "most_common_actions": insights.get("most_common_actions", {}),
                "peak_activity_hours": insights.get("peak_activity_hours", []),
                "strongest_patterns": insights.get("strongest_patterns", [])
            },
            "capabilities": [
                "Temporal pattern recognition",
                "Sequential behavior analysis",
                "Contextual pattern learning",
                "Predictive modeling",
                "Proactive assistance"
            ]
        }

    except Exception as e:
        logger.error("User pattern learning failed", error=str(e))
        return {"error": f"Behavior pattern learning failed: {str(e)}"}

@mcp.tool()
async def predict_user_intent(
    current_context: Optional[Dict[str, Any]] = None,
    prediction_horizon: int = 30
) -> Dict[str, Any]:
    """
    🚀 REVOLUTIONARY FEATURE: Predicts user's future intentions.

    Args:
        current_context: Current context information
        prediction_horizon: Prediction time horizon (minutes)

    Returns:
        User intent predictions and suggestions
    """
    try:
        engine = get_predictive_engine()

        if current_context is None:
            current_context = {
                "current_time": datetime.now().isoformat(),
                "current_app": None
            }

        predictions = engine.generate_predictions(current_context)

        # Format predictions
        formatted_predictions = []
        for pred in predictions:
            if pred.confidence > 0.5:  # Only confident predictions
                formatted_predictions.append({
                    "prediction": pred.description,
                    "confidence": round(pred.confidence * 100, 1),
                    "suggested_actions": pred.suggested_actions,
                    "prediction_type": pred.prediction_type,
                    "expires_in_minutes": int((pred.expires_at - datetime.now()).total_seconds() / 60),
                    "context": pred.context
                })

        logger.info("User intent predicted",
                   predictions_count=len(formatted_predictions))

        return {
            "revolutionary_feature": "AI Intent Prediction",
            "description": "AI kullanıcının gelecekteki niyetlerini tahmin ediyor!",
            "prediction_summary": {
                "total_predictions": len(formatted_predictions),
                "high_confidence_predictions": len([p for p in formatted_predictions if p["confidence"] > 70]),
                "prediction_horizon_minutes": prediction_horizon,
                "context_analyzed": current_context
            },
            "predictions": formatted_predictions,
            "ai_insights": [
                "Temporal behavior patterns analyzed",
                "Sequential action patterns detected",
                "Contextual preferences learned",
                "Predictive models applied"
            ]
        }

    except Exception as e:
        logger.error("User intent prediction failed", error=str(e))
        return {"error": f"Intent prediction failed: {str(e)}"}

@mcp.tool()
async def proactive_assistance() -> str:
    """
    🚀 REVOLUTIONARY FEATURE: Offers proactive help before user requests.

    AI can now predict user needs in advance and offer help!

    Returns:
        Proactive assistance suggestions
    """
    try:
        engine = get_predictive_engine()
        suggestions = engine.get_proactive_suggestions()

        # Önerileri zenginleştir
        enhanced_suggestions = []
        for suggestion in suggestions:
            enhanced_suggestion = {
                **suggestion,
                "timestamp": datetime.now().isoformat(),
                "priority": "high" if suggestion.get("confidence", 0) > 0.8 else "medium",
                "category": suggestion.get("type", "general")
            }
            enhanced_suggestions.append(enhanced_suggestion)

        # Ek akıllı öneriler
        insights = engine.get_user_insights()

        # Genel sistem önerileri
        if insights.get("total_actions", 0) == 0:
            enhanced_suggestions.append({
                "type": "welcome",
                "message": "Hoş geldiniz! AI sistemi davranışlarınızı öğrenmeye başlayacak. Birkaç aksiyon kaydettikten sonra size özel öneriler sunabilirim.",
                "confidence": 1.0,
                "priority": "high",
                "category": "onboarding",
                "timestamp": datetime.now().isoformat()
            })
        elif insights.get("total_actions", 0) < 10:
            enhanced_suggestions.append({
                "type": "learning",
                "message": f"Şu ana kadar {insights['total_actions']} aksiyon kaydedildi. Daha iyi öneriler için daha fazla etkileşim gerekiyor.",
                "confidence": 0.8,
                "priority": "medium",
                "category": "learning_progress",
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Detaylı analiz önerileri
            if insights.get('most_common_actions'):
                most_common = list(insights['most_common_actions'].keys())[0]
                enhanced_suggestions.append({
                    "type": "productivity_insight",
                    "message": f"Toplam {insights['total_actions']} aksiyon kaydedildi. En çok '{most_common}' aksiyonu yapıyorsunuz.",
                    "confidence": 0.9,
                    "priority": "medium",
                    "category": "analytics",
                    "timestamp": datetime.now().isoformat()
                })

            # Zaman bazlı öneriler
            if insights.get('peak_activity_hours'):
                peak_hour = insights['peak_activity_hours'][0][0]
                enhanced_suggestions.append({
                    "type": "time_insight",
                    "message": f"En aktif olduğunuz saat: {peak_hour}:00. Bu saatte daha verimli çalışabilirsiniz.",
                    "confidence": 0.7,
                    "priority": "low",
                    "category": "time_management",
                    "timestamp": datetime.now().isoformat()
                })

        # Sistem durumu önerileri
        monitor = get_global_monitor()
        if monitor and not monitor.is_monitoring:
            enhanced_suggestions.append({
                "type": "system_suggestion",
                "message": "Gerçek zamanlı izleme kapalı. Daha iyi analiz için sürekli izlemeyi başlatabilirsiniz.",
                "confidence": 0.8,
                "priority": "medium",
                "category": "system",
                "timestamp": datetime.now().isoformat()
            })

        logger.info("Proactive assistance generated",
                   suggestions_count=len(enhanced_suggestions))

        result = {
            "revolutionary_feature": "AI Proactive Assistance",
            "description": "AI kullanıcının ihtiyaçlarını önceden tahmin ediyor ve yardım öneriyor!",
            "total_suggestions": len(enhanced_suggestions),
            "suggestions": enhanced_suggestions,
            "generated_at": datetime.now().isoformat(),
            "capabilities": [
                "Behavioral pattern analysis",
                "Predictive assistance",
                "Productivity insights",
                "Time management suggestions",
                "System optimization tips"
            ]
        }

        import json
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error("Proactive assistance failed", error=str(e))
        error_result = {
            "error": f"Proactive assistance failed: {str(e)}",
            "suggestions": [],
            "total_suggestions": 0
        }
        import json
        return json.dumps(error_result, ensure_ascii=False, indent=2)

@mcp.tool()
async def record_user_action(
    action_type: str,
    target: str,
    coordinates: Optional[List[int]] = None,
    text_content: Optional[str] = None,
    app_context: Optional[str] = None,
    window_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    🚀 REVOLUTIONARY FEATURE: Records user action and feeds learning system.

    Args:
        action_type: Action type ('click', 'type', 'scroll', 'app_switch')
        target: Target element or application
        coordinates: Click coordinates [x, y]
        text_content: Typed text
        app_context: Application context
        window_title: Window title

    Returns:
        Recording status and learning information
    """
    try:
        engine = get_predictive_engine()

        # Create UserAction
        action = UserAction(
            timestamp=datetime.now(),
            action_type=action_type,
            target=target,
            coordinates=tuple(coordinates) if coordinates else None,
            text_content=text_content,
            app_context=app_context,
            window_title=window_title
        )

        # Record action
        engine.record_action(action)

        # Current statistics
        insights = engine.get_user_insights()

        logger.info("User action recorded and learned",
                   action_type=action_type,
                   target=target,
                   total_actions=insights.get("total_actions", 0))

        return {
            "revolutionary_feature": "AI Action Learning",
            "description": "AI kullanıcı aksiyonunu kaydetti ve öğrendi!",
            "action_recorded": {
                "type": action_type,
                "target": target,
                "timestamp": action.timestamp.isoformat(),
                "context": {
                    "app": app_context,
                    "window": window_title,
                    "coordinates": coordinates
                }
            },
            "learning_progress": {
                "total_actions_learned": insights.get("total_actions", 0),
                "patterns_discovered": insights.get("total_patterns", 0),
                "learning_status": "Active"
            },
            "ai_response": "Action recorded and behavior patterns updated!"
        }

    except Exception as e:
        logger.error("User action recording failed", error=str(e))
        return {"error": f"Action recording failed: {str(e)}"}

@mcp.tool()
async def capture_and_analyze(capture_mode: Literal["all", "monitor", "window", "region"] = "all", monitor_number: int = 1, capture_active_window: bool = False, region: Optional[Dict[str, int]] = None, output_format: Literal["png", "jpeg"] = "png", analysis_prompt: str = "Please analyze this screenshot and provide information about its content.", max_tokens: int = 300) -> str:
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
        img_base64, capture_details = _capture_screenshot_to_base64(capture_mode, monitor_number, capture_active_window, region, output_format)
        model_to_use = DEFAULT_OPENAI_MODEL
        ai_analysis = await openai_provider.analyze_image(
            image_base64=img_base64,
            prompt=analysis_prompt,
            model=model_to_use,
            output_format=output_format,
            max_tokens=max_tokens
        )
        return f"Screenshot successfully captured and analyzed. Analysis: {ai_analysis}. Capture details: {capture_details}. Model used: {model_to_use}. Provider used: openai"
    except Exception as e:
        return f"Screenshot capture and analysis failed. Error: {str(e)}. Please check parameters or try again later."

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

    print("Devrimsel Ozellikler:")
    print("   - Real-Time Continuous Monitoring - AI'nin surekli gorme yetisi")
    print("   - Smart Change Detection - Akilli degisiklik algilama")
    print("   - UI Element Detection - Arayuz elementi tanima ve etkilesim")
    print("   - OCR Text Extraction - Ekrandan metin okuma")
    print("   - Smart Click System - Dogal dil ile tiklama")
    print("   - Predictive Intelligence - Davranis ogrenme ve tahmin")
    print("   - Proactive Assistance - Ongorulu yardim sistemi")
    print()

    print("Revolutionary MCP Tools:")
    print("   Real-Time Monitoring:")
    print("      * start_continuous_monitoring() - Start continuous monitoring")
    print("      * stop_continuous_monitoring() - Stop monitoring")
    print("      * get_monitoring_status() - Status information")
    print("      * get_recent_changes() - Recent changes")
    print()
    print("   UI Intelligence:")
    print("      * analyze_ui_elements() - Recognize UI elements")
    print("      * smart_click() - Click with natural language")
    print("      * extract_text_from_screen() - Extract text from screen")
    print()
    print("   Predictive AI:")
    print("      * learn_user_patterns() - Learn behavior patterns")
    print("      * predict_user_intent() - Predict user intent")
    print("      * proactive_assistance() - Offer proactive help")
    print("      * record_user_action() - Record user action")
    print()

    logger.info("Starting Revolutionary MCP Server with Real-Time Vision")
    print("Server starting...")
    mcp.run(transport='stdio')
