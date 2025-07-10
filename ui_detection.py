"""
UI Element Detection System
Devrimsel özellik: AI'ya ekrandaki UI elementlerini tanıma ve etkileşim kurma yetisi
"""

import cv2
import numpy as np
import mss
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import base64
from io import BytesIO
from PIL import Image
import structlog
import pyautogui
import time
import difflib
import re

# OCR imports (will be imported conditionally)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = structlog.get_logger()

@dataclass
class UIElement:
    """Represents a UI element"""
    element_type: str  # 'button', 'text_field', 'menu', 'text', 'image'
    coordinates: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    text_content: str = ""
    clickable: bool = False
    description: str = ""
    center_point: Tuple[int, int] = field(init=False)
    
    def __post_init__(self):
        x, y, w, h = self.coordinates
        self.center_point = (x + w // 2, y + h // 2)

@dataclass
class UIAnalysisResult:
    """UI analysis result"""
    elements: List[UIElement]
    screenshot_base64: str
    analysis_time: float
    ocr_method: str
    total_text_found: int
    clickable_elements_count: int

class OCREngine:
    """OCR motor sınıfı"""
    
    def __init__(self, preferred_engine: str = "auto"):
        self.preferred_engine = preferred_engine
        self.easyocr_reader = None
        self._easyocr_initialized = False

        # EasyOCR'ı lazy loading ile başlat
        if preferred_engine == "easyocr" and EASYOCR_AVAILABLE:
            self._init_easyocr()

    def _init_easyocr(self):
        """EasyOCR'ı başlatır"""
        if not self._easyocr_initialized and EASYOCR_AVAILABLE:
            try:
                logger.info("Initializing EasyOCR...")
                self.easyocr_reader = easyocr.Reader(['en', 'tr'], gpu=False, verbose=False)
                self._easyocr_initialized = True
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning("EasyOCR initialization failed", error=str(e))
                self.easyocr_reader = None
                self._easyocr_initialized = False
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extracts text from image"""
        results = []

        # Check OCR libraries status
        if not TESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
            logger.warning("No OCR engines available. Install pytesseract or easyocr.")
            return []

        # EasyOCR'ı dene
        if self.preferred_engine == "easyocr" or self.preferred_engine == "auto":
            if EASYOCR_AVAILABLE:
                try:
                    # EasyOCR'ı başlat (eğer başlatılmamışsa)
                    if not self._easyocr_initialized:
                        self._init_easyocr()

                    if self.easyocr_reader:
                        logger.info("Running EasyOCR text extraction...")
                        ocr_results = self.easyocr_reader.readtext(image)

                        for (bbox, text, confidence) in ocr_results:
                            if confidence > 0.5 and text.strip():  # Minimum confidence threshold and non-empty text
                                # Adjust bounding box coordinates
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                x, y = int(min(x_coords)), int(min(y_coords))
                                w, h = int(max(x_coords) - x), int(max(y_coords) - y)

                                results.append({
                                    'text': text.strip(),
                                    'coordinates': (x, y, w, h),
                                    'confidence': confidence,
                                    'method': 'easyocr'
                                })

                        if results:
                            logger.info("EasyOCR extracted text", count=len(results))
                            return results
                        else:
                            logger.info("EasyOCR found no text")
                    else:
                        logger.warning("EasyOCR reader not available")
                except Exception as e:
                    logger.error("EasyOCR failed", error=str(e))

        # Tesseract'ı dene
        if self.preferred_engine == "tesseract" or self.preferred_engine == "auto" or not results:
            if TESSERACT_AVAILABLE:
                try:
                    # Tesseract için görüntüyü hazırla
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Görüntü kalitesini artır
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.medianBlur(gray, 3)

                    # OCR uygula
                    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--psm 6')

                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        confidence = int(data['conf'][i])

                        if text and confidence > 30:  # Minimum güven eşiği
                            # Koordinatları orijinal boyuta geri çevir
                            x, y, w, h = data['left'][i]//2, data['top'][i]//2, data['width'][i]//2, data['height'][i]//2
                            results.append({
                                'text': text,
                                'coordinates': (x, y, w, h),
                                'confidence': confidence / 100.0,  # 0-1 aralığına normalize et
                                'method': 'tesseract'
                            })

                    if results:
                        logger.info("Tesseract extracted text", count=len(results))
                        return results
                except Exception as e:
                    logger.error("Tesseract failed", error=str(e))

        if not results:
            logger.warning("No text extracted by any OCR engine")

        return results

class UIElementDetector:
    """UI element detection system"""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
        
    def detect_buttons(self, image: np.ndarray) -> List[UIElement]:
        """Detects buttons"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Dikdörtgen yaklaşımı
            x, y, w, h = cv2.boundingRect(contour)
            
            # Buton benzeri boyut kontrolü
            if 20 < w < 300 and 15 < h < 100:
                aspect_ratio = w / h
                if 1.5 < aspect_ratio < 8:  # Buton benzeri oran
                    area = cv2.contourArea(contour)
                    rect_area = w * h
                    
                    if area / rect_area > 0.7:  # Dikdörtgen benzeri
                        elements.append(UIElement(
                            element_type="button",
                            coordinates=(x, y, w, h),
                            confidence=0.7,
                            clickable=True,
                            description=f"Potential button ({w}x{h})"
                        ))
        
        return elements
    
    def detect_text_fields(self, image: np.ndarray) -> List[UIElement]:
        """Detects text fields"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        edges = cv2.Canny(processed, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Text field-like size check
            if 50 < w < 500 and 20 < h < 50:
                aspect_ratio = w / h
                if aspect_ratio > 3:  # Long and narrow
                    elements.append(UIElement(
                        element_type="text_field",
                        coordinates=(x, y, w, h),
                        confidence=0.6,
                        clickable=True,
                        description=f"Potential text field ({w}x{h})"
                    ))

        return elements

    def detect_menus(self, image: np.ndarray) -> List[UIElement]:
        """Detects menu elements including menu bars and menu items"""
        elements = []
        height, width = image.shape[:2]

        # Menu bar detection (top 10% of screen)
        menu_bar_region = image[:int(height * 0.1), :]
        gray_menu = cv2.cvtColor(menu_bar_region, cv2.COLOR_BGR2GRAY)

        # Detect horizontal lines (menu bar indicators)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray_menu, cv2.MORPH_OPEN, horizontal_kernel)

        # Find menu bar regions
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Menu bar characteristics: wide and thin, at top of screen
            if w > width * 0.3 and h < 50 and y < height * 0.1:
                elements.append(UIElement(
                    element_type="menu_bar",
                    coordinates=(x, y, w, h),
                    confidence=0.8,
                    clickable=True,
                    description=f"Menu bar ({w}x{h})"
                ))

        # Detect individual menu items in top region
        edges = cv2.Canny(gray_menu, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Menu item characteristics: moderate width, small height, in top region
            if 30 < w < 200 and 15 < h < 40 and y < height * 0.15:
                aspect_ratio = w / h
                if 1.5 < aspect_ratio < 10:  # Menu item proportions
                    elements.append(UIElement(
                        element_type="menu_item",
                        coordinates=(x, y, w, h),
                        confidence=0.7,
                        clickable=True,
                        description=f"Menu item ({w}x{h})"
                    ))

        return elements
    
    def analyze_screen(self, screenshot: Optional[np.ndarray] = None) -> UIAnalysisResult:
        """Analyzes screen and finds UI elements"""
        start_time = time.time()

        # Take screenshot (if not provided)
        if screenshot is None:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)
                screenshot = np.array(sct_img)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # Screenshot'ı base64'e çevir
        _, buffer = cv2.imencode('.png', screenshot)
        screenshot_base64 = base64.b64encode(buffer).decode('utf-8')
        
        elements = []
        
        # Detect UI elements
        try:
            # Detect buttons
            buttons = self.detect_buttons(screenshot)
            elements.extend(buttons)

            # Detect text fields
            text_fields = self.detect_text_fields(screenshot)
            elements.extend(text_fields)

            # Detect menus
            menus = self.detect_menus(screenshot)
            elements.extend(menus)

            logger.info("UI elements detected",
                       buttons=len(buttons),
                       text_fields=len(text_fields),
                       menus=len(menus))
        except Exception as e:
            logger.error("UI element detection failed", error=str(e))
        
        # Extract texts with OCR
        ocr_method = "none"
        try:
            # OCR engine'i auto olarak ayarla ve başlat
            if self.ocr_engine.preferred_engine == "auto":
                self.ocr_engine.preferred_engine = "easyocr"

            logger.info("Starting OCR for UI analysis...")
            ocr_results = self.ocr_engine.extract_text(screenshot)
            ocr_method = ocr_results[0]['method'] if ocr_results else "none"

            for ocr_result in ocr_results:
                elements.append(UIElement(
                    element_type="text",
                    coordinates=ocr_result['coordinates'],
                    confidence=ocr_result['confidence'],
                    text_content=ocr_result['text'],
                    clickable=False,
                    description=f"Text: '{ocr_result['text'][:30]}...'"
                ))

            logger.info("OCR completed for UI analysis",
                       method=ocr_method,
                       texts_found=len(ocr_results))
        except Exception as e:
            logger.error("OCR failed in UI analysis", error=str(e))
        
        analysis_time = time.time() - start_time
        clickable_count = sum(1 for elem in elements if elem.clickable)
        text_count = sum(1 for elem in elements if elem.element_type == "text")
        
        return UIAnalysisResult(
            elements=elements,
            screenshot_base64=screenshot_base64,
            analysis_time=analysis_time,
            ocr_method=ocr_method,
            total_text_found=text_count,
            clickable_elements_count=clickable_count
        )

class SmartClicker:
    """Smart clicking system"""
    
    def __init__(self):
        self.ui_detector = UIElementDetector()
        # PyAutoGUI security settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def calculate_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1 = re.sub(r'[^\w\s]', '', text1.lower().strip())
        text2 = re.sub(r'[^\w\s]', '', text2.lower().strip())

        if text1 == text2:
            return 1.0

        # Use difflib for sequence matching
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity

    def get_position_score(self, element: UIElement, target_text: str) -> float:
        """Calculate position-based score for common UI elements"""
        x, y, w, h = element.coordinates
        screen_height = 1080  # Assume standard screen height
        screen_width = 1920   # Assume standard screen width

        position_score = 0.0
        target_lower = target_text.lower()

        # File menu typically in top-left
        if 'file' in target_lower and 'menu' in target_lower:
            if y < screen_height * 0.1 and x < screen_width * 0.2:
                position_score = 0.3

        # Save/OK buttons typically in bottom-right or center
        elif any(word in target_lower for word in ['save', 'ok', 'submit', 'apply']):
            if y > screen_height * 0.7:
                position_score = 0.2

        # Cancel buttons typically near save buttons
        elif 'cancel' in target_lower:
            if y > screen_height * 0.7:
                position_score = 0.2

        return position_score

    def find_element_by_text(self, target_text: str, similarity_threshold: float = 0.4) -> Optional[UIElement]:
        """Finds element by text - enhanced algorithm with fuzzy matching"""
        try:
            analysis = self.ui_detector.analyze_screen()

            target_text_lower = target_text.lower()
            best_match = None
            best_score = 0

            # Enhanced keywords
            button_keywords = ['button', 'btn', 'click', 'submit', 'save', 'cancel', 'ok', 'yes', 'no', 'apply', 'close']
            field_keywords = ['field', 'input', 'text', 'email', 'password', 'search', 'box']
            menu_keywords = ['menu', 'file', 'edit', 'view', 'help', 'tools', 'options']

            for element in analysis.elements:
                score = 0

                # 1. Enhanced text content matching
                if element.text_content:
                    element_text_lower = element.text_content.lower()

                    # Exact match
                    if target_text_lower == element_text_lower:
                        score = 1.0
                    # Fuzzy matching
                    else:
                        fuzzy_score = self.calculate_fuzzy_similarity(target_text_lower, element_text_lower)
                        score = max(score, fuzzy_score)

                        # Substring matching
                        if target_text_lower in element_text_lower or element_text_lower in target_text_lower:
                            substring_score = min(len(target_text_lower), len(element_text_lower)) / max(len(target_text_lower), len(element_text_lower))
                            score = max(score, substring_score)

                        # Word-based matching
                        target_words = set(target_text_lower.split())
                        element_words = set(element_text_lower.split())
                        common_words = target_words.intersection(element_words)
                        if common_words:
                            word_score = len(common_words) / max(len(target_words), len(element_words))
                            score = max(score, word_score)

                # 2. Enhanced element type matching
                if any(keyword in target_text_lower for keyword in button_keywords):
                    if element.element_type == "button":
                        score += 0.3
                elif any(keyword in target_text_lower for keyword in field_keywords):
                    if element.element_type == "text_field":
                        score += 0.3
                elif any(keyword in target_text_lower for keyword in menu_keywords):
                    if element.element_type in ["menu_item", "menu_bar"]:
                        score += 0.4

                # 3. Position-based scoring
                position_score = self.get_position_score(element, target_text)
                score += position_score

                # 4. Clickability bonus
                if element.clickable:
                    score += 0.1

                # 5. Confidence bonus
                score += element.confidence * 0.1

                # 6. Element type specific bonuses
                if element.element_type == "menu_item" and any(word in target_text_lower for word in ['file', 'edit', 'view', 'help']):
                    score += 0.2

                # Update best match
                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = element

            if best_match:
                logger.info("Element found",
                           target=target_text,
                           found_type=best_match.element_type,
                           score=best_score,
                           text_content=best_match.text_content,
                           coordinates=best_match.coordinates)
            else:
                logger.warning("No element found",
                             target=target_text,
                             threshold=similarity_threshold,
                             total_elements=len(analysis.elements))

            return best_match
        except Exception as e:
            logger.error("Element search failed", error=str(e))
            return None
    
    def click_element(self, element: UIElement) -> bool:
        """Elementi tıklar"""
        try:
            x, y = element.center_point
            pyautogui.click(x, y)
            logger.info("Element clicked", 
                       coordinates=(x, y), 
                       element_type=element.element_type)
            return True
        except Exception as e:
            logger.error("Click failed", error=str(e))
            return False
    
    def smart_click(self, description: str) -> Dict[str, Any]:
        """Smart clicking based on natural language description"""
        try:
            # First find the element
            element = self.find_element_by_text(description)
            
            if element:
                success = self.click_element(element)
                return {
                    "success": success,
                    "element_found": True,
                    "element_type": element.element_type,
                    "coordinates": element.center_point,
                    "text_content": element.text_content,
                    "confidence": element.confidence
                }
            else:
                return {
                    "success": False,
                    "element_found": False,
                    "message": f"Element not found for description: '{description}'"
                }
        except Exception as e:
            logger.error("Smart click failed", error=str(e))
            return {
                "success": False,
                "element_found": False,
                "error": str(e)
            }

# Global instances
_ui_detector: Optional[UIElementDetector] = None
_smart_clicker: Optional[SmartClicker] = None

def get_ui_detector() -> UIElementDetector:
    """Returns global UI detector instance"""
    global _ui_detector
    if _ui_detector is None:
        _ui_detector = UIElementDetector()
        # Set OCR engine to easyocr
        _ui_detector.ocr_engine.preferred_engine = "easyocr"
        logger.info("UI Detector created with EasyOCR preference")
    return _ui_detector

def get_smart_clicker() -> SmartClicker:
    """Returns global smart clicker instance"""
    global _smart_clicker
    if _smart_clicker is None:
        _smart_clicker = SmartClicker()
    return _smart_clicker
