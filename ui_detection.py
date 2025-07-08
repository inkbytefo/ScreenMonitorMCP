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
    """UI elementi temsil eder"""
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
    """UI analiz sonucu"""
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
        """Görüntüden metin çıkarır"""
        results = []

        # OCR kütüphanelerinin durumunu kontrol et
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
                            if confidence > 0.5 and text.strip():  # Minimum güven eşiği ve boş olmayan metin
                                # Bounding box koordinatlarını düzenle
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
    """UI element algılama sistemi"""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
        
    def detect_buttons(self, image: np.ndarray) -> List[UIElement]:
        """Butonları algılar"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Kenar algılama
        edges = cv2.Canny(gray, 50, 150)
        
        # Konturları bul
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
        """Metin alanlarını algılar"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Kenar algılama
        edges = cv2.Canny(processed, 30, 100)
        
        # Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Metin alanı benzeri boyut kontrolü
            if 50 < w < 500 and 20 < h < 50:
                aspect_ratio = w / h
                if aspect_ratio > 3:  # Uzun ve dar
                    elements.append(UIElement(
                        element_type="text_field",
                        coordinates=(x, y, w, h),
                        confidence=0.6,
                        clickable=True,
                        description=f"Potential text field ({w}x{h})"
                    ))
        
        return elements
    
    def analyze_screen(self, screenshot: Optional[np.ndarray] = None) -> UIAnalysisResult:
        """Ekranı analiz eder ve UI elementlerini bulur"""
        start_time = time.time()
        
        # Screenshot al (eğer verilmemişse)
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
        
        # UI elementlerini algıla
        try:
            # Butonları algıla
            buttons = self.detect_buttons(screenshot)
            elements.extend(buttons)
            
            # Metin alanlarını algıla
            text_fields = self.detect_text_fields(screenshot)
            elements.extend(text_fields)
            
            logger.info("UI elements detected", 
                       buttons=len(buttons), 
                       text_fields=len(text_fields))
        except Exception as e:
            logger.error("UI element detection failed", error=str(e))
        
        # OCR ile metinleri çıkar
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
    """Akıllı tıklama sistemi"""
    
    def __init__(self):
        self.ui_detector = UIElementDetector()
        # PyAutoGUI güvenlik ayarları
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def find_element_by_text(self, target_text: str, similarity_threshold: float = 0.6) -> Optional[UIElement]:
        """Metne göre element bulur - geliştirilmiş algoritma"""
        try:
            analysis = self.ui_detector.analyze_screen()

            target_text_lower = target_text.lower()
            best_match = None
            best_score = 0

            # Anahtar kelimeler
            button_keywords = ['button', 'btn', 'click', 'submit', 'save', 'cancel', 'ok', 'yes', 'no']
            field_keywords = ['field', 'input', 'text', 'email', 'password', 'search']

            for element in analysis.elements:
                score = 0

                # 1. Metin içeriği ile eşleştirme
                if element.text_content:
                    element_text_lower = element.text_content.lower()

                    # Tam eşleşme
                    if target_text_lower == element_text_lower:
                        score = 1.0
                    # İçerik kontrolü
                    elif target_text_lower in element_text_lower or element_text_lower in target_text_lower:
                        score = min(len(target_text_lower), len(element_text_lower)) / max(len(target_text_lower), len(element_text_lower))
                    # Kelime bazlı eşleşme
                    else:
                        target_words = set(target_text_lower.split())
                        element_words = set(element_text_lower.split())
                        common_words = target_words.intersection(element_words)
                        if common_words:
                            score = len(common_words) / max(len(target_words), len(element_words))

                # 2. Element tipi ile eşleştirme
                if any(keyword in target_text_lower for keyword in button_keywords):
                    if element.element_type == "button":
                        score += 0.3
                elif any(keyword in target_text_lower for keyword in field_keywords):
                    if element.element_type == "text_field":
                        score += 0.3

                # 3. Tıklanabilirlik bonusu
                if element.clickable:
                    score += 0.1

                # 4. Güven skoru bonusu
                score += element.confidence * 0.1

                # En iyi eşleşmeyi güncelle
                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = element

            if best_match:
                logger.info("Element found",
                           target=target_text,
                           found_type=best_match.element_type,
                           score=best_score,
                           text_content=best_match.text_content)
            else:
                logger.warning("No element found", target=target_text, threshold=similarity_threshold)

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
        """Doğal dil açıklamasına göre akıllı tıklama"""
        try:
            # Önce elementi bul
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
    """Global UI detector instance'ını döndürür"""
    global _ui_detector
    if _ui_detector is None:
        _ui_detector = UIElementDetector()
        # OCR engine'i easyocr olarak ayarla
        _ui_detector.ocr_engine.preferred_engine = "easyocr"
        logger.info("UI Detector created with EasyOCR preference")
    return _ui_detector

def get_smart_clicker() -> SmartClicker:
    """Global smart clicker instance'ını döndürür"""
    global _smart_clicker
    if _smart_clicker is None:
        _smart_clicker = SmartClicker()
    return _smart_clicker
