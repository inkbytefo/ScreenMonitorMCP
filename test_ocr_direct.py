#!/usr/bin/env python3
"""
OCR'ı doğrudan test eder
"""

import mss
import cv2
import numpy as np
from ui_detection import get_ui_detector

def test_ocr_direct():
    """OCR'ı doğrudan test eder"""
    print("=== Doğrudan OCR Testi ===")
    
    try:
        # Screenshot al
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            sct_img = sct.grab(monitor)
            screenshot = np.array(sct_img)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        print(f"Screenshot alındı: {screenshot.shape}")
        
        # UI detector al
        detector = get_ui_detector()
        print(f"UI Detector alındı: {detector}")
        print(f"OCR Engine: {detector.ocr_engine}")
        print(f"Preferred engine: {detector.ocr_engine.preferred_engine}")
        
        # OCR engine'i easyocr'a ayarla
        detector.ocr_engine.preferred_engine = "easyocr"
        
        # OCR çalıştır
        print("OCR çalıştırılıyor...")
        results = detector.ocr_engine.extract_text(screenshot)
        
        print(f"OCR sonuçları: {len(results)} metin bulundu")
        for i, result in enumerate(results[:5]):  # İlk 5 sonucu göster
            print(f"  {i+1}. '{result['text']}' - güven: {result['confidence']:.2f}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ocr_direct()
    print(f"\nTest {'başarılı' if success else 'başarısız'}!")
