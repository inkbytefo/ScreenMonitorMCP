#!/usr/bin/env python3
"""
OCR Test ve Düzeltme Scripti
Tesseract kurulumunu kontrol eder ve gerekirse alternatif çözümler sunar
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_tesseract():
    """Tesseract kurulumunu kontrol eder"""
    print("Tesseract kontrolü...")
    
    # Windows'ta yaygın Tesseract yolları
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
    ]
    
    # PATH'te kontrol et
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract PATH'te bulundu")
            return True
    except:
        pass
    
    # Windows'ta yaygın yollarda kontrol et
    if platform.system() == "Windows":
        for path in windows_paths:
            if os.path.exists(path):
                print(f"✅ Tesseract bulundu: {path}")
                # pytesseract'a yolu söyle
                try:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Tesseract yolu ayarlandı: {path}")
                    return True
                except ImportError:
                    print("❌ pytesseract modülü bulunamadı")
                    return False
    
    print("❌ Tesseract bulunamadı")
    return False

def check_easyocr():
    """EasyOCR kurulumunu kontrol eder"""
    print("\nEasyOCR kontrolü...")
    try:
        import easyocr
        print("✅ EasyOCR modülü mevcut")
        
        # Test reader oluştur
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("✅ EasyOCR reader başarıyla oluşturuldu")
        return True
    except ImportError:
        print("❌ EasyOCR modülü bulunamadı")
        return False
    except Exception as e:
        print(f"❌ EasyOCR hatası: {e}")
        return False

def install_tesseract_windows():
    """Windows'ta Tesseract kurulum önerileri"""
    print("\n🔧 Windows'ta Tesseract Kurulum Önerileri:")
    print("1. https://github.com/UB-Mannheim/tesseract/wiki adresinden Tesseract'ı indirin")
    print("2. Kurulum sırasında 'Add to PATH' seçeneğini işaretleyin")
    print("3. Alternatif olarak Chocolatey ile: choco install tesseract")
    print("4. Veya winget ile: winget install UB-Mannheim.TesseractOCR")

def test_ocr_with_sample():
    """Basit bir OCR testi yapar"""
    print("\n🧪 OCR Test...")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Test görüntüsü oluştur
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Sistem fontunu kullanmaya çalış
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), "Test Text 123", fill='black', font=font)
        
        # PIL'den numpy array'e çevir
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # OCR test et
        from ui_detection import OCREngine
        
        ocr = OCREngine("auto")
        results = ocr.extract_text(img_array)
        
        if results:
            print(f"✅ OCR başarılı! Bulunan metin: {results[0]['text']}")
            print(f"   Kullanılan motor: {results[0]['method']}")
            return True
        else:
            print("❌ OCR hiç metin bulamadı")
            return False
            
    except Exception as e:
        print(f"❌ OCR test hatası: {e}")
        return False

if __name__ == "__main__":
    print("=== OCR Sistem Kontrolü ===")
    
    tesseract_ok = check_tesseract()
    easyocr_ok = check_easyocr()
    
    if not tesseract_ok and platform.system() == "Windows":
        install_tesseract_windows()
    
    if tesseract_ok or easyocr_ok:
        test_ocr_with_sample()
    else:
        print("\n❌ Hiçbir OCR motoru kullanılabilir değil!")
        print("En az bir OCR motorunun kurulu olması gerekiyor.")
