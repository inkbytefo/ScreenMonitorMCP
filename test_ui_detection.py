#!/usr/bin/env python3
"""
Test script for UI Detection System
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ui_detection import UIElementDetector, get_ui_detector
    UI_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"UI Detection not available: {e}")
    UI_DETECTION_AVAILABLE = False

async def test_ui_detection():
    """Test UI detection system"""
    print("🎯 Testing UI Element Detection System")
    print("=" * 50)
    
    if not UI_DETECTION_AVAILABLE:
        print("❌ UI Detection system not available")
        return
    
    try:
        # Create detector
        detector = get_ui_detector()
        print("✅ UI Detector initialized")
        
        # Analyze screen
        print("📊 Analyzing current screen...")
        analysis = detector.analyze_screen()
        
        print(f"📈 Analysis Results:")
        print(f"   Total Elements: {len(analysis.elements)}")
        print(f"   Clickable Elements: {analysis.clickable_elements_count}")
        print(f"   Text Elements: {analysis.total_text_found}")
        print(f"   Analysis Time: {analysis.analysis_time:.3f}s")
        print(f"   OCR Method: {analysis.ocr_method}")
        print()
        
        # Show element details
        if analysis.elements:
            print("🔍 Detected Elements:")
            for i, element in enumerate(analysis.elements[:10]):  # Show first 10
                print(f"   {i+1}. {element.element_type.upper()}")
                print(f"      Position: {element.coordinates}")
                print(f"      Center: {element.center_point}")
                print(f"      Clickable: {element.clickable}")
                if element.text_content:
                    print(f"      Text: '{element.text_content[:50]}...'")
                print(f"      Confidence: {element.confidence:.2f}")
                print()
        
        print("✅ UI Detection test completed!")
        
    except Exception as e:
        print(f"❌ UI Detection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(test_ui_detection())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
