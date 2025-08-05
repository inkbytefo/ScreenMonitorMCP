#!/usr/bin/env python3
"""
Memory System Integration Test for ScreenMonitorMCP v2

This script tests the memory system integration with streaming and AI services.

Author: inkbytefo
Version: 2.0.0
License: MIT
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.memory_system import memory_system
from core.ai_service import ai_service
from core.streaming import stream_manager
from core.screen_capture import ScreenCapture

async def test_memory_system():
    """Test basic memory system functionality."""
    print("🧠 Testing Memory System...")
    
    try:
        # Test memory system initialization
        await memory_system.initialize()
        print("✅ Memory system initialized")
        
        # Test storing a sample analysis
        test_analysis = {
            "analysis": "Test screen analysis for memory system",
            "confidence": 0.95,
            "objects_detected": ["window", "text", "button"],
            "scene_description": "Desktop environment with multiple windows"
        }
        
        entry_id = await memory_system.store_analysis(
            analysis_result=test_analysis,
            stream_id="test_stream_001",
            sequence=1,
            tags=["test", "memory_integration"]
        )
        print(f"✅ Analysis stored with ID: {entry_id}")
        
        # Test querying memory
        results = await memory_system.query_memory(
            query="desktop environment",
            limit=5
        )
        print(f"✅ Query returned {len(results)} results")
        
        # Test getting statistics
        stats = await memory_system.get_statistics()
        print(f"✅ Memory statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False

async def test_ai_service_memory_integration():
    """Test AI service integration with memory system."""
    print("\n🤖 Testing AI Service Memory Integration...")
    
    try:
        # Test memory statistics
        stats = await ai_service.get_memory_statistics()
        print(f"✅ AI service memory stats: {stats}")
        
        # Test direct memory query
        results = await ai_service.query_memory_direct(
            query="test analysis",
            limit=3
        )
        print(f"✅ Direct memory query returned {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ AI service memory integration test failed: {e}")
        return False

async def test_streaming_memory_integration():
    """Test streaming integration with memory system."""
    print("\n📺 Testing Streaming Memory Integration...")
    
    try:
        # Test memory configuration
        stream_manager.enable_memory_system(True)
        stream_manager.set_analysis_interval(3)
        print("✅ Memory system enabled for streaming")
        
        # Test getting memory stats
        stats = stream_manager.get_memory_stats()
        print(f"✅ Stream memory stats: {stats}")
        
        # Test creating a stream with memory enabled
        stream_id = await stream_manager.create_stream(
            stream_type="screen",
            fps=2,
            quality=70
        )
        print(f"✅ Stream created with memory integration: {stream_id}")
        
        # Clean up
        await stream_manager.stop_stream(stream_id)
        print("✅ Stream stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming memory integration test failed: {e}")
        return False

async def test_screen_capture_with_memory():
    """Test screen capture with memory storage."""
    print("\n📸 Testing Screen Capture with Memory...")
    
    try:
        screen_capture = ScreenCapture()
        
        # Capture screen
        image_bytes = await screen_capture.capture_screen(
            monitor=0,
            format="jpeg"
        )
        
        # Convert to base64 for AI analysis
        import base64
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        capture_result = {
            "success": True,
            "image_data": image_data
        }
        
        if capture_result.get("success"):
            print("✅ Screen captured successfully")
            
            # Test AI analysis with memory storage
            image_data = capture_result.get("image_data")
            if image_data:
                analysis_result = await ai_service.analyze_image(
                    image_base64=image_data,
                    prompt="Analyze this screen capture for testing memory integration",
                    store_in_memory=True,
                    stream_id="test_capture_stream",
                    sequence=1,
                    tags=["test", "screen_capture", "memory_test"]
                )
                print(f"✅ Screen analysis completed and stored in memory")
                print(f"   Analysis length: {len(analysis_result.get('analysis', ''))} characters")
            else:
                print("⚠️ No image data in capture result")
        else:
            print("❌ Screen capture failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Screen capture with memory test failed: {e}")
        return False

async def run_all_tests():
    """Run all memory integration tests."""
    print("🚀 Starting Memory System Integration Tests\n")
    
    tests = [
        ("Memory System", test_memory_system),
        ("AI Service Memory Integration", test_ai_service_memory_integration),
        ("Streaming Memory Integration", test_streaming_memory_integration),
        ("Screen Capture with Memory", test_screen_capture_with_memory)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory system integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        sys.exit(1)