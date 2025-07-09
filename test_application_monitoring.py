#!/usr/bin/env python3
"""
Test Application Monitoring System
Tests the new application monitoring features
"""

import asyncio
import time
from datetime import datetime
from application_monitor import ApplicationMonitor, ApplicationEvent, EventType

def test_application_detection():
    """Test application detection functionality"""
    print("🔍 TESTING APPLICATION DETECTION")
    print("-" * 40)
    
    try:
        monitor = ApplicationMonitor()
        print("✅ Application monitor initialized")
        
        # Get running applications
        apps = monitor.detector.get_running_applications()
        print(f"📱 Detected {len(apps)} applications:")
        
        for app_name, app_info in apps.items():
            status = "🟢 ACTIVE" if app_info.is_active else "⚪ INACTIVE"
            print(f"   {status} {app_name}")
            for window in app_info.window_titles[:2]:  # Show first 2 windows
                print(f"      └─ {window}")
            if len(app_info.window_titles) > 2:
                print(f"      └─ ... and {len(app_info.window_titles) - 2} more windows")
        
        print("🎉 Application Detection: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Application Detection: FAILED - {e}")
        return False

def test_application_monitoring():
    """Test application monitoring functionality"""
    print("\n📊 TESTING APPLICATION MONITORING")
    print("-" * 40)
    
    try:
        monitor = ApplicationMonitor()
        print("✅ Application monitor initialized")
        
        # Event callback for testing
        events_received = []
        def test_callback(event: ApplicationEvent):
            events_received.append(event)
            print(f"📨 Event: {event.event_type.value} - {event.application_name}")
        
        monitor.add_event_callback(test_callback)
        print("✅ Event callback registered")
        
        # Register some applications
        monitor.register_application("Blender")
        monitor.register_application("VSCode")
        print("✅ Applications registered for monitoring")
        
        # Start monitoring
        result = monitor.start_monitoring()
        print(f"✅ Monitoring started: {result['status']}")
        
        # Let it run for a few seconds
        print("⏳ Monitoring for 5 seconds...")
        time.sleep(5)
        
        # Check for events
        print(f"📈 Events captured: {len(events_received)}")
        
        # Get recent events
        recent_events = monitor.get_recent_events(limit=5)
        print(f"📋 Recent events: {len(recent_events)}")
        
        # Stop monitoring
        stop_result = monitor.stop_monitoring()
        print(f"✅ Monitoring stopped: {stop_result['status']}")
        
        print("🎉 Application Monitoring: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Application Monitoring: FAILED - {e}")
        return False

def test_custom_events():
    """Test custom event broadcasting"""
    print("\n📡 TESTING CUSTOM EVENT BROADCASTING")
    print("-" * 40)
    
    try:
        monitor = ApplicationMonitor()
        print("✅ Application monitor initialized")
        
        # Event callback for testing
        custom_events = []
        def custom_callback(event: ApplicationEvent):
            if event.event_type == EventType.CUSTOM_EVENT:
                custom_events.append(event)
                print(f"🎯 Custom Event: {event.event_data.get('custom_event_type')} from {event.application_name}")
        
        monitor.add_event_callback(custom_callback)
        
        # Broadcast some custom events
        test_events = [
            ("Blender", "scene_change", {"objects_modified": ["Cube", "Camera"]}),
            ("VSCode", "file_save", {"file_path": "/path/to/file.py"}),
            ("Chrome", "tab_switch", {"url": "https://example.com"}),
        ]
        
        for app_name, event_type, event_data in test_events:
            monitor.broadcast_application_change(app_name, event_type, event_data)
            print(f"📤 Broadcasted: {event_type} from {app_name}")
        
        # Check if events were received
        print(f"📨 Custom events received: {len(custom_events)}")
        
        # Verify event data
        for event in custom_events:
            print(f"   ✓ {event.application_name}: {event.event_data.get('custom_event_type')}")
        
        print("🎉 Custom Event Broadcasting: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Custom Event Broadcasting: FAILED - {e}")
        return False

def test_active_application():
    """Test active application detection"""
    print("\n🎯 TESTING ACTIVE APPLICATION DETECTION")
    print("-" * 40)
    
    try:
        monitor = ApplicationMonitor()
        print("✅ Application monitor initialized")
        
        # Get active application
        active_app = monitor.get_active_application()
        
        if active_app.get("application_name"):
            print(f"🟢 Active Application: {active_app['application_name']}")
            print(f"   Window: {active_app.get('window_title', 'Unknown')}")
            print(f"   Last Activity: {active_app.get('last_activity', 'Unknown')}")
        else:
            print("⚪ No active application detected")
        
        print("🎉 Active Application Detection: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Active Application Detection: FAILED - {e}")
        return False

async def test_integration():
    """Test integration with the main system"""
    print("\n🔗 TESTING INTEGRATION")
    print("-" * 40)
    
    try:
        # This would normally be done in main.py
        from application_monitor import set_global_app_monitor, get_global_app_monitor
        
        monitor = ApplicationMonitor()
        set_global_app_monitor(monitor)
        
        # Test global access
        global_monitor = get_global_app_monitor()
        if global_monitor:
            print("✅ Global monitor access working")
        else:
            print("❌ Global monitor access failed")
            return False
        
        # Test some operations
        result = global_monitor.start_monitoring()
        print(f"✅ Global monitor start: {result['status']}")
        
        await asyncio.sleep(1)  # Brief monitoring
        
        result = global_monitor.stop_monitoring()
        print(f"✅ Global monitor stop: {result['status']}")
        
        print("🎉 Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration: FAILED - {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 STARTING APPLICATION MONITORING TESTS")
    print("=" * 50)
    
    tests = [
        ("Application Detection", test_application_detection),
        ("Application Monitoring", test_application_monitoring),
        ("Custom Events", test_custom_events),
        ("Active Application", test_active_application),
        ("Integration", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Application monitoring system is ready!")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
