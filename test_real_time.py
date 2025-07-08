#!/usr/bin/env python3
"""
Test script for Real-Time Monitoring System
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_monitor import RealTimeMonitor, MonitoringConfig, ChangeEvent

def test_change_callback(change_event: ChangeEvent):
    """Test callback for change events"""
    print(f"🔥 Change detected: {change_event.change_type} ({change_event.change_percentage:.2%})")
    print(f"   Description: {change_event.description}")
    print(f"   Regions: {len(change_event.affected_regions)}")
    print(f"   Time: {change_event.timestamp}")
    print()

async def test_real_time_monitoring():
    """Test real-time monitoring system"""
    print("🚀 Testing Real-Time Monitoring System")
    print("=" * 50)
    
    # Create configuration
    config = MonitoringConfig(
        fps=1,  # 1 FPS for testing
        change_threshold=0.05,  # 5% threshold
        smart_detection=True,
        save_screenshots=False  # Don't save screenshots for testing
    )
    
    # Create monitor
    monitor = RealTimeMonitor(config)
    monitor.add_change_callback(test_change_callback)
    
    print("📊 Configuration:")
    print(f"   FPS: {config.fps}")
    print(f"   Change Threshold: {config.change_threshold:.1%}")
    print(f"   Smart Detection: {config.smart_detection}")
    print()
    
    # Start monitoring
    print("▶️  Starting monitoring...")
    result = monitor.start_monitoring()
    print(f"   Status: {result['status']}")
    print(f"   Message: {result['message']}")
    print()
    
    print("👀 Monitoring for 10 seconds...")
    print("   Try moving windows, opening apps, or clicking around!")
    print()
    
    # Wait for 10 seconds
    await asyncio.sleep(10)
    
    # Stop monitoring
    print("⏹️  Stopping monitoring...")
    result = monitor.stop_monitoring()
    print(f"   Status: {result['status']}")
    print(f"   Message: {result['message']}")
    
    if 'stats' in result:
        stats = result['stats']
        print(f"   Duration: {stats['duration']}")
        print(f"   Total Frames: {stats['total_frames']}")
        print(f"   Changes Detected: {stats['changes_detected']}")
        print(f"   Average FPS: {stats['avg_fps']:.2f}")
    
    print()
    print("✅ Test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_real_time_monitoring())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
