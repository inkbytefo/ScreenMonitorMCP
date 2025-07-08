#!/usr/bin/env python3
"""
Test script for Predictive Intelligence System
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predictive_intelligence import PredictiveEngine, UserAction, get_predictive_engine
    PREDICTIVE_AVAILABLE = True
except ImportError as e:
    print(f"Predictive Intelligence not available: {e}")
    PREDICTIVE_AVAILABLE = False

async def test_predictive_intelligence():
    """Test predictive intelligence system"""
    print("🧠 Testing Predictive Intelligence System")
    print("=" * 50)
    
    if not PREDICTIVE_AVAILABLE:
        print("❌ Predictive Intelligence system not available")
        return
    
    try:
        # Create engine
        engine = get_predictive_engine()
        print("✅ Predictive Engine initialized")
        
        # Simulate some user actions
        print("📝 Recording sample user actions...")
        
        sample_actions = [
            UserAction(
                timestamp=datetime.now(),
                action_type="click",
                target="File menu",
                app_context="VSCode"
            ),
            UserAction(
                timestamp=datetime.now(),
                action_type="type",
                target="editor",
                text_content="print('hello')",
                app_context="VSCode"
            ),
            UserAction(
                timestamp=datetime.now(),
                action_type="click",
                target="Save button",
                app_context="VSCode"
            ),
            UserAction(
                timestamp=datetime.now(),
                action_type="app_switch",
                target="Chrome",
                app_context="Chrome"
            ),
            UserAction(
                timestamp=datetime.now(),
                action_type="type",
                target="search bar",
                text_content="python documentation",
                app_context="Chrome"
            )
        ]
        
        # Record actions
        for action in sample_actions:
            engine.record_action(action)
        
        print(f"✅ Recorded {len(sample_actions)} sample actions")
        
        # Update patterns
        engine.update_patterns()
        print("✅ Patterns updated")
        
        # Get insights
        print("📊 Analyzing user behavior...")
        insights = engine.get_user_insights()
        
        print(f"📈 Behavior Analysis:")
        print(f"   Total Actions: {insights['total_actions']}")
        print(f"   Total Patterns: {insights['total_patterns']}")
        print(f"   Pattern Types: {insights['pattern_breakdown']}")
        print(f"   Most Common Actions: {insights['most_common_actions']}")
        print()
        
        # Generate predictions
        print("🔮 Generating predictions...")
        current_context = {
            "current_app": "VSCode",
            "current_time": datetime.now()
        }
        
        predictions = engine.generate_predictions(current_context)
        print(f"✅ Generated {len(predictions)} predictions")
        
        for i, pred in enumerate(predictions[:3]):  # Show first 3
            print(f"   {i+1}. {pred.description}")
            print(f"      Confidence: {pred.confidence:.2f}")
            print(f"      Type: {pred.prediction_type}")
            print()
        
        # Get proactive suggestions
        print("💡 Getting proactive suggestions...")
        suggestions = engine.get_proactive_suggestions()
        print(f"✅ Generated {len(suggestions)} suggestions")
        
        for i, suggestion in enumerate(suggestions[:3]):  # Show first 3
            print(f"   {i+1}. {suggestion.get('message', 'No message')}")
            print(f"      Type: {suggestion.get('type', 'unknown')}")
            print(f"      Confidence: {suggestion.get('confidence', 0):.2f}")
            print()
        
        print("✅ Predictive Intelligence test completed!")
        
    except Exception as e:
        print(f"❌ Predictive Intelligence test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(test_predictive_intelligence())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
