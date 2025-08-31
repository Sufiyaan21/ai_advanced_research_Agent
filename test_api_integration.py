import os
import sys
sys.path.append('.')

from research import AgenticResearchConfig

def test_api_integration():
    print("🧪 Testing API Integration...")
    
    config = AgenticResearchConfig()
    api_status = config.get_api_status()
    
    print("\n📊 API Status:")
    for api_name, is_active in api_status.items():
        status = "✅ ACTIVE" if is_active else "❌ INACTIVE"
        print(f"  {status} {api_name.upper()}")
    
    active_count = sum(1 for status in api_status.values() if status)
    print(f"\n🎯 Result: {active_count}/{len(api_status)} APIs configured")
    
    return api_status

if __name__ == "__main__":
    test_api_integration()