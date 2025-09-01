"""
YouTube Transcript Test Script
Test the fixed YouTube transcript extraction with your specific video
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_youtube_extraction():
    """Test the fixed YouTube transcript extraction"""
    
    print("🔍 Testing YouTube Transcript Extraction")
    print("=" * 50)
    
    # Test with your problematic video
    test_url = "https://youtu.be/b0KaGBOU4Ys?si=zUoOxq_-uMyCTgu7"
    
    try:
        # Import the fixed research module
        from research import AgenticResearchAssistant
        
        print(f"📹 Testing URL: {test_url}")
        print()
        
        # Initialize assistant
        assistant = AgenticResearchAssistant()
        
        # Test video ID extraction
        video_id = assistant.video_processor.extract_video_id(test_url)
        print(f"🆔 Extracted Video ID: {video_id}")
        
        if not video_id:
            print("❌ FAILED: Could not extract video ID")
            return
        
        # Test accessibility
        accessibility = assistant.video_processor.check_video_accessibility(video_id)
        print(f"🌐 Accessibility Check: {accessibility}")
        
        if not accessibility['accessible']:
            print(f"❌ FAILED: Video not accessible - {accessibility['reason']}")
            return
        
        # Test transcript extraction
        print("\n🔄 Starting transcript extraction...")
        print("-" * 40)
        
        result = assistant.video_processor.get_youtube_transcript(test_url)
        
        if result:
            print("🎉 SUCCESS!")
            print(f"📝 Title: {result.title}")
            print(f"📊 Content Length: {len(result.content)} characters")
            print(f"🎯 Confidence: {result.confidence}")
            print(f"🔧 Extraction Method: {result.metadata.get('extraction_method', 'unknown')}")
            print(f"🌐 Language: {result.metadata.get('language', 'unknown')}")
            print(f"📺 Type: {result.metadata.get('transcript_type', 'unknown')}")
            print()
            print("📄 First 300 characters of transcript:")
            print("-" * 40)
            print(result.content[:300] + "..." if len(result.content) > 300 else result.content)
            print()
            
            # Test with research assistant
            print("🧠 Testing full research integration...")
            import asyncio
            
            async def test_full_research():
                insights = await assistant.research_topic(
                    topic="AI and Technology",
                    video_urls=[test_url]
                )
                return insights
            
            # Run the test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            insights = loop.run_until_complete(test_full_research())
            loop.close()
            
            if insights and insights.sources_count > 0:
                print("✅ Full research integration successful!")
                print(f"📊 Total sources: {insights.sources_count}")
                print(f"🎥 Video insights: {len(insights.video_insights)}")
                print()
                if insights.video_insights:
                    print("🎬 Video Insight Preview:")
                    print(insights.video_insights[0][:200] + "...")
            else:
                print("⚠️ Full research integration had issues")
                
        else:
            print("❌ FAILED: No transcript extracted")
            print()
            print("🔧 Troubleshooting suggestions:")
            print("- Check if video has captions enabled")
            print("- Try a different YouTube video")
            print("- Verify internet connection")
            print("- Check YouTube API limits")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 Make sure all dependencies are installed:")
        print("pip install youtube-transcript-api yt-dlp")
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("🔧 Check the error logs above for details")

def test_multiple_videos():
    """Test with multiple video URLs to verify robustness"""
    
    test_videos = [
        "https://youtu.be/b0KaGBOU4Ys?si=zUoOxq_-uMyCTgu7",  # Your original video
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",        # Famous test video
        "https://youtu.be/jNQXAC9IVRw",                      # Another test
    ]
    
    print("\n🔍 Testing Multiple Videos")
    print("=" * 50)
    
    try:
        from research import AgenticResearchAssistant
        assistant = AgenticResearchAssistant()
        
        success_count = 0
        
        for i, url in enumerate(test_videos, 1):
            print(f"\n📹 Test {i}/{len(test_videos)}: {url}")
            print("-" * 30)
            
            try:
                result = assistant.video_processor.get_youtube_transcript(url)
                if result:
                    print(f"✅ SUCCESS: {len(result.content)} chars extracted")
                    print(f"   Title: {result.title[:60]}...")
                    success_count += 1
                    print("❌ FAILED: No transcript available")
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
        
        print(f"\n📊 Results: {success_count}/{len(test_videos)} videos successfully processed")
        
        if success_count == 0:
            print("\n🔧 If all tests fail, try:")
            print("- pip install --upgrade youtube-transcript-api")
            print("- Check your internet connection")
            print("- Try videos you know have captions")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")

if __name__ == "__main__":
    print("🤖 Multi-Modal Research Assistant - YouTube Fix Test")
    print("=" * 60)
    
    # Test 1: Single video (your problematic one)
    test_youtube_extraction()
    
    # Test 2: Multiple videos for robustness
    try:
        test_multiple_videos()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Multiple video test failed: {e}")
    
    print("\n✅ Testing completed!")
    print("\n💡 Next steps:")
    print("1. If tests pass, replace your research.py with the fixed version")
    print("2. Restart your Streamlit app")
    print("3. Try your YouTube URL again in the web interface")