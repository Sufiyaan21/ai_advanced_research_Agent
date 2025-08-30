#!/usr/bin/env python3
"""
Demo script for the Multi-Modal Research Assistant
Shows how to use the research assistant programmatically
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from research import AgenticResearchAssistant
from config import get_config, setup_environment

async def demo_basic_research():
    """Demo basic research functionality"""
    print("🤖 Multi-Modal Research Assistant - Demo")
    print("=" * 50)
    
    # Setup environment
    print("🔧 Setting up environment...")
    setup_environment()
    
    # Initialize assistant
    print("🚀 Initializing research assistant...")
    assistant = AgenticResearchAssistant()
    
    # Demo research topic
    topic = "Artificial Intelligence in Healthcare"
    print(f"🔍 Research topic: {topic}")
    
    # Demo video URLs (replace with actual URLs for real demo)
    video_urls = [
        # "https://youtube.com/watch?v=DEMO_VIDEO_1",
        # "https://youtube.com/watch?v=DEMO_VIDEO_2"
    ]
    
    # Demo image paths (replace with actual paths for real demo)
    image_paths = [
        # "path/to/demo_image1.jpg",
        # "path/to/demo_image2.png"
    ]
    
    # Demo audio paths (replace with actual paths for real demo)
    audio_paths = [
        # "path/to/demo_audio.mp3"
    ]
    
    print("📊 Starting research process...")
    
    try:
        # Perform research
        insights = await assistant.research_topic(
            topic=topic,
            video_urls=video_urls if video_urls else None,
            image_paths=image_paths if image_paths else None,
            audio_paths=audio_paths if audio_paths else None
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 RESEARCH RESULTS")
        print("=" * 50)
        
        print(f"🎯 Topic: {topic}")
        print(f"📚 Sources Analyzed: {insights.sources_count}")
        print(f"🎯 Confidence Score: {insights.confidence_score:.2f}")
        
        # Text insights
        if insights.text_insights:
            print(f"\n📰 TEXT INSIGHTS ({len(insights.text_insights)}):")
            for i, insight in enumerate(insights.text_insights, 1):
                print(f"  {i}. {insight}")
        
        # Video insights
        if insights.video_insights:
            print(f"\n🎥 VIDEO/AUDIO INSIGHTS ({len(insights.video_insights)}):")
            for i, insight in enumerate(insights.video_insights, 1):
                print(f"  {i}. {insight}")
        
        # Image insights
        if insights.image_insights:
            print(f"\n🖼️ IMAGE INSIGHTS ({len(insights.image_insights)}):")
            for i, insight in enumerate(insights.image_insights, 1):
                print(f"  {i}. {insight}")
        
        # Fused summary
        print(f"\n🔗 FUSED SUMMARY ({len(insights.fused_summary)}):")
        for i, point in enumerate(insights.fused_summary, 1):
            print(f"  {i}. {point}")
        
        # Follow-up questions
        print(f"\n❓ FOLLOW-UP QUESTIONS ({len(insights.follow_up_questions)}):")
        for i, question in enumerate(insights.follow_up_questions, 1):
            print(f"  {i}. {question}")
        
        # Export report
        print(f"\n💾 Exporting report...")
        report_path = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = assistant.export_research_report(report_path)
        print(f"📄 Report saved to: {report_path}")
        
        # Export JSON
        json_path = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'insights': {
                'text_insights': insights.text_insights,
                'video_insights': insights.video_insights,
                'image_insights': insights.image_insights,
                'fused_summary': insights.fused_summary,
                'follow_up_questions': insights.follow_up_questions,
                'confidence_score': insights.confidence_score,
                'sources_count': insights.sources_count
            },
            'sources': [
                {
                    'title': source.title,
                    'url': source.url,
                    'source_type': source.source_type,
                    'confidence': source.confidence,
                    'metadata': source.metadata
                }
                for source in assistant.all_sources
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"📄 JSON report saved to: {json_path}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_configuration():
    """Demo configuration system"""
    print("\n🔧 CONFIGURATION DEMO")
    print("=" * 30)
    
    config = get_config()
    status = config.get_status()
    
    print("📊 Configuration Status:")
    print(f"  🔑 API Keys: {status['api_keys']}")
    print(f"  📁 Directories: {status['model_config']['temp_dir']}, {status['model_config']['upload_dir']}")
    print(f"  🤖 Whisper Model: {status['model_config']['whisper_model']}")
    print(f"  🔍 Max Search Results: {status['model_config']['max_search_results']}")
    
    missing_keys = status['missing_keys']
    if missing_keys:
        print(f"  ⚠️ Missing API keys: {', '.join(missing_keys)}")
    else:
        print(f"  ✅ All API keys configured")

def demo_file_processing():
    """Demo file processing capabilities"""
    print("\n📁 FILE PROCESSING DEMO")
    print("=" * 30)
    
    # Check if demo files exist
    demo_files = {
        'images': ['demo_image1.jpg', 'demo_image2.png'],
        'audio': ['demo_audio.mp3', 'demo_lecture.wav'],
        'video': ['demo_video.mp4', 'demo_presentation.avi']
    }
    
    for file_type, files in demo_files.items():
        print(f"\n{file_type.upper()} FILES:")
        for file_name in files:
            file_path = Path(file_name)
            if file_path.exists():
                print(f"  ✅ {file_name} (found)")
            else:
                print(f"  ❌ {file_name} (not found)")
    
    print(f"\n💡 To test file processing:")
    print(f"  1. Add sample files to the project directory")
    print(f"  2. Update the demo script with actual file paths")
    print(f"  3. Run the demo again")

async def main():
    """Main demo function"""
    print("🎬 Starting Multi-Modal Research Assistant Demo")
    print("=" * 60)
    
    # Configuration demo
    demo_configuration()
    
    # File processing demo
    demo_file_processing()
    
    # Basic research demo
    await demo_basic_research()
    
    print("\n" + "=" * 60)
    print("🎯 DEMO COMPLETE")
    print("=" * 60)
    print("💡 Next steps:")
    print("  1. Set up your API keys in config.json or .env")
    print("  2. Add sample files for testing")
    print("  3. Run 'python run_app.py' to start the web interface")
    print("  4. Or use the research assistant programmatically in your code")

if __name__ == "__main__":
    asyncio.run(main())

