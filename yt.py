# Test script
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "b0KaGBOU4Ys"  # Your SimpliLearn video
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    print(f"✅ Success! Got {len(transcript)} segments")
    print(f"Sample: {transcript[0]['text']}")
except Exception as e:
    print(f"❌ Error: {e}")