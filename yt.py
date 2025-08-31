# Test script
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "b0KaGBOU4Ys"  # Your SimpliLearn video
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    print(f"✅ Success! Got {len(transcript)} segments")
    print(f"Sample: {transcript[0]['text']}")
except Exception as e:
    print(f"❌ Error: {e}")

# quick_test.py
# def quick_youtube_test(video_url):
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
#         import re
        
#         # Extract video ID
#         match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', video_url)
#         if not match:
#             return "❌ Invalid URL format"
        
#         video_id = match.group(1)
#         print(f"Video ID: {video_id}")
        
#         # Test API
#         api = YouTubeTranscriptApi()
#         transcript_list = api.list_transcripts(video_id)
        
#         transcripts = list(transcript_list)
#         if transcripts:
#             print(f"✅ {len(transcripts)} transcript(s) available:")
#             for t in transcripts:
#                 print(f"   • {t.language} ({'auto' if t.is_generated else 'manual'})")
#             return "✅ Should work!"
#         else:
#             return "❌ No transcripts available"
            
#     except Exception as e:
#         return f"❌ Error: {e}"

# # Test your URL
# result = quick_youtube_test("https://youtu.be/b0KaGBOU4Ys?si=-G3xS4v14ECLI-S9")
# print(result)