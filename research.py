# Complete Agentic Multi-Modal Research Assistant
# High-performance implementation with all API integrations

import asyncio
import aiohttp
import json
import os
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Import availability checks
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Audio transcription will be limited.")

import cv2
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR features will be limited.")

# Core imports for different functionalities
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("Warning: duckduckgo_search not available. Web search will be limited.")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    print("Warning: youtube_transcript_api not available. YouTube features will be limited.")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt_dlp not available. Video download features will be limited.")

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        pipeline, AutoTokenizer, AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: Transformers not available ({str(e)}). Some AI features will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image processing will be limited.")

import requests
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("Warning: wikipedia not available. Wikipedia search will be limited.")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    print("Warning: praw not available. Reddit features will be limited.")

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Some NLP features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResearchSource:
    title: str
    content: str
    url: str
    source_type: str
    confidence: float
    metadata: Dict

@dataclass
class ResearchInsight:
    text_insights: List[str]
    video_insights: List[str]
    image_insights: List[str]
    fused_summary: List[str]
    follow_up_questions: List[str]
    confidence_score: float
    sources_count: int

class AgenticResearchConfig:
    """Configuration class for API keys and settings"""
    
    def __init__(self):
        # Import the new config system
        from config import get_config
        config = get_config()
        
        # API Keys from config system
        self.OPENAI_API_KEY = config.api_config.openai_api_key
        self.GOOGLE_CSE_API_KEY = config.api_config.google_cse_api_key
        self.GOOGLE_CSE_ID = config.api_config.google_cse_search_engine_id
        self.REDDIT_CLIENT_ID = config.api_config.reddit_client_id
        self.REDDIT_CLIENT_SECRET = config.api_config.reddit_client_secret
        self.REDDIT_USER_AGENT = config.api_config.reddit_user_agent
        self.ASSEMBLYAI_API_KEY = config.api_config.assemblyai_api_key
        self.SERPAPI_KEY = config.api_config.serpapi_api_key
        
        # Model configurations from config system
        self.WHISPER_MODEL = config.model_config.whisper_model_size
        self.MAX_SEARCH_RESULTS = config.model_config.max_search_results
        self.MAX_VIDEO_DURATION = config.model_config.max_video_duration
        
        # File paths from config system
        self.TEMP_DIR = config.model_config.temp_dir
        self.TEMP_DIR.mkdir(exist_ok=True)

class WebSearchEngine:
    """High-performance web search with multiple engines"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_duckduckgo(self, query: str) -> List[ResearchSource]:
        """DuckDuckGo search - unlimited and free"""
        if not DDGS_AVAILABLE:
            logger.warning("DuckDuckGo search not available")
            return []
            
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            sources = []
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.config.MAX_SEARCH_RESULTS))
                
            for result in results:
                source = ResearchSource(
                    title=result.get('title', ''),
                    content=result.get('body', ''),
                    url=result.get('href', ''),
                    source_type='web_search',
                    confidence=0.8,
                    metadata={'engine': 'duckduckgo', 'timestamp': datetime.now().isoformat()}
                )
                sources.append(source)
                
            return sources
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_wikipedia(self, query: str) -> List[ResearchSource]:
        """Wikipedia search for authoritative content"""
        if not WIKIPEDIA_AVAILABLE:
            logger.warning("Wikipedia search not available")
            return []
            
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            sources = []
            
            # Search for relevant pages
            search_results = wikipedia.search(query, results=5)
            
            for title in search_results[:3]:  # Top 3 results
                try:
                    page = wikipedia.page(title)
                    source = ResearchSource(
                        title=page.title,
                        content=page.summary[:2000],  # First 2000 chars
                        url=page.url,
                        source_type='wikipedia',
                        confidence=0.9,
                        metadata={'engine': 'wikipedia', 'categories': page.categories[:5] if hasattr(page, 'categories') else []}
                    )
                    sources.append(source)
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation by taking first option
                    try:
                        page = wikipedia.page(e.options[0])
                        source = ResearchSource(
                            title=page.title,
                            content=page.summary[:2000],
                            url=page.url,
                            source_type='wikipedia',
                            confidence=0.85,
                            metadata={'engine': 'wikipedia', 'disambiguated': True}
                        )
                        sources.append(source)
                    except:
                        continue
                except Exception as page_error:
                    logger.warning(f"Failed to process Wikipedia page {title}: {str(page_error)}")
                    continue
                    
            return sources
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_reddit(self, query: str) -> List[ResearchSource]:
        """Reddit search for community insights"""
        if not PRAW_AVAILABLE or not (self.config.REDDIT_CLIENT_ID and self.config.REDDIT_CLIENT_SECRET):
            logger.warning("Reddit search not available - missing praw or credentials")
            return []
            
        try:
            logger.info(f"Searching Reddit for: {query}")
            sources = []
            
            reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent=self.config.REDDIT_USER_AGENT
            )
            
            # Search across relevant subreddits
            subreddits = ['MachineLearning', 'artificial', 'technology', 'science', 'AskScience']
            
            for subreddit_name in subreddits[:2]:  # Limit to 2 subreddits
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, limit=3):
                        if submission.selftext and len(submission.selftext) > 100:
                            source = ResearchSource(
                                title=submission.title,
                                content=submission.selftext[:1500],
                                url=f"https://reddit.com{submission.permalink}",
                                source_type='reddit',
                                confidence=0.7,
                                metadata={
                                    'subreddit': subreddit_name,
                                    'score': submission.score,
                                    'comments': submission.num_comments
                                }
                            )
                            sources.append(source)
                except Exception as subreddit_error:
                    logger.warning(f"Failed to search subreddit {subreddit_name}: {str(subreddit_error)}")
                    continue
                    
            return sources
            
        except Exception as e:
            logger.error(f"Reddit search failed: {str(e)}")
            return []
    
    async def comprehensive_search(self, query: str) -> List[ResearchSource]:
        """Combine multiple search engines for comprehensive results"""
        logger.info(f"Starting comprehensive search for: {query}")
        
        # Run all searches concurrently
        tasks = [
            self.search_duckduckgo(query),
            self.search_wikipedia(query),
            self.search_reddit(query)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all sources
        all_sources = []
        for result in results:
            if isinstance(result, list):
                all_sources.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search task failed: {str(result)}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        logger.info(f"Found {len(unique_sources)} unique sources")
        return unique_sources

class VideoAudioProcessor:
    """Advanced video and audio processing with multiple methods"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Load Whisper model for speech-to-text"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available. Audio transcription disabled.")
            return
        if not self.whisper_model:
            logger.info(f"Loading Whisper model: {self.config.WHISPER_MODEL}")
            try:
                self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
    
    def extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        try:
            # Handle different YouTube URL formats
            patterns = [
                r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]+)',
                r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_url)
                if match:
                    return match.group(1)
            
            # If no pattern matches, assume it's already a video ID
            if len(video_url) == 11 and (video_url.isalnum() or '-' in video_url or '_' in video_url):
                return video_url
                
            return None
        except Exception as e:
            logger.error(f"Failed to extract video ID from {video_url}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_youtube_transcript(self, video_url: str) -> Optional[ResearchSource]:
        """Extract transcript from YouTube video"""
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            logger.warning("YouTube transcript API not available")
            return None
            
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                logger.error(f"Could not extract video ID from URL: {video_url}")
                return None
            
            logger.info(f"Extracting YouTube transcript for video: {video_id}")
            
            # Try to get transcript
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in transcript_data])
            
            # Get video info if yt-dlp is available
            title = f"YouTube Video {video_id}"
            duration = 0
            
            if YT_DLP_AVAILABLE:
                try:
                    ydl_opts = {'quiet': True, 'no_warnings': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                        title = info.get('title', title)
                        duration = info.get('duration', 0)
                except Exception as ydl_error:
                    logger.warning(f"Could not get video info: {ydl_error}")
            
            return ResearchSource(
                title=title,
                content=transcript_text,
                url=video_url,
                source_type='youtube_transcript',
                confidence=0.85,
                metadata={
                    'duration': duration,
                    'video_id': video_id,
                    'transcript_type': 'auto_generated'
                }
            )
            
        except Exception as e:
            logger.error(f"YouTube transcript extraction failed for {video_url}: {str(e)}")
            return None
    
    def transcribe_audio_with_whisper(self, audio_path: str) -> Optional[ResearchSource]:
        """Transcribe audio file using Whisper"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available for audio transcription")
            return None
            
        try:
            logger.info(f"Transcribing audio with Whisper: {audio_path}")
            
            self.load_whisper_model()
            if not self.whisper_model:
                return None
                
            result = self.whisper_model.transcribe(audio_path)
            
            return ResearchSource(
                title=f"Audio Transcription: {Path(audio_path).name}",
                content=result['text'],
                url=f"file://{audio_path}",
                source_type='audio_transcription',
                confidence=0.9,
                metadata={
                    'language': result.get('language', 'unknown'),
                    'model': self.config.WHISPER_MODEL
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {str(e)}")
            return None
    
    def download_and_transcribe_youtube(self, video_url: str) -> Optional[ResearchSource]:
        """
        Download YouTube video audio, transcribe with Whisper, and ensure cleanup.
        This version is more robust and handles errors gracefully.
        """
        if not (YT_DLP_AVAILABLE and WHISPER_AVAILABLE):
            logger.warning("YouTube download or Whisper not available. Skipping transcription.")
            return None

        actual_audio_file = None
        try:
            logger.info(f"Downloading and transcribing YouTube video: {video_url}")
            
            # Define a unique path for the temporary audio file
            temp_audio_path = self.config.TEMP_DIR / f"temp_audio_{int(time.time())}"
            
            # --- Corrected and Robust yt-dlp Options ---
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(temp_audio_path), # Let yt-dlp handle the extension
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get('title', 'YouTube Video')
                duration = info.get('duration', 0)
                
                # Get the exact filename yt-dlp created
                actual_audio_file = Path(ydl.prepare_filename(info)).with_suffix('.mp3')

            if not actual_audio_file or not actual_audio_file.exists():
                logger.error(f"Audio file not found after download for {video_url}. FFmpeg might be missing.")
                return None

            # Transcribe with Whisper
            self.load_whisper_model()
            if not self.whisper_model:
                return None
                
            result = self.whisper_model.transcribe(str(actual_audio_file))
            
            return ResearchSource(
                title=title,
                content=result['text'],
                url=video_url,
                source_type='youtube_whisper',
                confidence=0.9,
                metadata={
                    'duration': duration,
                    'language': result.get('language', 'unknown'),
                    'method': 'whisper_transcription'
                }
            )

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"Failed to download video {video_url}. It might be private or removed. Error: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during YouTube processing for {video_url}: {str(e)}")
            return None
        finally:
            # --- Robust Cleanup ---
            if actual_audio_file and actual_audio_file.exists():
                try:
                    actual_audio_file.unlink()
                    logger.info(f"Cleaned up temporary file: {actual_audio_file}")
                except OSError as e:
                    logger.error(f"Error removing temporary file {actual_audio_file}: {e}")