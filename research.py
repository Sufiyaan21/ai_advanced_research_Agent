"""
Enhanced Multi-Modal Research Assistant with Fixed YouTube Integration
Fixes YouTube transcript API usage and ensures all input fields work correctly
"""

import asyncio
import aiohttp
import json
import os
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Safe imports with fallbacks
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Audio transcription will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Video processing will be limited.")

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    PIL_AVAILABLE = False
    logger.warning("pytesseract/PIL not available. OCR features will be limited.")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo_search not available. Web search will be limited.")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
    logger.info("YouTube Transcript API loaded successfully")
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    logger.warning("youtube_transcript_api not available.")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("yt_dlp not available.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. AI features will be limited.")

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    logger.warning("wikipedia not available.")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("praw not available. Reddit features will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        SPACY_AVAILABLE = False
        logger.warning("spaCy model not loaded")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available.")

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

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
    """Enhanced configuration class with proper API key handling"""
    
    def __init__(self):
        try:
            from config import get_config
            config = get_config()
            
            # API Keys - properly handle both env vars and config file
            self.OPENAI_API_KEY = config.api_config.openai_api_key or os.getenv('OPENAI_API_KEY')
            self.GOOGLE_CSE_API_KEY = config.api_config.google_cse_api_key or os.getenv('GOOGLE_CSE_API_KEY')
            self.GOOGLE_CSE_ID = config.api_config.google_cse_search_engine_id or os.getenv('GOOGLE_CSE_ID')
            self.REDDIT_CLIENT_ID = config.api_config.reddit_client_id or os.getenv('REDDIT_CLIENT_ID')
            self.REDDIT_CLIENT_SECRET = config.api_config.reddit_client_secret or os.getenv('REDDIT_CLIENT_SECRET')
            self.REDDIT_USER_AGENT = config.api_config.reddit_user_agent or "MultiModalResearchAssistant/1.0"
            self.ASSEMBLYAI_API_KEY = config.api_config.assemblyai_api_key or os.getenv('ASSEMBLYAI_API_KEY')
            self.SERPAPI_KEY = config.api_config.serpapi_api_key or os.getenv('SERPAPI_KEY')
            self.HUGGINGFACE_API_KEY = config.api_config.huggingface_api_key or os.getenv('HUGGINGFACE_API_KEY')
            
            self.WHISPER_MODEL = config.model_config.whisper_model_size
            self.MAX_SEARCH_RESULTS = config.model_config.max_search_results
            self.MAX_VIDEO_DURATION = config.model_config.max_video_duration
            
            self.TEMP_DIR = config.model_config.temp_dir
            self.TEMP_DIR.mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using environment variables.")
            # Fallback to environment variables
            self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            self.GOOGLE_CSE_API_KEY = os.getenv('GOOGLE_CSE_API_KEY')
            self.GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
            self.REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
            self.REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
            self.REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'MultiModalResearchAssistant/1.0')
            self.ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
            self.SERPAPI_KEY = os.getenv('SERPAPI_KEY')
            self.HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
            
            self.WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
            self.MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '10'))
            self.MAX_VIDEO_DURATION = int(os.getenv('MAX_VIDEO_DURATION', '600'))
            self.TEMP_DIR = Path("temp_files")
            self.TEMP_DIR.mkdir(exist_ok=True)
    
    def get_api_status(self):
        """Get status of all API keys"""
        return {
            'openai': bool(self.OPENAI_API_KEY),
            'google_cse': bool(self.GOOGLE_CSE_API_KEY and self.GOOGLE_CSE_ID),
            'reddit': bool(self.REDDIT_CLIENT_ID and self.REDDIT_CLIENT_SECRET),
            'assemblyai': bool(self.ASSEMBLYAI_API_KEY),
            'serpapi': bool(self.SERPAPI_KEY),
            'huggingface': bool(self.HUGGINGFACE_API_KEY),
        }

class WebSearchEngine:
    """Enhanced web search with proper API integration"""
    
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
    async def search_google_cse(self, query: str) -> List[ResearchSource]:
        """Google Custom Search Engine with proper API integration"""
        if not (self.config.GOOGLE_CSE_API_KEY and self.config.GOOGLE_CSE_ID):
            logger.info("Google CSE API keys not configured, skipping...")
            return []
            
        try:
            logger.info(f"Searching Google CSE for: {query}")
            sources = []
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.config.GOOGLE_CSE_API_KEY,
                'cx': self.config.GOOGLE_CSE_ID,
                'q': query,
                'num': min(self.config.MAX_SEARCH_RESULTS, 10)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('items', [])
                    
                    for item in items:
                        source = ResearchSource(
                            title=item.get('title', ''),
                            content=item.get('snippet', ''),
                            url=item.get('link', ''),
                            source_type='google_cse',
                            confidence=0.9,
                            metadata={'engine': 'google_cse', 'timestamp': datetime.now().isoformat()}
                        )
                        sources.append(source)
                        
                    logger.info(f"Found {len(sources)} Google CSE results")
                else:
                    logger.error(f"Google CSE API error: {response.status}")
                    
            return sources
            
        except Exception as e:
            logger.error(f"Google CSE search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_serpapi(self, query: str) -> List[ResearchSource]:
        """SerpAPI search with proper API integration"""
        if not self.config.SERPAPI_KEY:
            logger.info("SerpAPI key not configured, skipping...")
            return []
            
        try:
            logger.info(f"Searching SerpAPI for: {query}")
            sources = []
            
            url = "https://serpapi.com/search.json"
            params = {
                'api_key': self.config.SERPAPI_KEY,
                'engine': 'google',
                'q': query,
                'num': min(self.config.MAX_SEARCH_RESULTS, 10)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    organic_results = data.get('organic_results', [])
                    
                    for result in organic_results:
                        source = ResearchSource(
                            title=result.get('title', ''),
                            content=result.get('snippet', ''),
                            url=result.get('link', ''),
                            source_type='serpapi',
                            confidence=0.9,
                            metadata={'engine': 'serpapi', 'timestamp': datetime.now().isoformat()}
                        )
                        sources.append(source)
                        
                    logger.info(f"Found {len(sources)} SerpAPI results")
                else:
                    logger.error(f"SerpAPI error: {response.status}")
                    
            return sources
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_duckduckgo(self, query: str) -> List[ResearchSource]:
        """DuckDuckGo search - unlimited and free"""
        if not DDGS_AVAILABLE:
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
                    source_type='duckduckgo',
                    confidence=0.8,
                    metadata={'engine': 'duckduckgo', 'timestamp': datetime.now().isoformat()}
                )
                sources.append(source)
                
            logger.info(f"Found {len(sources)} DuckDuckGo results")
            return sources
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_wikipedia(self, query: str) -> List[ResearchSource]:
        """Wikipedia search for authoritative content"""
        if not WIKIPEDIA_AVAILABLE:
            return []
            
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            sources = []
            
            search_results = wikipedia.search(query, results=5)
            
            for title in search_results[:3]:
                try:
                    page = wikipedia.page(title)
                    source = ResearchSource(
                        title=page.title,
                        content=page.summary[:2000],
                        url=page.url,
                        source_type='wikipedia',
                        confidence=0.9,
                        metadata={'engine': 'wikipedia'}
                    )
                    sources.append(source)
                except Exception:
                    continue
                    
            logger.info(f"Found {len(sources)} Wikipedia results")
            return sources
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_reddit(self, query: str) -> List[ResearchSource]:
        """Reddit search with proper API integration"""
        if not PRAW_AVAILABLE or not (self.config.REDDIT_CLIENT_ID and self.config.REDDIT_CLIENT_SECRET):
            logger.info("Reddit API not configured, skipping...")
            return []
            
        try:
            logger.info(f"Searching Reddit for: {query}")
            sources = []
            
            reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent=self.config.REDDIT_USER_AGENT
            )
            
            # Search multiple relevant subreddits
            subreddits = ['technology', 'science', 'artificial', 'MachineLearning', 'futurology']
            
            for subreddit_name in subreddits[:3]:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, limit=2, sort='relevance'):
                        if submission.selftext and len(submission.selftext) > 100:
                            source = ResearchSource(
                                title=submission.title,
                                content=submission.selftext[:1500],
                                url=f"https://reddit.com{submission.permalink}",
                                source_type='reddit',
                                confidence=0.7,
                                metadata={'subreddit': subreddit_name, 'score': submission.score}
                            )
                            sources.append(source)
                except Exception as e:
                    logger.warning(f"Error searching subreddit {subreddit_name}: {e}")
                    continue
                    
            logger.info(f"Found {len(sources)} Reddit results")
            return sources
            
        except Exception as e:
            logger.error(f"Reddit search failed: {str(e)}")
            return []
    
    async def comprehensive_search(self, query: str) -> List[ResearchSource]:
        """Combine multiple search engines for comprehensive results"""
        logger.info(f"Starting comprehensive search for: {query}")
        
        # Create search tasks - prioritize based on API availability
        tasks = []
        
        # Add API-based searches if available
        if self.config.GOOGLE_CSE_API_KEY and self.config.GOOGLE_CSE_ID:
            tasks.append(self.search_google_cse(query))
        
        if self.config.SERPAPI_KEY:
            tasks.append(self.search_serpapi(query))
        
        if self.config.REDDIT_CLIENT_ID and self.config.REDDIT_CLIENT_SECRET:
            tasks.append(self.search_reddit(query))
        
        # Always add free searches
        tasks.extend([
            self.search_duckduckgo(query),
            self.search_wikipedia(query)
        ])
        
        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_sources = []
        for result in results:
            if isinstance(result, list):
                all_sources.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Search task failed: {result}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        logger.info(f"Found {len(unique_sources)} unique sources from {len(tasks)} search engines")
        return unique_sources

class VideoAudioProcessor:
    """Enhanced video and audio processing with fixed YouTube API"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Load Whisper model for speech-to-text"""
        if not WHISPER_AVAILABLE:
            return
        if not self.whisper_model:
            try:
                logger.info(f"Loading Whisper model: {self.config.WHISPER_MODEL}")
                self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
    
    def extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL with better pattern matching"""
        try:
            # Clean up the URL
            video_url = video_url.strip()
            
            patterns = [
                r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
                r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
                r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_url)
                if match:
                    video_id = match.group(1)
                    logger.info(f"Extracted video ID: {video_id} from URL: {video_url}")
                    return video_id
            
            logger.warning(f"Could not extract video ID from: {video_url}")
            return None
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None
    
    def get_video_info(self, video_url: str) -> Optional[Dict]:
        """Get video information using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            return None
            
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                return None
                
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'writesubtitles': False,
                'writeautomaticsub': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                full_url = f"https://youtube.com/watch?v={video_id}"
                info = ydl.extract_info(full_url, download=False)
                
                return {
                    'title': info.get('title', f'YouTube Video {video_id}'),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0)
                }
                
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_youtube_transcript(self, video_url: str) -> Optional[ResearchSource]:
        """FIXED: Enhanced YouTube transcript extraction with comprehensive error handling"""
    
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            logger.error("❌ YouTube transcript API not available")
            return None
    
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (
                TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
                TooManyRequests, YouTubeRequestFailed
        )
        except ImportError:
            logger.error("❌ Could not import YouTube transcript API")
            return None
    
    # Extract video ID with enhanced validation
        video_id = self.extract_video_id(video_url)
        if not video_id:
            logger.error(f"❌ Invalid YouTube URL: {video_url}")
            return None
    
        logger.info(f"🎯 Processing YouTube video: {video_id}")
    
    # Pre-flight accessibility check
        try:
            check_url = f"https://www.youtube.com/watch?v={video_id}"
            response = requests.get(check_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        
            if response.status_code != 200:
                logger.error(f"❌ Video not accessible (HTTP {response.status_code})")
                return None
        
        # Check for common error indicators
            error_indicators = [
                'This video is private', 'Video unavailable', 
                'This video has been removed', 'Sign in to confirm your age'
        ]
        
            for indicator in error_indicators:
                if indicator in response.text:
                    logger.error(f"❌ Video issue: {indicator}")
                    return None
        
            logger.info("✅ Video accessibility confirmed")
        
        except Exception as access_error:
            logger.warning(f"⚠️ Accessibility check failed: {access_error}")
    
    # Enhanced transcript extraction with multiple strategies
        transcript_data = None
        extraction_info = {}
    
        try:
            api = YouTubeTranscriptApi()
        
        # Strategy 1: Direct language-specific requests
            logger.info("🔄 Strategy 1: Direct language extraction...")
            for lang_code in ['en', 'en-US', 'en-GB']:
                try:
                    transcript_data = api.get_transcript(video_id, languages=[lang_code])
                    extraction_info = {'method': 'direct', 'language': lang_code}
                    logger.info(f"✅ Success with {lang_code}")
                    break
                except NoTranscriptFound:
                    continue
                except Exception as e:
                    logger.debug(f"Language {lang_code} failed: {e}")
                    continue
        
        # Strategy 2: Comprehensive listing approach
            if not transcript_data:
                logger.info("🔄 Strategy 2: Transcript listing...")
                try:
                    transcript_list = api.list_transcripts(video_id)
                    transcripts = list(transcript_list)
                
                    if not transcripts:
                        logger.error("❌ No transcripts available")
                        return None
                
                    logger.info(f"📋 Found {len(transcripts)} transcript(s)")
                
                # Try manual English first, then auto English, then any
                    transcript_list = api.list_transcripts(video_id)  # Refresh
                
                    for transcript in transcript_list:
                        try:
                            if transcript.language_code.startswith('en'):
                                transcript_data = transcript.fetch()
                                extraction_info = {
                                    'method': 'listing',
                                    'language': transcript.language,
                                    'type': 'auto' if transcript.is_generated else 'manual'
                                }
                                logger.info(f"✅ Success with {transcript.language}")
                                break
                        except Exception as fetch_error:
                            logger.debug(f"Fetch failed for {transcript.language}: {fetch_error}")
                            continue
                
                # If no English found, try any language
                    if not transcript_data:
                        transcript_list = api.list_transcripts(video_id)
                        for transcript in transcript_list:
                            try:
                                transcript_data = transcript.fetch()
                                extraction_info = {
                                    'method': 'listing_any',
                                    'language': transcript.language,
                                    'type': 'auto' if transcript.is_generated else 'manual'
                                }
                                logger.info(f"✅ Using {transcript.language} transcript")
                                break
                            except:
                                continue
                            
                except Exception as list_error:
                    logger.error(f"Listing approach failed: {list_error}")
        
        # Strategy 3: No-filter fallback
            if not transcript_data:
                logger.info("🔄 Strategy 3: No-filter fallback...")
                try:
                    transcript_data = api.get_transcript(video_id)
                    extraction_info = {'method': 'no_filter', 'language': 'auto'}
                    logger.info("✅ No-filter success")
                except Exception as no_filter_error:
                    logger.error(f"No-filter failed: {no_filter_error}")
        
        # Process results
            if not transcript_data:
                logger.error(f"❌ All extraction strategies failed for: {video_id}")
                return None
        
        # Process transcript safely
            transcript_text = self._process_transcript_safely(transcript_data)
            if not transcript_text or len(transcript_text.strip()) < 20:
                logger.error("❌ Transcript too short after processing")
                return None
        
        # Get video title
            title = self._get_video_title_safely(video_id)
        
            logger.info(f"🎉 SUCCESS: {len(transcript_text)} characters extracted")
        
            return ResearchSource(
                title=title,
                content=transcript_text,
                url=video_url,
                source_type='youtube_transcript',
                confidence=0.9 if extraction_info.get('type') == 'manual' else 0.8,
                metadata={
                    'video_id': video_id,
                    'extraction_method': extraction_info.get('method', 'unknown'),
                    'language': extraction_info.get('language', 'unknown'),
                    'transcript_type': extraction_info.get('type', 'unknown'),
                    'transcript_length': len(transcript_text)
                }
            )
        
        except TranscriptsDisabled:
            logger.error(f"❌ Transcripts disabled for video: {video_id}")
            return None
        except VideoUnavailable:
            logger.error(f"❌ Video unavailable: {video_id}")
            return None
        except TooManyRequests:
            logger.error(f"❌ Rate limit exceeded: {video_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None

    def _process_transcript_safely(self, transcript_data: List[Dict]) -> str:
        """Safely process transcript data with enhanced error handling"""
        try:
            if not transcript_data or not isinstance(transcript_data, list):
                return ""
        
            text_segments = []
            for segment in transcript_data:
                if isinstance(segment, dict):
                    text = segment.get('text', '').strip()
                    if text:
                    # Clean subtitle artifacts
                        text = re.sub(r'\[.*?\]', '', text)  # [Music], [Applause]
                        text = re.sub(r'\(.*?\)', '', text)  # (inaudible)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if text:
                            text_segments.append(text)
        
            full_transcript = ' '.join(text_segments)
            full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
        
            logger.info(f"Processed {len(text_segments)} segments → {len(full_transcript)} characters")
            return full_transcript
        
        except Exception as e:
            logger.error(f"Transcript processing failed: {e}")
            return ""

    def _get_video_title_safely(self, video_id: str) -> str:
        """Safely get video title with fallbacks"""
        try:
        # Try yt-dlp first
            if YT_DLP_AVAILABLE:
                try:
                    import yt_dlp
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                        if info and info.get('title'):
                            return info['title']
                except:
                    pass
        
        # Fallback to web scraping
            try:
                response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=10)
                if response.status_code == 200:
                    match = re.search(r'"title":"([^"]*)"', response.text)
                    if match:
                        return match.group(1).replace(' - YouTube', '').strip()
            except:
                pass
        
            return f"YouTube Video {video_id}"
        
        except Exception:
            return f"YouTube Video {video_id}"
    
    def _extract_transcript_with_ytdlp(self, video_id: str) -> Optional[str]:
        """Alternative transcript extraction using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            return None
            
        try:
            logger.info("Attempting transcript extraction with yt-dlp...")
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'en-US'],
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://youtube.com/watch?v={video_id}"
                info = ydl.extract_info(url, download=False)
                
                # Try to extract subtitles/captions
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # Look for English subtitles
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in subtitles:
                        # Manual subtitles are preferred
                        return self._process_subtitle_data(subtitles[lang])
                    elif lang in automatic_captions:
                        # Fallback to automatic captions
                        return self._process_subtitle_data(automatic_captions[lang])
                
                return None
                
        except Exception as e:
            logger.error(f"yt-dlp transcript extraction failed: {e}")
            return None
    
    def _process_subtitle_data(self, subtitle_formats: List[Dict]) -> Optional[str]:
        """Process subtitle data from yt-dlp"""
        try:
            # Look for the best subtitle format
            for fmt in subtitle_formats:
                if fmt.get('ext') in ['vtt', 'srt', 'ttml']:
                    # Here you would download and parse the subtitle file
                    # For now, return None as this requires additional processing
                    pass
            return None
        except Exception:
            return None
    
    def transcribe_audio_with_whisper(self, audio_path: str) -> Optional[ResearchSource]:
        """Transcribe audio file using Whisper with progress tracking"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available for audio transcription")
            return None
            
        try:
            logger.info(f"Transcribing audio with Whisper: {audio_path}")
            
            self.load_whisper_model()
            if not self.whisper_model:
                return None
            
            # Check file size and duration
            file_path = Path(audio_path)
            if not file_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return None
                
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Audio file size: {file_size:.1f} MB")
            
            # Transcribe with error handling
            result = self.whisper_model.transcribe(audio_path)
            
            return ResearchSource(
                title=f"Audio Transcription: {file_path.name}",
                content=result['text'],
                url=f"file://{audio_path}",
                source_type='audio_transcription',
                confidence=0.9,
                metadata={
                    'model': self.config.WHISPER_MODEL,
                    'language': result.get('language', 'unknown'),
                    'file_size_mb': file_size
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return None
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file for transcription"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for video processing")
            return None
            
        try:
            import tempfile
            import subprocess
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            
            # Use ffmpeg to extract audio (if available)
            try:
                subprocess.run([
                    'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                    '-ar', '16000', '-ac', '1', audio_path, '-y'
                ], check=True, capture_output=True)
                
                logger.info(f"Extracted audio from video: {video_path}")
                return audio_path
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("ffmpeg not available, trying OpenCV audio extraction")
                # Fallback: basic audio extraction with OpenCV (limited)
                return None
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None

class ImageProcessor:
    """Enhanced image analysis and OCR with better error handling"""
    
    def __init__(self):
        self.blip_processor = None
        self.blip_model = None
        
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with better error handling"""
        if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("OCR dependencies not available")
            return ""
            
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Handle both file paths and uploaded file objects
            if hasattr(image_path, 'read'):
                # It's an uploaded file object
                image = Image.open(image_path)
            else:
                # It's a file path
                image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use multiple OCR configurations for better results
            configs = [
                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
                '--psm 3',  # Fully automatic page segmentation
                '--psm 6',  # Uniform block of text
                '--psm 8'   # Single word
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config).strip()
                    if len(text) > len(best_text):
                        best_text = text
                except:
                    continue
            
            # Clean up extracted text
            best_text = re.sub(r'\s+', ' ', best_text).strip()
            
            logger.info(f"Extracted {len(best_text)} characters from image")
            return best_text
            
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""
    
    def generate_image_caption(self, image_path: str) -> str:
        """Generate caption for image with fallback"""
        if not TRANSFORMERS_AVAILABLE:
            return "Image caption generation not available - Transformers not installed"
            
        try:
            logger.info(f"Generating caption for image: {image_path}")
            
            # Initialize BLIP model if needed
            if not self.blip_processor:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                logger.info("Loading BLIP model for image captioning...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("BLIP model loaded successfully")
            
            # Handle both file paths and uploaded file objects
            if hasattr(image_path, 'read'):
                image = Image.open(image_path)
            else:
                image = Image.open(image_path)
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            logger.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Image captioning failed: {str(e)}")
            return f"Could not generate caption (error: {type(e).__name__})"
    
    def analyze_image(self, image_path: str) -> ResearchSource:
        """Comprehensive image analysis"""
        logger.info(f"Analyzing image: {image_path}")
        
        ocr_text = self.extract_text_from_image(image_path)
        caption = self.generate_image_caption(image_path)
        
        # Combine OCR and caption
        content_parts = [f"Visual Description: {caption}"]
        if ocr_text:
            content_parts.append(f"Extracted Text: {ocr_text}")
        
        content = "\n".join(content_parts)
        
        # Get image name
        if hasattr(image_path, 'name'):
            image_name = image_path.name
        else:
            image_name = Path(image_path).name
        
        return ResearchSource(
            title=f"Image Analysis: {image_name}",
            content=content,
            url=f"file://{image_path}",
            source_type='image',
            confidence=0.8,
            metadata={
                'has_text': bool(ocr_text),
                'text_length': len(ocr_text),
                'caption_length': len(caption)
            }
        )

class TextProcessor:
    """Enhanced text analysis and summarization"""
    
    def __init__(self):
        self.summarizer = None
        self.sentiment_analyzer = None
        
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Enhanced text summarization with fallbacks"""
        if not text or len(text.strip()) < 50:
            return text
            
        if not TRANSFORMERS_AVAILABLE:
            # Fallback: Extract first few sentences
            sentences = re.split(r'[.!?]+', text.strip())
            summary_sentences = []
            char_count = 0
            
            for sentence in sentences[:5]:
                sentence = sentence.strip()
                if sentence and char_count + len(sentence) < max_length:
                    summary_sentences.append(sentence)
                    char_count += len(sentence)
                else:
                    break
                    
            return '. '.join(summary_sentences) + ('.' if summary_sentences else '')
            
        try:
            if not self.summarizer:
                logger.info("Loading summarization model...")
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                logger.info("Summarization model loaded")
            
            # Prepare text for summarization
            input_text = text[:1024]  # BART limit
            
            summary = self.summarizer(
                input_text, 
                max_length=min(max_length, len(input_text)//2), 
                min_length=min(50, len(input_text)//4), 
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            # Fallback to extractive summary
            sentences = re.split(r'[.!?]+', text.strip())
            return '. '.join(sentences[:3]) + '.' if sentences else text[:max_length]
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text with fallback"""
        if not TRANSFORMERS_AVAILABLE:
            return {'label': 'NEUTRAL', 'score': 0.5}
            
        try:
            if not self.sentiment_analyzer:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            
            result = self.sentiment_analyzer(text[:512])
            return result[0]
            
        except Exception:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        if not SPACY_AVAILABLE:
            # Simple regex-based fallback
            import re
            # Find capitalized words that might be entities
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return list(set(entities))[:10]
            
        try:
            doc = nlp(text[:5000])
            entities = list(set([ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]))
            return entities[:10]
        except Exception:
            return []

class AgenticResearchAssistant:
    """Enhanced main research assistant class with better error handling"""
    
    def __init__(self):
        self.config = AgenticResearchConfig()
        self.web_search = WebSearchEngine(self.config)
        self.video_processor = VideoAudioProcessor(self.config)
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        
        self.all_sources = []
        self.current_topic = None
        self.text_insights = []
        self.video_insights = []
        self.image_insights = []
        
        # Log API status
        api_status = self.config.get_api_status()
        logger.info(f"API Status: {api_status}")
        
    async def research_topic(
        self,
        topic: str,
        video_urls: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        text_sources: Optional[str] = None
    ) -> ResearchInsight:
        """Conduct comprehensive research on a topic with progress tracking"""
        
        logger.info(f"Starting research on topic: {topic}")
        self.current_topic = topic
        self.all_sources = []
        
        # Phase 1: Web search
        logger.info("Phase 1: Conducting web search...")
        try:
            async with self.web_search as search_engine:
                web_sources = await search_engine.comprehensive_search(topic)
                self.all_sources.extend(web_sources)
                logger.info(f"Web search completed: {len(web_sources)} sources found")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
        
        # Phase 2: Process text sources
        if text_sources and text_sources.strip():
            logger.info("Phase 2: Processing additional text sources...")
            try:
                source = ResearchSource(
                    title="User Provided Text",
                    content=text_sources.strip(),
                    url="user_input",
                    source_type='user_text',
                    confidence=0.9,
                    metadata={'length': len(text_sources)}
                )
                self.all_sources.append(source)
                logger.info("User text processed successfully")
            except Exception as e:
                logger.error(f"Error processing user text: {e}")
        
        # Phase 3: Process videos
        if video_urls:
            logger.info(f"Phase 3: Processing {len(video_urls)} video URLs...")
            for i, url in enumerate(video_urls, 1):
                if url.strip():
                    try:
                        logger.info(f"Processing video {i}/{len(video_urls)}: {url}")
                        source = self.video_processor.get_youtube_transcript(url.strip())
                        if source:
                            self.all_sources.append(source)
                            logger.info(f"Video {i} processed successfully")
                        else:
                            logger.warning(f"Video {i} could not be processed - no transcript available")
                    except Exception as e:
                        logger.error(f"Error processing video {i}: {e}")
        
        # Phase 4: Process audio files
        if audio_paths:
            logger.info(f"Phase 4: Processing {len(audio_paths)} audio files...")
            for i, path in enumerate(audio_paths, 1):
                try:
                    logger.info(f"Processing audio {i}/{len(audio_paths)}: {path}")
                    
                    # Check if it's a video file that needs audio extraction
                    file_ext = Path(path).suffix.lower() if isinstance(path, str) else ''
                    if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                        # Extract audio from video first
                        audio_path = self.video_processor.extract_audio_from_video(path)
                        if audio_path:
                            source = self.video_processor.transcribe_audio_with_whisper(audio_path)
                            # Clean up temporary audio file
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
                        else:
                            logger.warning(f"Could not extract audio from video: {path}")
                            continue
                    else:
                        # Direct audio transcription
                        source = self.video_processor.transcribe_audio_with_whisper(path)
                    
                    if source:
                        self.all_sources.append(source)
                        logger.info(f"Audio {i} processed successfully")
                    else:
                        logger.warning(f"Audio {i} could not be processed")
                except Exception as e:
                    logger.error(f"Error processing audio {i}: {e}")
        
        # Phase 5: Process images
        if image_paths:
            logger.info(f"Phase 5: Processing {len(image_paths)} images...")
            for i, path in enumerate(image_paths, 1):
                try:
                    logger.info(f"Processing image {i}/{len(image_paths)}")
                    source = self.image_processor.analyze_image(path)
                    self.all_sources.append(source)
                    logger.info(f"Image {i} processed successfully")
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
        
        # Phase 6: Generate insights
        logger.info("Phase 6: Generating insights...")
        insights = self._generate_insights()
        
        logger.info(f"Research completed: {insights.sources_count} sources, confidence: {insights.confidence_score:.2f}")
        return insights
    
    def _generate_insights(self) -> ResearchInsight:
        """Generate comprehensive insights from all sources"""
        text_insights = []
        video_insights = []
        image_insights = []
        
        logger.info(f"Generating insights from {len(self.all_sources)} sources")
        
        for source in self.all_sources:
            try:
                if source.source_type in ['google_cse', 'serpapi', 'duckduckgo', 'wikipedia', 'reddit', 'user_text']:
                    summary = self.text_processor.summarize_text(source.content)
                    insight = f"[{source.title}] {summary}"
                    text_insights.append(insight)
                
                elif source.source_type in ['youtube_transcript', 'audio_transcription']:
                    summary = self.text_processor.summarize_text(source.content)
                    insight = f"[{source.title}] {summary}"
                    video_insights.append(insight)
                
                elif source.source_type == 'image':
                    image_insights.append(f"[{source.title}] {source.content}")
                    
            except Exception as e:
                logger.error(f"Error generating insight for {source.title}: {e}")
        
        # Generate fused summary
        all_content = "\n".join([s.content for s in self.all_sources[:10]])  # Limit for performance
        fused_summary = self._generate_fused_summary(all_content)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions()
        
        # Calculate confidence
        if self.all_sources:
            confidence_score = sum([s.confidence for s in self.all_sources]) / len(self.all_sources)
        else:
            confidence_score = 0.0
        
        return ResearchInsight(
            text_insights=text_insights[:8],  # Limit for display
            video_insights=video_insights[:5],
            image_insights=image_insights[:5],
            fused_summary=fused_summary,
            follow_up_questions=follow_up_questions,
            confidence_score=min(confidence_score, 1.0),
            sources_count=len(self.all_sources)
        )
    
    def _generate_fused_summary(self, content: str) -> List[str]:
        """Generate enhanced fused summary points"""
        summary_points = []
        
        try:
            # Topic overview
            if self.current_topic:
                api_count = sum(1 for status in self.config.get_api_status().values() if status)
                summary_points.append(
                    f"Comprehensive analysis of '{self.current_topic}' using {len(self.all_sources)} sources "
                    f"from {api_count + 2} different platforms and APIs"
                )
            
            # Source diversity analysis
            source_types = {}
            for source in self.all_sources:
                source_types[source.source_type] = source_types.get(source.source_type, 0) + 1
            
            if len(source_types) > 1:
                type_descriptions = []
                for source_type, count in source_types.items():
                    if source_type == 'google_cse':
                        type_descriptions.append(f"{count} Google CSE results")
                    elif source_type == 'serpapi':
                        type_descriptions.append(f"{count} SerpAPI results")
                    elif source_type == 'reddit':
                        type_descriptions.append(f"{count} Reddit discussions")
                    elif source_type == 'youtube_transcript':
                        type_descriptions.append(f"{count} YouTube transcripts")
                    elif source_type == 'image':
                        type_descriptions.append(f"{count} image analyses")
                    else:
                        type_descriptions.append(f"{count} {source_type} sources")
                
                summary_points.append(f"Multi-modal analysis includes: {', '.join(type_descriptions)}")
            
            # Content-based summary
            if content and len(content) > 100:
                main_summary = self.text_processor.summarize_text(content[:3000], max_length=150)
                if main_summary and len(main_summary) > 20:
                    summary_points.append(f"Key findings: {main_summary}")
            
            # API integration status
            api_status = self.config.get_api_status()
            active_apis = [name for name, status in api_status.items() if status]
            if active_apis:
                summary_points.append(
                    f"Enhanced search capabilities using {len(active_apis)} integrated APIs: "
                    f"{', '.join(active_apis)}"
                )
            
            # Confidence and reliability
            avg_confidence = sum([s.confidence for s in self.all_sources]) / max(len(self.all_sources), 1)
            confidence_desc = "high" if avg_confidence > 0.8 else "moderate" if avg_confidence > 0.6 else "basic"
            summary_points.append(
                f"Analysis confidence level: {confidence_desc} ({avg_confidence:.1%}) based on "
                f"source diversity and API integration quality"
            )
            
        except Exception as e:
            logger.error(f"Error generating fused summary: {e}")
            summary_points.append(f"Research completed with {len(self.all_sources)} sources analyzed")
        
        return summary_points[:6]  # Limit to 6 points
    
    def _generate_follow_up_questions(self) -> List[str]:
        """Generate contextual follow-up research questions"""
        questions = []
        
        try:
            if self.current_topic:
                # Generate topic-specific questions
                base_questions = [
                    f"What are the latest developments and trends in {self.current_topic}?",
                    f"How does {self.current_topic} compare to alternative approaches or solutions?",
                    f"What are the practical applications and real-world implementations of {self.current_topic}?",
                    f"What are the potential challenges or limitations of {self.current_topic}?",
                    f"Who are the key researchers, companies, or organizations working on {self.current_topic}?"
                ]
                
                # Add questions based on available sources
                if any(s.source_type == 'youtube_transcript' for s in self.all_sources):
                    questions.append(f"Are there any recent conferences or presentations about {self.current_topic}?")
                
                if any(s.source_type == 'reddit' for s in self.all_sources):
                    questions.append(f"What do community discussions reveal about public opinion on {self.current_topic}?")
                
                if any(s.source_type == 'image' for s in self.all_sources):
                    questions.append(f"What visual documentation or diagrams exist for {self.current_topic}?")
                
                # Select most relevant questions
                questions.extend(base_questions)
                
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            questions = [
                "What additional sources could provide more comprehensive coverage?",
                "How has this topic evolved over the past year?",
                "What are the main debates or controversies surrounding this topic?"
            ]
        
        return questions[:5]  # Limit to 5 questions
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get detailed research statistics"""
        if not self.all_sources:
            return {}
        
        stats = {
            'total_sources': len(self.all_sources),
            'source_breakdown': {},
            'api_usage': {},
            'confidence_stats': {
                'average': sum(s.confidence for s in self.all_sources) / len(self.all_sources),
                'highest': max(s.confidence for s in self.all_sources),
                'lowest': min(s.confidence for s in self.all_sources)
            }
        }
        
        # Count sources by type
        for source in self.all_sources:
            source_type = source.source_type
            stats['source_breakdown'][source_type] = stats['source_breakdown'].get(source_type, 0) + 1
        
        # API usage statistics
        api_status = self.config.get_api_status()
        stats['api_usage'] = {
            'available_apis': sum(1 for status in api_status.values() if status),
            'total_apis': len(api_status),
            'api_details': api_status
        }
        
        return stats
    
    def export_research_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive research report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stats = self.get_research_statistics()
        
        report = f"""
Multi-Modal Research Assistant - Comprehensive Report
=====================================================
Generated: {timestamp}
Topic: {self.current_topic or 'N/A'}
API Integration Status: {stats.get('api_usage', {}).get('available_apis', 0)}/{stats.get('api_usage', {}).get('total_apis', 0)} APIs active

EXECUTIVE SUMMARY
================
Sources Analyzed: {len(self.all_sources)}
Average Confidence: {stats.get('confidence_stats', {}).get('average', 0):.2f}
Source Types: {len(stats.get('source_breakdown', {}))}

SOURCE BREAKDOWN
===============
"""
        
        for source_type, count in stats.get('source_breakdown', {}).items():
            report += f"- {source_type}: {count} sources\n"
        
        report += f"""

API INTEGRATION STATUS
=====================
"""
        for api_name, status in stats.get('api_usage', {}).get('api_details', {}).items():
            status_symbol = "✅" if status else "❌"
            report += f"{status_symbol} {api_name.upper()}: {'Active' if status else 'Not configured'}\n"
        
        report += f"""

DETAILED SOURCE ANALYSIS
========================
"""
        
        for i, source in enumerate(self.all_sources, 1):
            report += f"""
{i}. {source.title}
   Type: {source.source_type}
   URL: {source.url}
   Confidence: {source.confidence:.2f}
   Content: {source.content[:300]}...
   Metadata: {source.metadata}
"""
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Report exported to: {filename}")
            except Exception as e:
                logger.error(f"Failed to export report: {e}")
        
        return report

# Export the main class for easy importing
__all__ = ['AgenticResearchAssistant', 'ResearchInsight', 'ResearchSource']