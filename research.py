"""
Enhanced Multi-Modal Research Assistant with FIXED YouTube Integration
This version addresses the XML parsing issues and transcript extraction failures
"""

import asyncio
import aiohttp
import json
import os
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings
import requests

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
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
        TooManyRequests, YouTubeRequestFailed
    )
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

from tenacity import retry, stop_after_attempt, wait_exponential

# New modules for summarization, image generation, and report synthesis
try:
    from summarization import summarize_transcript
    from image_generation import StableDiffusionGenerator
    from report import synthesize_report
    NEW_FEATURES_AVAILABLE = True
except Exception as _e:
    NEW_FEATURES_AVAILABLE = False
    logger.warning(f"Optional modules not fully available: {_e}")

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
    report_executive_summary: str = field(default="")
    report_key_findings: List[str] = field(default_factory=list)
    report_visuals: List[str] = field(default_factory=list)
    report_references: List[str] = field(default_factory=list)

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
    
    async def comprehensive_search(self, query: str) -> List[ResearchSource]:
        """Combine multiple search engines for comprehensive results"""
        logger.info(f"Starting comprehensive search for: {query}")
        
        # Create search tasks - prioritize based on API availability
        tasks = []
        
        # Add API-based searches if available
        if self.config.GOOGLE_CSE_API_KEY and self.config.GOOGLE_CSE_ID:
            tasks.append(self.search_google_cse(query))
        
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
    """FIXED: Enhanced video and audio processing with robust YouTube API"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.whisper_model = None
        
    def extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL with comprehensive cleaning"""
        try:
            video_url = video_url.strip()
            
            # Remove tracking parameters that can cause issues
            video_url = re.sub(r'[?&]si=[^&]*', '', video_url)
            video_url = re.sub(r'[?&]t=[^&]*', '', video_url)
            video_url = re.sub(r'[?&]list=[^&]*', '', video_url)
            video_url = re.sub(r'[?&]index=[^&]*', '', video_url)
            
            patterns = [
                r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
                r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
                r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
                r'(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})',
                r'(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})',
                r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_url)
                if match:
                    video_id = match.group(1)
                    # Validate video ID format
                    if len(video_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', video_id):
                        logger.info(f"✅ Extracted video ID: {video_id}")
                        return video_id
            
            logger.error(f"❌ Could not extract valid video ID from: {video_url}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting video ID: {e}")
            return None
    
    def check_video_accessibility(self, video_id: str) -> Dict[str, Any]:
        """Enhanced video accessibility check"""
        try:
            check_url = f"https://www.youtube.com/watch?v={video_id}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(check_url, timeout=15, headers=headers)
            
            if response.status_code != 200:
                return {
                    'accessible': False,
                    'reason': f'HTTP {response.status_code}',
                    'suggestion': 'Video may not exist or be unavailable'
                }
            
            page_content = response.text.lower()
            
            # Check for restrictions
            restrictions = [
                ('this video is private', 'Video is private'),
                ('video unavailable', 'Video unavailable'),
                ('this video has been removed', 'Video removed'),
                ('sign in to confirm your age', 'Age-restricted'),
                ('this video is not available', 'Geo-restricted'),
                ('error occurred', 'YouTube error')
            ]
            
            for indicator, reason in restrictions:
                if indicator in page_content:
                    return {
                        'accessible': False,
                        'reason': reason,
                        'suggestion': 'Try a different public video'
                    }
            
            return {'accessible': True, 'reason': 'Video appears accessible'}
            
        except Exception as e:
            return {
                'accessible': False,
                'reason': f'Check failed: {str(e)}',
                'suggestion': 'Network or connectivity issue'
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_youtube_transcript(self, video_url: str) -> Optional[ResearchSource]:
        """FIXED: YouTube transcript extraction with comprehensive error handling"""
        
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            logger.error("❌ YouTube transcript API not available")
            return self._try_alternative_methods(video_url)
        
        # Extract and validate video ID
        video_id = self.extract_video_id(video_url)
        if not video_id:
            logger.error(f"❌ Invalid YouTube URL: {video_url}")
            return None
        
        logger.info(f"🎯 Processing YouTube video: {video_id}")
        
        # Check accessibility first
        accessibility = self.check_video_accessibility(video_id)
        if not accessibility['accessible']:
            logger.error(f"❌ Video not accessible: {accessibility['reason']}")
            return None
        
        logger.info("✅ Video accessibility confirmed")
        
        # Initialize API with enhanced error handling
        try:
            api = YouTubeTranscriptApi()
        except Exception as init_error:
            logger.error(f"❌ API initialization failed: {init_error}")
            return self._try_alternative_methods(video_url)
        
        transcript_data = None
        extraction_info = {}
        
        # Strategy 1: Direct language-specific requests with wider range
        logger.info("🔄 Strategy 1: Direct language extraction...")
        language_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-IN']
        
        for lang_code in language_codes:
            try:
                transcript_data = api.get_transcript(video_id, languages=[lang_code])
                extraction_info = {
                    'method': 'direct_language',
                    'language': lang_code,
                    'type': 'targeted'
                }
                logger.info(f"✅ Direct success with {lang_code}")
                break
            except NoTranscriptFound:
                continue
            except TranscriptsDisabled:
                logger.error(f"❌ Transcripts disabled for video: {video_id}")
                return None
            except VideoUnavailable:
                logger.error(f"❌ Video unavailable: {video_id}")
                return None
            except Exception as e:
                logger.debug(f"Direct {lang_code} failed: {type(e).__name__}: {e}")
                continue
        
        # Strategy 2: Enhanced transcript listing with better error handling
        if not transcript_data:
            logger.info("🔄 Strategy 2: Enhanced transcript listing...")
            try:
                # Use a fresh API instance
                api_fresh = YouTubeTranscriptApi()
                transcript_list = api_fresh.list_transcripts(video_id)
                
                # Convert to list safely
                available_transcripts = []
                try:
                    for transcript in transcript_list:
                        available_transcripts.append({
                            'language': transcript.language,
                            'language_code': transcript.language_code,
                            'is_generated': transcript.is_generated,
                            'transcript_obj': transcript
                        })
                except Exception as conversion_error:
                    logger.error(f"❌ Error converting transcript list: {conversion_error}")
                    return self._try_alternative_methods(video_url)
                
                if not available_transcripts:
                    logger.error("❌ No transcripts available")
                    return self._try_alternative_methods(video_url)
                
                logger.info(f"📋 Found {len(available_transcripts)} transcript(s)")
                
                # Enhanced prioritization
                priority_order = []
                
                # Priority 1: Manual English
                for t in available_transcripts:
                    if t['language_code'].startswith('en') and not t['is_generated']:
                        priority_order.append(t)
                
                # Priority 2: Auto English
                for t in available_transcripts:
                    if t['language_code'].startswith('en') and t['is_generated']:
                        priority_order.append(t)
                
                # Priority 3: Manual other languages
                for t in available_transcripts:
                    if not t['language_code'].startswith('en') and not t['is_generated']:
                        priority_order.append(t)
                
                # Priority 4: Auto other languages
                for t in available_transcripts:
                    if not t['language_code'].startswith('en') and t['is_generated']:
                        priority_order.append(t)
                
                # Try each transcript with individual error handling
                for transcript_info in priority_order:
                    try:
                        logger.info(f"🔄 Trying {transcript_info['language']} ({'auto' if transcript_info['is_generated'] else 'manual'})")
                        
                        # Add delay to avoid rate limiting
                        time.sleep(0.5)
                        
                        transcript_data = transcript_info['transcript_obj'].fetch()
                        extraction_info = {
                            'method': 'listing_enhanced',
                            'language': transcript_info['language'],
                            'language_code': transcript_info['language_code'],
                            'type': 'auto' if transcript_info['is_generated'] else 'manual'
                        }
                        logger.info(f"✅ Listing success with {transcript_info['language']}")
                        break
                        
                    except Exception as fetch_error:
                        logger.warning(f"⚠️ Fetch failed for {transcript_info['language']}: {type(fetch_error).__name__}")
                        continue
                        
            except TranscriptsDisabled:
                logger.error(f"❌ Transcripts disabled for video: {video_id}")
                return None
            except VideoUnavailable:
                logger.error(f"❌ Video unavailable: {video_id}")
                return None
            except Exception as listing_error:
                logger.error(f"❌ Listing failed: {type(listing_error).__name__}: {listing_error}")
        
        # Strategy 3: Simplified no-filter approach with better error handling
        if not transcript_data:
            logger.info("🔄 Strategy 3: Simplified no-filter approach...")
            try:
                # Create a completely fresh API instance
                api_final = YouTubeTranscriptApi()
                
                # Add a longer delay
                time.sleep(1)
                
                transcript_data = api_final.get_transcript(video_id)
                extraction_info = {
                    'method': 'no_filter_simple',
                    'language': 'auto-detected',
                    'type': 'unknown'
                }
                logger.info("✅ No-filter success")
                
            except Exception as final_error:
                logger.error(f"❌ No-filter failed: {type(final_error).__name__}: {final_error}")
                return self._try_alternative_methods(video_url)
        
        # Process results
        if not transcript_data:
            logger.error(f"❌ All extraction strategies failed for: {video_id}")
            return self._try_alternative_methods(video_url)
        
        # Enhanced transcript processing
        transcript_text = self._process_transcript_safely(transcript_data)
        
        if not transcript_text or len(transcript_text.strip()) < 20:
            logger.error(f"❌ Transcript too short: {len(transcript_text) if transcript_text else 0} chars")
            return self._try_alternative_methods(video_url)
        
        # Get video metadata
        video_info = self._get_video_metadata_safe(video_id)
        title = video_info.get('title', f'YouTube Video {video_id}')
        
        logger.info(f"🎉 SUCCESS: {len(transcript_text)} characters extracted from '{title}'")
        
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
                'language_code': extraction_info.get('language_code', 'unknown'),
                'transcript_type': extraction_info.get('type', 'unknown'),
                'transcript_length': len(transcript_text),
                'video_info': video_info
            }
        )
    
    def _process_transcript_safely(self, transcript_data: List[Dict]) -> str:
        """Enhanced transcript processing with better error handling"""
        try:
            if not transcript_data or not isinstance(transcript_data, list):
                logger.error("❌ Invalid transcript data structure")
                return ""
            
            # Process segments with enhanced cleaning
            text_segments = []
            total_duration = 0
            
            for i, segment in enumerate(transcript_data):
                try:
                    if not isinstance(segment, dict):
                        logger.debug(f"Skipping non-dict segment {i}")
                        continue
                    
                    text = segment.get('text', '').strip()
                    start_time = segment.get('start', 0)
                    duration = segment.get('duration', 0)
                    
                    if not text:
                        continue
                    
                    # Enhanced text cleaning
                    original_text = text
                    
                    # Remove subtitle artifacts
                    text = re.sub(r'\[.*?\]', '', text)  # [Music], [Applause]
                    text = re.sub(r'\(.*?\)', '', text)  # (inaudible)
                    text = re.sub(r'<.*?>', '', text)   # HTML tags
                    text = re.sub(r'♪.*?♪', '', text)   # Music symbols
                    text = re.sub(r'>>.*?<<', '', text) # Speaker indicators
                    
                    # Clean punctuation and spacing
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    
                    # Only keep meaningful text
                    if text and len(text) > 2 and not re.match(r'^[^\w]*$', text):
                        text_segments.append({
                            'text': text,
                            'start': start_time,
                            'duration': duration,
                            'original': original_text
                        })
                        total_duration = max(total_duration, start_time + duration)
                
                except Exception as segment_error:
                    logger.debug(f"Error processing segment {i}: {segment_error}")
                    continue
            
            if not text_segments:
                logger.error("❌ No valid text segments after processing")
                return ""
            
            logger.info(f"✅ Processed {len(text_segments)} valid segments")
            
            # Intelligent text combination
            combined_text = []
            current_paragraph = ""
            last_end_time = 0
            
            for segment in text_segments:
                text = segment['text']
                start_time = segment['start']
                duration = segment['duration']
                
                # Determine if this should start a new paragraph
                time_gap = start_time - last_end_time
                
                if time_gap > 3:  # 3+ second gap = new paragraph
                    if current_paragraph.strip():
                        combined_text.append(current_paragraph.strip())
                    current_paragraph = text
                else:
                    # Continue current paragraph
                    if current_paragraph:
                        # Smart punctuation handling
                        if current_paragraph.rstrip()[-1:] in '.!?':
                            current_paragraph += " " + text
                        elif text[0:1].isupper():
                            current_paragraph += ". " + text
                        else:
                            current_paragraph += " " + text
                    else:
                        current_paragraph = text
                
                last_end_time = start_time + duration
            
            # Add final paragraph
            if current_paragraph.strip():
                combined_text.append(current_paragraph.strip())
            
            # Join paragraphs
            full_transcript = '\n\n'.join(combined_text)
            
            # Final cleanup
            full_transcript = re.sub(r'\n\s*\n', '\n\n', full_transcript)  # Clean up multiple newlines
            full_transcript = re.sub(r'([.!?])\s*([.!?])', r'\1', full_transcript)  # Remove duplicate punctuation
            full_transcript = full_transcript.strip()
            
            logger.info(f"✅ Final transcript: {len(full_transcript)} characters, ~{total_duration/60:.1f} min duration")
            
            return full_transcript
            
        except Exception as e:
            logger.error(f"❌ Transcript processing failed: {e}")
            return ""
    
    def _get_video_metadata_safe(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata with multiple fallback methods"""
        metadata = {'video_id': video_id}
        
        # Method 1: yt-dlp (most comprehensive)
        if YT_DLP_AVAILABLE:
            try:
                import yt_dlp
                
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'skip_download': True,
                    'extractor_args': {'youtube': {'skip': ['dash', 'hls']}}
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    url = f"https://youtube.com/watch?v={video_id}"
                    info = ydl.extract_info(url, download=False)
                    
                    if info:
                        metadata.update({
                            'title': info.get('title', f'YouTube Video {video_id}'),
                            'description': (info.get('description', '') or '')[:500],
                            'duration': info.get('duration', 0),
                            'uploader': info.get('uploader', ''),
                            'upload_date': info.get('upload_date', ''),
                            'view_count': info.get('view_count', 0)
                        })
                        logger.info(f"✅ Metadata extracted via yt-dlp: {metadata['title']}")
                        return metadata
                        
            except Exception as e:
                logger.debug(f"yt-dlp metadata failed: {e}")
        
        # Method 2: Web scraping fallback
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                f"https://www.youtube.com/watch?v={video_id}", 
                headers=headers, 
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.text
                
                # Extract title with multiple patterns
                title_patterns = [
                    r'"title":"([^"]*)"',
                    r'<title>([^<]*)</title>',
                    r'property="og:title" content="([^"]*)"'
                ]
                
                for pattern in title_patterns:
                    match = re.search(pattern, content)
                    if match:
                        title = match.group(1)
                        # Clean title
                        title = title.replace('\\u0026', '&')
                        title = title.replace(' - YouTube', '')
                        title = re.sub(r'\\[ux][0-9a-fA-F]{4}', '', title)
                        metadata['title'] = title.strip()
                        break
                
                # Extract view count
                view_patterns = [
                    r'"viewCount":"(\d+)"',
                    r'"views":{"simpleText":"([\d,]+)"'
                ]
                
                for pattern in view_patterns:
                    match = re.search(pattern, content)
                    if match:
                        try:
                            views = match.group(1).replace(',', '')
                            metadata['view_count'] = int(views)
                            break
                        except:
                            continue
                            
                logger.info(f"✅ Metadata extracted via web scraping")
                
        except Exception as e:
            logger.debug(f"Web scraping metadata failed: {e}")
        
        # Ensure we have at least a basic title
        if 'title' not in metadata:
            metadata['title'] = f'YouTube Video {video_id}'
        
        return metadata
    
    def _try_alternative_methods(self, video_url: str) -> Optional[ResearchSource]:
        """Try alternative transcript extraction methods"""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return None
        
        logger.info("🔄 Trying alternative transcript extraction...")
        
        # Alternative method: yt-dlp subtitle extraction
        if YT_DLP_AVAILABLE:
            try:
                transcript_text = self._extract_subtitles_ytdlp(video_id)
                if transcript_text and len(transcript_text.strip()) > 50:
                    metadata = self._get_video_metadata_safe(video_id)
                    
                    logger.info(f"✅ Alternative extraction successful: {len(transcript_text)} chars")
                    
                    return ResearchSource(
                        title=metadata.get('title', f'YouTube Video {video_id}'),
                        content=transcript_text,
                        url=video_url,
                        source_type='youtube_transcript_alt',
                        confidence=0.7,
                        metadata={
                            'video_id': video_id,
                            'extraction_method': 'alternative_ytdlp',
                            'transcript_length': len(transcript_text),
                            'video_info': metadata
                        }
                    )
                    
            except Exception as e:
                logger.debug(f"Alternative yt-dlp failed: {e}")
        
        logger.error(f"❌ All transcript extraction methods failed for: {video_id}")
        return None
    
    def _extract_subtitles_ytdlp(self, video_id: str) -> Optional[str]:
        """Extract subtitles using yt-dlp with enhanced options"""
        try:
            import yt_dlp
            import tempfile
            import os
            
            logger.info("🔄 yt-dlp subtitle extraction...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', 'en-US', 'en-GB'],
                    'subtitlesformat': 'vtt/srt/best',
                    'skip_download': True,
                    'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True
                }
                
                url = f"https://youtube.com/watch?v={video_id}"
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                    # Look for subtitle files
                    subtitle_files = []
                    for ext in ['vtt', 'srt', 'ttml']:
                        subtitle_files.extend(Path(temp_dir).glob(f"*.{ext}"))
                    
                    # Prioritize English subtitles
                    for subtitle_file in subtitle_files:
                        if 'en' in subtitle_file.name.lower():
                            with open(subtitle_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                transcript = self._parse_subtitle_content(content)
                                if transcript and len(transcript) > 50:
                                    logger.info(f"✅ Subtitle extraction successful from {subtitle_file.name}")
                                    return transcript
            
            return None
            
        except Exception as e:
            logger.debug(f"yt-dlp subtitle extraction failed: {e}")
            return None
    
    def _parse_subtitle_content(self, content: str) -> Optional[str]:
        """Parse VTT or SRT subtitle content with enhanced cleaning"""
        try:
            lines = content.split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines, timestamps, and headers
                if not line:
                    continue
                if '-->' in line:  # Timestamp line
                    continue
                if line.isdigit():  # SRT sequence number
                    continue
                if line.startswith(('WEBVTT', 'NOTE', 'Kind:', 'Language:')):
                    continue
                
                # Clean the text line
                if line and not re.match(r'^\d+$', line):
                    # Remove HTML/XML tags
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    # Remove timestamp markers
                    clean_line = re.sub(r'\d{2}:\d{2}:\d{2}[\.,]\d{3}', '', clean_line)
                    # Clean whitespace
                    clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                    
                    if clean_line and len(clean_line) > 2:
                        text_lines.append(clean_line)
            
            if not text_lines:
                return None
            
            # Combine and clean final transcript
            transcript = ' '.join(text_lines)
            transcript = re.sub(r'\s+', ' ', transcript).strip()
            
            return transcript if len(transcript) > 50 else None
            
        except Exception as e:
            logger.error(f"Subtitle parsing failed: {e}")
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
                image = Image.open(image_path)
            else:
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
    
    def analyze_image(self, image_path: str) -> ResearchSource:
        """Comprehensive image analysis"""
        logger.info(f"Analyzing image: {image_path}")
        
        ocr_text = self.extract_text_from_image(image_path)
        
        # Combine OCR results
        content_parts = []
        if ocr_text:
            content_parts.append(f"Extracted Text: {ocr_text}")
        else:
            content_parts.append("Visual content detected (no text extracted)")
        
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
                'file_name': image_name
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

class AgenticResearchAssistant:
    """Enhanced main research assistant class with fixed YouTube processing"""
    
    def __init__(self):
        self.config = AgenticResearchConfig()
        self.web_search = WebSearchEngine(self.config)
        self.video_processor = VideoAudioProcessor(self.config)
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        
        self.all_sources = []
        self.current_topic = None
        
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
        """Conduct comprehensive research with enhanced error handling"""
        
        logger.info(f"🚀 Starting research on topic: {topic}")
        self.current_topic = topic
        self.all_sources = []
        
        # Phase 1: Web search
        logger.info("📍 Phase 1: Web search...")
        try:
            async with self.web_search as search_engine:
                web_sources = await search_engine.comprehensive_search(topic)
                self.all_sources.extend(web_sources)
                logger.info(f"✅ Web search: {len(web_sources)} sources")
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
        
        # Phase 2: Process text sources
        if text_sources and text_sources.strip():
            logger.info("📍 Phase 2: Processing text sources...")
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
                logger.info("✅ User text processed")
            except Exception as e:
                logger.error(f"❌ Text processing error: {e}")
        
        # Phase 3: Process videos with FIXED extraction
        if video_urls:
            logger.info(f"📍 Phase 3: Processing {len(video_urls)} videos...")
            for i, url in enumerate(video_urls, 1):
                if url.strip():
                    try:
                        logger.info(f"🎥 Processing video {i}/{len(video_urls)}: {url}")
                        
                        # Use the FIXED transcript extraction
                        source = self.video_processor.get_youtube_transcript(url.strip())
                        
                        if source:
                            self.all_sources.append(source)
                            logger.info(f"✅ Video {i} processed: {len(source.content)} chars extracted")
                        else:
                            logger.warning(f"⚠️ Video {i} failed: No transcript available")
                            
                    except Exception as e:
                        logger.error(f"❌ Video {i} error: {e}")
        
        # Phase 4: Process images
        if image_paths:
            logger.info(f"📍 Phase 4: Processing {len(image_paths)} images...")
            for i, path in enumerate(image_paths, 1):
                try:
                    logger.info(f"🖼️ Processing image {i}/{len(image_paths)}")
                    source = self.image_processor.analyze_image(path)
                    self.all_sources.append(source)
                    logger.info(f"✅ Image {i} processed")
                except Exception as e:
                    logger.error(f"❌ Image {i} error: {e}")
        
        # Phase 5: Process audio (if Whisper available)
        if audio_paths and WHISPER_AVAILABLE:
            logger.info(f"📍 Phase 5: Processing {len(audio_paths)} audio files...")
            for i, path in enumerate(audio_paths, 1):
                try:
                    logger.info(f"🎵 Processing audio {i}/{len(audio_paths)}")
                    source = self.video_processor.transcribe_audio_with_whisper(path)
                    if source:
                        self.all_sources.append(source)
                        logger.info(f"✅ Audio {i} processed")
                    else:
                        logger.warning(f"⚠️ Audio {i} failed")
                except Exception as e:
                    logger.error(f"❌ Audio {i} error: {e}")
        
        # Phase 6: Generate insights
        logger.info("📍 Phase 6: Generating insights...")
        insights = self._generate_insights()

        # New: Build transcript-level summary and images, then synthesize report
        if TRANSFORMERS_AVAILABLE: # Changed from NEW_FEATURES_AVAILABLE to TRANSFORMERS_AVAILABLE
            try:
                # Aggregate transcript text and links
                transcript_sources = [s for s in self.all_sources if s.source_type.startswith('youtube')]
                combined_transcript = "\n\n".join(s.content for s in transcript_sources)
                video_links = [s.url for s in transcript_sources]

                report_overview = ""
                timeline = []
                takeaways = []
                if combined_transcript.strip():
                    tsum = summarize_transcript(combined_transcript)
                    report_overview = tsum.get("overview", "")
                    timeline = tsum.get("timeline_highlights", [])
                    takeaways = tsum.get("key_takeaways", [])

                # Use fused_summary + web snippets for additional findings
                web_snippets = []
                for s in self.all_sources:
                    if s.source_type in ['duckduckgo', 'google_cse', 'wikipedia']:
                        snippet = f"[{s.title}] {s.content[:200]}".strip()
                        web_snippets.append(snippet)

                # Generate images with GPU preference
                img_paths: List[str] = []
                try:
                    images_dir = Path("reports") / "images"
                    # Force GPU usage
                    sd = StableDiffusionGenerator()
                    if hasattr(sd, 'device') and sd.device == "cuda":
                        print(f"🎨 Generating images on GPU: {torch.cuda.get_device_name()}")
                    else:
                        print("⚠️ Image generation will use CPU (slower)")
                    
                    overview_for_prompt = (report_overview or " ").strip()[:500]
                    generated = sd.generate_for_summary(overview_for_prompt, images_dir, base_name="summary")
                    for g in generated:
                        img_paths.append(g.path)
                except Exception as ie:
                    logger.warning(f"Image generation skipped: {ie}")

                # Synthesize report
                from report import synthesize_report
                report = synthesize_report(
                    topic=topic,
                    video_overview=report_overview or "",
                    timeline_highlights=timeline,
                    takeaways=takeaways,
                    web_snippets=web_snippets,
                    image_paths=img_paths,
                    youtube_links=video_links,
                )

                insights.report_executive_summary = report.executive_summary
                insights.report_key_findings = report.key_findings
                insights.report_visuals = report.visuals
                insights.report_references = report.references
            except Exception as e:
                logger.warning(f"Report synthesis failed: {e}")
        
        logger.info(f"🎉 Research completed: {insights.sources_count} sources, {insights.confidence_score:.1%} confidence")
        return insights
    
    def transcribe_audio_with_whisper(self, audio_path: str) -> Optional[ResearchSource]:
        """Transcribe audio file using Whisper"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available for audio transcription")
            return None
            
        try:
            logger.info(f"Transcribing audio with Whisper: {audio_path}")
            
            if not self.whisper_model:
                logger.info(f"Loading Whisper model: {self.config.WHISPER_MODEL}")
                self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
            
            result = self.whisper_model.transcribe(audio_path)
            
            file_name = Path(audio_path).name if isinstance(audio_path, str) else getattr(audio_path, 'name', 'audio_file')
            
            return ResearchSource(
                title=f"Audio Transcription: {file_name}",
                content=result['text'],
                url=f"file://{audio_path}",
                source_type='audio_transcription',
                confidence=0.9,
                metadata={
                    'model': self.config.WHISPER_MODEL,
                    'language': result.get('language', 'unknown'),
                    'file_name': file_name
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return None
    
    def _generate_insights(self) -> ResearchInsight:
        """Generate comprehensive insights from all sources"""
        text_insights = []
        video_insights = []
        image_insights = []
        
        logger.info(f"Generating insights from {len(self.all_sources)} sources")
        
        for source in self.all_sources:
            try:
                if source.source_type in ['google_cse', 'duckduckgo', 'wikipedia', 'user_text']:
                    summary = self.text_processor.summarize_text(source.content)
                    insight = f"[{source.title}] {summary}"
                    text_insights.append(insight)
                
                elif source.source_type in ['youtube_transcript', 'youtube_transcript_alt', 'audio_transcription']:
                    summary = self.text_processor.summarize_text(source.content)
                    insight = f"[{source.title}] {summary}"
                    video_insights.append(insight)
                
                elif source.source_type == 'image':
                    image_insights.append(f"[{source.title}] {source.content}")
                    
            except Exception as e:
                logger.error(f"Error generating insight for {source.title}: {e}")
        
        # Generate fused summary
        all_content = "\n".join([s.content for s in self.all_sources[:10]])
        fused_summary = self._generate_fused_summary(all_content)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions()
        
        # Calculate confidence
        if self.all_sources:
            confidence_score = sum([s.confidence for s in self.all_sources]) / len(self.all_sources)
        else:
            confidence_score = 0.0
        
        return ResearchInsight(
            text_insights=text_insights[:8],
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
                    elif source_type == 'duckduckgo':
                        type_descriptions.append(f"{count} DuckDuckGo results")
                    elif source_type in ['youtube_transcript', 'youtube_transcript_alt']:
                        type_descriptions.append(f"{count} YouTube transcripts")
                    elif source_type == 'image':
                        type_descriptions.append(f"{count} image analyses")
                    elif source_type == 'wikipedia':
                        type_descriptions.append(f"{count} Wikipedia articles")
                    else:
                        type_descriptions.append(f"{count} {source_type} sources")
                
                summary_points.append(f"Multi-modal analysis includes: {', '.join(type_descriptions)}")
            
            # Content-based summary
            if content and len(content) > 100:
                main_summary = self.text_processor.summarize_text(content[:3000], max_length=150)
                if main_summary and len(main_summary) > 20:
                    summary_points.append(f"Key findings: {main_summary}")
            
            # YouTube-specific insights
            youtube_sources = [s for s in self.all_sources if 'youtube' in s.source_type]
            if youtube_sources:
                total_duration = 0
                for source in youtube_sources:
                    video_info = source.metadata.get('video_info', {})
                    duration = video_info.get('duration', 0)
                    if duration:
                        total_duration += duration
                
                if total_duration > 0:
                    summary_points.append(
                        f"Analyzed {len(youtube_sources)} YouTube video(s) with ~{total_duration//60:.0f} minutes of content"
                    )
            
            # Confidence and reliability
            avg_confidence = sum([s.confidence for s in self.all_sources]) / max(len(self.all_sources), 1)
            confidence_desc = "high" if avg_confidence > 0.8 else "moderate" if avg_confidence > 0.6 else "basic"
            summary_points.append(
                f"Analysis confidence: {confidence_desc} ({avg_confidence:.1%}) based on source quality and diversity"
            )
            
        except Exception as e:
            logger.error(f"Error generating fused summary: {e}")
            summary_points.append(f"Research completed with {len(self.all_sources)} sources analyzed")
        
        return summary_points[:6]
    
    def _generate_follow_up_questions(self) -> List[str]:
        """Generate contextual follow-up research questions"""
        questions = []
        
        try:
            if self.current_topic:
                base_questions = [
                    f"What are the latest developments in {self.current_topic}?",
                    f"How does {self.current_topic} compare to alternative approaches?",
                    f"What are the practical applications of {self.current_topic}?",
                    f"What challenges or limitations exist for {self.current_topic}?",
                    f"Who are the key players or organizations in {self.current_topic}?"
                ]
                
                # Add source-specific questions
                if any(s.source_type.startswith('youtube') for s in self.all_sources):
                    questions.append(f"Are there recent presentations or demos about {self.current_topic}?")
                
                if any(s.source_type == 'image' for s in self.all_sources):
                    questions.append(f"What visual documentation exists for {self.current_topic}?")
                
                questions.extend(base_questions)
                
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            questions = [
                "What additional sources could provide more coverage?",
                "How has this topic evolved recently?",
                "What are the main debates around this topic?"
            ]
        
        return questions[:5]
    
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
            },
            'youtube_stats': {}
        }
        
        # Count sources by type
        for source in self.all_sources:
            source_type = source.source_type
            stats['source_breakdown'][source_type] = stats['source_breakdown'].get(source_type, 0) + 1
        
        # YouTube-specific statistics
        youtube_sources = [s for s in self.all_sources if 'youtube' in s.source_type]
        if youtube_sources:
            total_chars = sum(len(s.content) for s in youtube_sources)
            extraction_methods = {}
            for source in youtube_sources:
                method = source.metadata.get('extraction_method', 'unknown')
                extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            stats['youtube_stats'] = {
                'videos_processed': len(youtube_sources),
                'total_transcript_chars': total_chars,
                'extraction_methods': extraction_methods
            }
        
        # API usage statistics
        api_status = self.config.get_api_status()
        stats['api_usage'] = {
            'available_apis': sum(1 for status in api_status.values() if status),
            'total_apis': len(api_status),
            'api_details': api_status
        }
        
        return stats

# Export the main classes
__all__ = ['AgenticResearchAssistant', 'ResearchInsight', 'ResearchSource', 'VideoAudioProcessor']