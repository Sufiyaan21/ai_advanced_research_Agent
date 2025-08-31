# Complete Agentic Multi-Modal Research Assistant
# High-performance implementation with proper error handling

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
    """Configuration class for API keys and settings"""
    
    def __init__(self):
        try:
            from config import get_config
            config = get_config()
            
            self.OPENAI_API_KEY = config.api_config.openai_api_key
            self.GOOGLE_CSE_API_KEY = config.api_config.google_cse_api_key
            self.GOOGLE_CSE_ID = config.api_config.google_cse_search_engine_id
            self.REDDIT_CLIENT_ID = config.api_config.reddit_client_id
            self.REDDIT_CLIENT_SECRET = config.api_config.reddit_client_secret
            self.REDDIT_USER_AGENT = config.api_config.reddit_user_agent or "MultiModalResearchAssistant/1.0"
            self.ASSEMBLYAI_API_KEY = config.api_config.assemblyai_api_key
            self.SERPAPI_KEY = config.api_config.serpapi_api_key
            
            self.WHISPER_MODEL = config.model_config.whisper_model_size
            self.MAX_SEARCH_RESULTS = config.model_config.max_search_results
            self.MAX_VIDEO_DURATION = config.model_config.max_video_duration
            
            self.TEMP_DIR = config.model_config.temp_dir
            self.TEMP_DIR.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            # Set defaults
            self.OPENAI_API_KEY = None
            self.GOOGLE_CSE_API_KEY = None
            self.GOOGLE_CSE_ID = None
            self.REDDIT_CLIENT_ID = None
            self.REDDIT_CLIENT_SECRET = None
            self.REDDIT_USER_AGENT = "MultiModalResearchAssistant/1.0"
            self.ASSEMBLYAI_API_KEY = None
            self.SERPAPI_KEY = None
            self.WHISPER_MODEL = "base"
            self.MAX_SEARCH_RESULTS = 10
            self.MAX_VIDEO_DURATION = 600
            self.TEMP_DIR = Path("temp_files")
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
                    
            return sources
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_reddit(self, query: str) -> List[ResearchSource]:
        """Reddit search for community insights"""
        if not PRAW_AVAILABLE or not (self.config.REDDIT_CLIENT_ID and self.config.REDDIT_CLIENT_SECRET):
            return []
            
        try:
            logger.info(f"Searching Reddit for: {query}")
            sources = []
            
            reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent=self.config.REDDIT_USER_AGENT
            )
            
            subreddits = ['technology', 'science']
            
            for subreddit_name in subreddits[:1]:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, limit=2):
                        if submission.selftext and len(submission.selftext) > 100:
                            source = ResearchSource(
                                title=submission.title,
                                content=submission.selftext[:1500],
                                url=f"https://reddit.com{submission.permalink}",
                                source_type='reddit',
                                confidence=0.7,
                                metadata={'subreddit': subreddit_name}
                            )
                            sources.append(source)
                except Exception:
                    continue
                    
            return sources
            
        except Exception as e:
            logger.error(f"Reddit search failed: {str(e)}")
            return []
    
    async def comprehensive_search(self, query: str) -> List[ResearchSource]:
        """Combine multiple search engines for comprehensive results"""
        logger.info(f"Starting comprehensive search for: {query}")
        
        tasks = [
            self.search_duckduckgo(query),
            self.search_wikipedia(query),
            self.search_reddit(query)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_sources = []
        for result in results:
            if isinstance(result, list):
                all_sources.extend(result)
        
        # Remove duplicates
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        logger.info(f"Found {len(unique_sources)} unique sources")
        return unique_sources

class VideoAudioProcessor:
    """Advanced video and audio processing"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Load Whisper model for speech-to-text"""
        if not WHISPER_AVAILABLE:
            return
        if not self.whisper_model:
            try:
                self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
    
    def extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        try:
            patterns = [
                r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]+)',
                r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_url)
                if match:
                    return match.group(1)
            
            if len(video_url) == 11:
                return video_url
                
            return None
        except Exception:
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_youtube_transcript(self, video_url: str) -> Optional[ResearchSource]:
        """Extract transcript from YouTube video"""
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            return None
            
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                return None
            
            logger.info(f"Extracting YouTube transcript for video: {video_id}")
            
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in transcript_data])
            
            return ResearchSource(
                title=f"YouTube Video {video_id}",
                content=transcript_text,
                url=video_url,
                source_type='youtube_transcript',
                confidence=0.85,
                metadata={'video_id': video_id}
            )
            
        except Exception as e:
            logger.error(f"YouTube transcript extraction failed: {str(e)}")
            return None
    
    def transcribe_audio_with_whisper(self, audio_path: str) -> Optional[ResearchSource]:
        """Transcribe audio file using Whisper"""
        if not WHISPER_AVAILABLE:
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
                metadata={'model': self.config.WHISPER_MODEL}
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return None

class ImageProcessor:
    """Image analysis and OCR"""
    
    def __init__(self):
        self.blip_processor = None
        self.blip_model = None
        
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
            return ""
            
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""
    
    def generate_image_caption(self, image_path: str) -> str:
        """Generate caption for image"""
        if not TRANSFORMERS_AVAILABLE:
            return "Image caption generation not available"
            
        try:
            if not self.blip_processor:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            image = Image.open(image_path)
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Image captioning failed: {str(e)}")
            return "Could not generate caption"
    
    def analyze_image(self, image_path: str) -> ResearchSource:
        """Analyze image completely"""
        ocr_text = self.extract_text_from_image(image_path)
        caption = self.generate_image_caption(image_path)
        
        content = f"Caption: {caption}\n"
        if ocr_text:
            content += f"Extracted Text: {ocr_text}"
        
        return ResearchSource(
            title=f"Image: {Path(image_path).name}",
            content=content,
            url=f"file://{image_path}",
            source_type='image',
            confidence=0.8,
            metadata={'has_text': bool(ocr_text)}
        )

class TextProcessor:
    """Text analysis and summarization"""
    
    def __init__(self):
        self.summarizer = None
        self.sentiment_analyzer = None
        
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text using BART"""
        if not TRANSFORMERS_AVAILABLE:
            # Fallback to simple extraction
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
            
        try:
            if not self.summarizer:
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            if len(text) < 100:
                return text
                
            summary = self.summarizer(text[:1024], max_length=max_length, min_length=50, do_sample=False)
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return text[:max_length]
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
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
            return []
            
        try:
            doc = nlp(text[:5000])
            entities = list(set([ent.text for ent in doc.ents]))
            return entities[:10]
        except Exception:
            return []

class AgenticResearchAssistant:
    """Main research assistant class"""
    
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
        
    async def research_topic(
        self,
        topic: str,
        video_urls: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        text_sources: Optional[str] = None
    ) -> ResearchInsight:
        """Conduct comprehensive research on a topic"""
        
        logger.info(f"Starting research on topic: {topic}")
        self.current_topic = topic
        self.all_sources = []
        
        # Web search
        async with self.web_search as search_engine:
            web_sources = await search_engine.comprehensive_search(topic)
            self.all_sources.extend(web_sources)
        
        # Process text sources
        if text_sources:
            source = ResearchSource(
                title="User Provided Text",
                content=text_sources,
                url="user_input",
                source_type='text',
                confidence=0.9,
                metadata={}
            )
            self.all_sources.append(source)
        
        # Process videos
        if video_urls:
            for url in video_urls:
                source = self.video_processor.get_youtube_transcript(url)
                if source:
                    self.all_sources.append(source)
        
        # Process audio
        if audio_paths:
            for path in audio_paths:
                source = self.video_processor.transcribe_audio_with_whisper(path)
                if source:
                    self.all_sources.append(source)
        
        # Process images
        if image_paths:
            for path in image_paths:
                source = self.image_processor.analyze_image(path)
                self.all_sources.append(source)
        
        # Generate insights
        insights = self._generate_insights()
        
        return insights
    
    def _generate_insights(self) -> ResearchInsight:
        """Generate insights from all sources"""
        text_insights = []
        video_insights = []
        image_insights = []
        
        for source in self.all_sources:
            if source.source_type in ['web_search', 'wikipedia', 'reddit', 'text']:
                summary = self.text_processor.summarize_text(source.content)
                text_insights.append(f"[{source.title}] {summary}")
            
            elif source.source_type in ['youtube_transcript', 'audio_transcription']:
                summary = self.text_processor.summarize_text(source.content)
                video_insights.append(f"[{source.title}] {summary}")
            
            elif source.source_type == 'image':
                image_insights.append(f"[{source.title}] {source.content}")
        
        # Generate fused summary
        all_content = "\n".join([s.content for s in self.all_sources[:5]])
        fused_summary = self._generate_fused_summary(all_content)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions()
        
        # Calculate confidence
        confidence_score = sum([s.confidence for s in self.all_sources]) / max(len(self.all_sources), 1)
        
        return ResearchInsight(
            text_insights=text_insights[:5],
            video_insights=video_insights[:5],
            image_insights=image_insights[:5],
            fused_summary=fused_summary,
            follow_up_questions=follow_up_questions,
            confidence_score=min(confidence_score, 1.0),
            sources_count=len(self.all_sources)
        )
    
    def _generate_fused_summary(self, content: str) -> List[str]:
        """Generate fused summary points"""
        summary_points = []
        
        # Extract key themes
        if self.current_topic:
            summary_points.append(f"Research on '{self.current_topic}' reveals insights from {len(self.all_sources)} sources")
        
        # Add source diversity
        source_types = set([s.source_type for s in self.all_sources])
        if len(source_types) > 1:
            summary_points.append(f"Analysis includes {', '.join(source_types)} providing multi-modal perspective")
        
        # Add main finding
        if content:
            main_summary = self.text_processor.summarize_text(content[:2000])
            summary_points.append(main_summary)
        
        # Add confidence statement
        avg_confidence = sum([s.confidence for s in self.all_sources]) / max(len(self.all_sources), 1)
        summary_points.append(f"Overall confidence level: {avg_confidence:.1%} based on source quality")
        
        return summary_points[:5]
    
    def _generate_follow_up_questions(self) -> List[str]:
        """Generate follow-up research questions"""
        questions = []
        
        if self.current_topic:
            questions.append(f"What are the latest developments in {self.current_topic}?")
            questions.append(f"How does {self.current_topic} compare to alternative approaches?")
            questions.append(f"What are the practical applications of {self.current_topic}?")
        
        return questions[:3]
    
    def export_research_report(self, filename: Optional[str] = None) -> str:
        """Export research report as text"""
        report = f"""
Multi-Modal Research Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Topic: {self.current_topic or 'N/A'}

SOURCES ANALYZED: {len(self.all_sources)}
{'='*50}

"""
        for i, source in enumerate(self.all_sources, 1):
            report += f"\n{i}. {source.title}\n"
            report += f"   Type: {source.source_type}\n"
            report += f"   Confidence: {source.confidence:.2f}\n"
            report += f"   URL: {source.url}\n"
            report += f"   Summary: {source.content[:200]}...\n"
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report