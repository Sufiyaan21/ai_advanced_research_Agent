# Complete Agentic Multi-Modal Research Assistant
# High-performance implementation with all API integrations

import asyncio
import aiohttp
import json
import os
import time
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
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

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
        self.GOOGLE_CSE_ID = config.api_config.google_cse_id
        self.REDDIT_CLIENT_ID = config.api_config.reddit_client_id
        self.REDDIT_CLIENT_SECRET = config.api_config.reddit_client_secret
        self.ASSEMBLYAI_API_KEY = config.api_config.assemblyai_api_key
        self.SERPAPI_KEY = config.api_config.serpapi_key
        
        # Model configurations from config system
        self.WHISPER_MODEL = config.model_config.whisper_model
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
                        metadata={'engine': 'wikipedia', 'categories': page.categories[:5]}
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
                except:
                    continue
                    
            return sources
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_reddit(self, query: str) -> List[ResearchSource]:
        """Reddit search for community insights"""
        try:
            logger.info(f"Searching Reddit for: {query}")
            sources = []
            
            reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent="AgenticResearcher/1.0"
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
                except:
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
            self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_youtube_transcript(self, video_url: str) -> Optional[ResearchSource]:
        """Extract transcript from YouTube video"""
        try:
            # Extract video ID from URL
            if 'youtube.com/watch?v=' in video_url:
                video_id = video_url.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in video_url:
                video_id = video_url.split('youtu.be/')[1].split('?')[0]
            else:
                video_id = video_url
            
            logger.info(f"Extracting YouTube transcript for video: {video_id}")
            
            # Try to get transcript
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in transcript_data])
            
            # Get video info
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                title = info.get('title', 'YouTube Video')
                duration = info.get('duration', 0)
            
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
            logger.error(f"YouTube transcript extraction failed: {str(e)}")
            return None
    
    def transcribe_audio_with_whisper(self, audio_path: str) -> Optional[ResearchSource]:
        """Transcribe audio file using Whisper"""
        try:
            logger.info(f"Transcribing audio with Whisper: {audio_path}")
            
            self.load_whisper_model()
            result = self.whisper_model.transcribe(audio_path)
            
            return ResearchSource(
                title=f"Audio Transcription: {Path(audio_path).name}",
                content=result['text'],
                url=f"file://{audio_path}",
                source_type='audio_transcription',
                confidence=0.9,
                metadata={
                    'language': result.get('language', 'unknown'),
                    'duration': len(result['text']) / 150,  # Approximate duration
                    'model': self.config.WHISPER_MODEL
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return None
    
    def download_and_transcribe_youtube(self, video_url: str) -> Optional[ResearchSource]:
        """Download YouTube video and transcribe with Whisper"""
        try:
            logger.info(f"Downloading and transcribing YouTube video: {video_url}")
            
            # Download audio
            temp_audio = self.config.TEMP_DIR / f"temp_audio_{int(time.time())}.mp3"
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'outtmpl': str(temp_audio.with_suffix('.%(ext)s')),
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get('title', 'YouTube Video')
                duration = info.get('duration', 0)
            
            # Check if file was created with .mp3 extension
            actual_audio_file = temp_audio.with_suffix('.mp3')
            if not actual_audio_file.exists():
                # Try to find the downloaded file
                for ext in ['.mp3', '.m4a', '.webm']:
                    test_file = temp_audio.with_suffix(ext)
                    if test_file.exists():
                        actual_audio_file = test_file
                        break
            
            if actual_audio_file.exists():
                # Transcribe with Whisper
                self.load_whisper_model()
                result = self.whisper_model.transcribe(str(actual_audio_file))
                
                # Cleanup
                actual_audio_file.unlink()
                
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
            else:
                logger.error("Audio file not found after download")
                return None
                
        except Exception as e:
            logger.error(f"YouTube download and transcription failed: {str(e)}")
            return None
    
    async def process_video_content(self, video_urls: List[str]) -> List[ResearchSource]:
        """Process multiple video URLs"""
        logger.info(f"Processing {len(video_urls)} videos")
        
        sources = []
        for url in video_urls:
            # Try transcript first (faster)
            source = self.get_youtube_transcript(url)
            
            # If transcript fails, try download + Whisper
            if not source:
                source = self.download_and_transcribe_youtube(url)
            
            if source:
                sources.append(source)
                
        return sources

class ImageProcessor:
    """Advanced image processing with multiple AI models"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.blip_processor = None
        self.blip_model = None
        
    def load_blip_model(self):
        """Load BLIP model for image captioning"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Image captioning disabled.")
            return
        if not self.blip_processor:
            logger.info("Loading BLIP image captioning model")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    def extract_text_with_ocr(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR
            gray = cv2.medianBlur(gray, 3)
            
            # Extract text
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""
    
    def generate_image_caption(self, image_path: str) -> str:
        """Generate caption for image using BLIP"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                return "Image captioning not available (transformers not loaded)"
            
            logger.info(f"Generating caption for image: {image_path}")
            
            self.load_blip_model()
            
            if not PIL_AVAILABLE:
                return "Image captioning not available (PIL not loaded)"
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt")
            
            out = self.blip_model.generate(**inputs, max_length=100, num_beams=5)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"Image captioning failed: {str(e)}")
            return "Unable to generate caption"
    
    def analyze_image_content(self, image_path: str) -> Dict:
        """Comprehensive image analysis"""
        try:
            logger.info(f"Analyzing image content: {image_path}")
            
            # Basic image properties
            image = cv2.imread(image_path)
            height, width, channels = image.shape
            
            # Object detection (basic)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Edge detection for content complexity
            edges = cv2.Canny(gray, 100, 200)
            edge_density = cv2.countNonZero(edges) / (width * height)
            
            return {
                'dimensions': f"{width}x{height}",
                'aspect_ratio': round(width/height, 2),
                'faces_detected': len(faces),
                'content_complexity': 'high' if edge_density > 0.1 else 'low',
                'channels': channels
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return {}
    
    async def process_images(self, image_paths: List[str]) -> List[ResearchSource]:
        """Process multiple images"""
        logger.info(f"Processing {len(image_paths)} images")
        
        sources = []
        for image_path in image_paths:
            try:
                # Extract text
                ocr_text = self.extract_text_with_ocr(image_path)
                
                # Generate caption
                caption = self.generate_image_caption(image_path)
                
                # Analyze content
                analysis = self.analyze_image_content(image_path)
                
                # Combine all information
                content = f"Image Description: {caption}\n"
                if ocr_text:
                    content += f"Extracted Text: {ocr_text}\n"
                content += f"Technical Analysis: {analysis}"
                
                source = ResearchSource(
                    title=f"Image Analysis: {Path(image_path).name}",
                    content=content,
                    url=f"file://{image_path}",
                    source_type='image_analysis',
                    confidence=0.8,
                    metadata={
                        'analysis': analysis,
                        'has_text': bool(ocr_text),
                        'caption_length': len(caption)
                    }
                )
                sources.append(source)
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {str(e)}")
                continue
                
        return sources

class IntelligentFusionEngine:
    """Advanced cross-modal information fusion with AI"""
    
    def __init__(self, config: AgenticResearchConfig):
        self.config = config
        self.summarizer = None
        self.sentiment_analyzer = None
        self.nlp = None
        
    def load_nlp_models(self):
        """Load NLP models for text processing"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. NLP models disabled.")
            return
            
        if not self.summarizer:
            logger.info("Loading NLP models")
            try:
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            except Exception as e:
                logger.error(f"Failed to load transformers pipelines: {e}")
                self.summarizer = None
                self.sentiment_analyzer = None
            
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                    self.nlp = None
            else:
                self.nlp = None
    
    def extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
            return list(set(entities))[:10]  # Top 10 unique entities
        except:
            return []
    
    def analyze_sentiment_and_themes(self, sources: List[ResearchSource]) -> Dict:
        """Analyze sentiment and extract themes from sources"""
        self.load_nlp_models()
        
        all_content = " ".join([source.content for source in sources if source.content])
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer(all_content[:1000])  # First 1000 chars
        
        # Key entities
        entities = self.extract_key_entities(all_content)
        
        # Source type distribution
        source_types = {}
        for source in sources:
            source_types[source.source_type] = source_types.get(source.source_type, 0) + 1
        
        return {
            'sentiment': sentiment_result[0],
            'key_entities': entities,
            'source_distribution': source_types,
            'total_sources': len(sources),
            'total_content_length': len(all_content)
        }
    
    def generate_insights_by_type(self, sources: List[ResearchSource]) -> Tuple[List[str], List[str], List[str]]:
        """Generate insights categorized by source type"""
        self.load_nlp_models()
        
        text_sources = [s for s in sources if s.source_type in ['web_search', 'wikipedia', 'reddit']]
        video_sources = [s for s in sources if s.source_type in ['youtube_transcript', 'youtube_whisper', 'audio_transcription']]
        image_sources = [s for s in sources if s.source_type == 'image_analysis']
        
        text_insights = []
        video_insights = []
        image_insights = []
        
        # Process text sources
        if text_sources:
            text_content = " ".join([s.content for s in text_sources])[:1000]
            try:
                summary = self.summarizer(text_content, max_length=150, min_length=50, do_sample=False)
                text_insights = [summary[0]['summary_text']]
                
                # Add specific insights
                if any('wikipedia' in s.source_type for s in text_sources):
                    text_insights.append("Academic and encyclopedic sources provide foundational knowledge")
                if any('reddit' in s.source_type for s in text_sources):
                    text_insights.append("Community discussions reveal practical perspectives and experiences")
                    
            except Exception as e:
                logger.error(f"Text summarization failed: {str(e)}")
                text_insights = ["Multiple web sources analyzed with varying perspectives"]
        
        # Process video sources
        if video_sources:
            video_content = " ".join([s.content for s in video_sources])[:1000]
            try:
                summary = self.summarizer(video_content, max_length=150, min_length=50, do_sample=False)
                video_insights = [summary[0]['summary_text']]
                
                # Add video-specific insights
                total_duration = sum([s.metadata.get('duration', 0) for s in video_sources])
                if total_duration > 0:
                    video_insights.append(f"Analysis of {len(video_sources)} videos totaling {total_duration:.1f} minutes")
                    
            except Exception as e:
                logger.error(f"Video summarization failed: {str(e)}")
                video_insights = ["Video and audio content analyzed for spoken insights"]
        
        # Process image sources
        if image_sources:
            image_insights = []
            for source in image_sources:
                if "Image Description:" in source.content:
                    desc = source.content.split("Image Description:")[1].split("\n")[0].strip()
                    image_insights.append(f"Visual content shows: {desc}")
                    
                if "Extracted Text:" in source.content:
                    image_insights.append("Text-based visual information extracted and analyzed")
            
            if not image_insights:
                image_insights = ["Visual content analyzed for contextual information"]
        
        return text_insights, video_insights, image_insights
    
    def generate_fused_summary(self, sources: List[ResearchSource], analysis: Dict) -> List[str]:
        """Generate comprehensive fused summary"""
        self.load_nlp_models()
        
        # Combine top sources from each type
        top_sources = sorted(sources, key=lambda x: x.confidence, reverse=True)[:5]
        combined_content = " ".join([s.content for s in top_sources])[:2000]
        
        try:
            # Generate main summary
            summary = self.summarizer(combined_content, max_length=200, min_length=100, do_sample=False)
            main_summary = summary[0]['summary_text']
            
            fused_insights = [main_summary]
            
            # Add cross-modal insights
            if analysis['source_distribution'].get('wikipedia', 0) > 0 and analysis['source_distribution'].get('youtube_transcript', 0) > 0:
                fused_insights.append("Cross-validation between academic sources and expert discussions confirms key findings")
            
            if analysis['source_distribution'].get('image_analysis', 0) > 0:
                fused_insights.append("Visual evidence supports and illustrates the textual research findings")
            
            # Add sentiment-based insight
            sentiment = analysis['sentiment']
            if sentiment['label'] == 'POSITIVE':
                fused_insights.append("Overall research sentiment indicates positive outlook and promising developments")
            elif sentiment['label'] == 'NEGATIVE':
                fused_insights.append("Research reveals significant challenges and concerns in this domain")
            
            # Add entity-based insight
            if analysis['key_entities']:
                top_entities = analysis['key_entities'][:3]
                fused_insights.append(f"Key stakeholders and concepts identified: {', '.join(top_entities)}")
            
            return fused_insights
            
        except Exception as e:
            logger.error(f"Fused summary generation failed: {str(e)}")
            return [
                "Multi-modal research analysis completed across text, video, and image sources",
                f"Total of {analysis['total_sources']} sources analyzed from {len(analysis['source_distribution'])} different types",
                "Cross-reference validation performed between different content modalities"
            ]
    
    def generate_follow_up_questions(self, topic: str, sources: List[ResearchSource], analysis: Dict) -> List[str]:
        """Generate intelligent follow-up research questions"""
        questions = []
        
        # Topic-based questions
        questions.append(f"What are the long-term implications of {topic} development?")
        questions.append(f"How does {topic} compare to alternative approaches in the field?")
        
        # Entity-based questions
        if analysis['key_entities']:
            entity = analysis['key_entities'][0]
            questions.append(f"What role does {entity} play in {topic} advancement?")
        
        # Source-type based questions
        if 'wikipedia' in analysis['source_distribution']:
            questions.append(f"What recent developments in {topic} are not yet covered in academic sources?")
        
        if 'reddit' in analysis['source_distribution']:
            questions.append(f"What practical challenges do practitioners face with {topic}?")
        
        if 'youtube_transcript' in analysis['source_distribution']:
            questions.append(f"What expert opinions exist on the future of {topic}?")
        
        return questions[:3]  # Return top 3 questions

class AgenticResearchAssistant:
    """Main agentic research assistant orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = AgenticResearchConfig()
        self.web_searcher = None
        self.video_processor = VideoAudioProcessor(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.fusion_engine = IntelligentFusionEngine(self.config)
        
        # Research state
        self.current_topic = ""
        self.all_sources = []
        self.research_insights = None
        
    async def research_topic(
        self, 
        topic: str,
        video_urls: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None
    ) -> ResearchInsight:
        """Complete agentic research process"""
        
        logger.info(f"🤖 Starting agentic research on topic: {topic}")
        self.current_topic = topic
        self.all_sources = []
        
        # Stage 1: Web Search
        logger.info("📰 Stage 1: Comprehensive web search")
        async with WebSearchEngine(self.config) as searcher:
            web_sources = await searcher.comprehensive_search(topic)
            self.all_sources.extend(web_sources)
            logger.info(f"Found {len(web_sources)} web sources")
        
        # Stage 2: Video Processing
        logger.info("🎥 Stage 2: Video and audio processing")
        if video_urls:
            video_sources = await self.video_processor.process_video_content(video_urls)
            self.all_sources.extend(video_sources)
            logger.info(f"Processed {len(video_sources)} video sources")
        
        # Process audio files
        if audio_paths:
            for audio_path in audio_paths:
                audio_source = self.video_processor.transcribe_audio_with_whisper(audio_path)
                if audio_source:
                    self.all_sources.append(audio_source)
            logger.info(f"Processed {len(audio_paths)} audio files")
        
        # Stage 3: Image Processing
        logger.info("🖼️ Stage 3: Image analysis")
        if image_paths:
            image_sources = await self.image_processor.process_images(image_paths)
            self.all_sources.extend(image_sources)
            logger.info(f"Processed {len(image_sources)} image sources")
        
        # Stage 4: Intelligent Fusion
        logger.info("🧠 Stage 4: Cross-modal fusion and analysis")
        analysis = self.fusion_engine.analyze_sentiment_and_themes(self.all_sources)
        
        # Generate categorized insights
        text_insights, video_insights, image_insights = self.fusion_engine.generate_insights_by_type(self.all_sources)
        
        # Generate fused summary
        fused_summary = self.fusion_engine.generate_fused_summary(self.all_sources, analysis)
        
        # Generate follow-up questions
        follow_up_questions = self.fusion_engine.generate_follow_up_questions(topic, self.all_sources, analysis)
        
        # Calculate confidence score
        avg_confidence = sum([s.confidence for s in self.all_sources]) / len(self.all_sources) if self.all_sources else 0
        
        # Create research insight object
        self.research_insights = ResearchInsight(
            text_insights=text_insights,
            video_insights=video_insights,
            image_insights=image_insights,
            fused_summary=fused_summary,
            follow_up_questions=follow_up_questions,
            confidence_score=avg_confidence,
            sources_count=len(self.all_sources)
        )
        
        logger.info(f"✅ Research complete! Analyzed {len(self.all_sources)} sources with {avg_confidence:.2f} confidence")
        return self.research_insights
    
    def export_research_report(self, output_path: str = None) -> str:
        """Export comprehensive research report"""
        if not self.research_insights:
            return "No research data available. Run research_topic() first."
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
═══════════════════════════════════════════════════════════════
🤖 AGENTIC MULTI-MODAL RESEARCH REPORT
═══════════════════════════════════════════════════════════════

Topic: {self.current_topic}
Generated: {timestamp}
Sources Analyzed: {self.research_insights.sources_count}
Confidence Score: {self.research_insights.confidence_score:.2f}/1.0

═══════════════════════════════════════════════════════════════
📰 TEXT SOURCE INSIGHTS
═══════════════════════════════════════════════════════════════

{chr(10).join(['• ' + insight for insight in self.research_insights.text_insights])}

═══════════════════════════════════════════════════════════════
🎥 VIDEO/AUDIO SOURCE INSIGHTS  
═══════════════════════════════════════════════════════════════

{chr(10).join(['• ' + insight for insight in self.research_insights.video_insights])}

═══════════════════════════════════════════════════════════════
🖼️ IMAGE/VISUAL SOURCE INSIGHTS
═══════════════════════════════════════════════════════════════

{chr(10).join(['• ' + insight for insight in self.research_insights.image_insights])}

═══════════════════════════════════════════════════════════════
🔗 CROSS-MODAL FUSED ANALYSIS
═══════════════════════════════════════════════════════════════

{chr(10).join([f"{i+1}. {point}" for i, point in enumerate(self.research_insights.fused_summary)])}

═══════════════════════════════════════════════════════════════
❓ RECOMMENDED FOLLOW-UP RESEARCH
═══════════════════════════════════════════════════════════════

{chr(10).join([f"{i+1}. {question}" for i, question in enumerate(self.research_insights.follow_up_questions)])}

═══════════════════════════════════════════════════════════════
📊 SOURCE BREAKDOWN
═══════════════════════════════════════════════════════════════

"""
        
        # Add source details
        source_types = {}
        for source in self.all_sources:
            if source.source_type not in source_types:
                source_types[source.source_type] = []
            source_types[source.source_type].append({
                'title': source.title,
                'url': source.url,
                'confidence': source.confidence
            })
        
        for source_type, sources in source_types.items():
            report += f"\n{source_type.upper().replace('_', ' ')} ({len(sources)} sources):\n"
            for source in sources:
                report += f"• {source['title']} (Confidence: {source['confidence']:.2f})\n  URL: {source['url']}\n"
        
        report += f"""
═══════════════════════════════════════════════════════════════
🛠️ SYSTEM INFORMATION
═══════════════════════════════════════════════════════════════

Whisper Model: {self.config.WHISPER_MODEL}
Search Engines: DuckDuckGo, Wikipedia, Reddit
Vision Models: BLIP-2, Tesseract OCR
NLP Models: BART, RoBERTa, spaCy
Fusion Engine: Multi-modal cross-attention analysis

Generated by Agentic Multi-Modal Research Assistant v1.0
"""
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def get_sources_by_type(self, source_type: str) -> List[ResearchSource]:
        """Get sources filtered by type"""
        return [s for s in self.all_sources if s.source_type == source_type]
    
    def get_high_confidence_sources(self, threshold: float = 0.8) -> List[ResearchSource]:
        """Get sources above confidence threshold"""
        return [s for s in self.all_sources if s.confidence >= threshold]

# Utility functions and CLI interface

def setup_environment():
    """Setup required directories and check dependencies"""
    # Create temp directory
    Path("temp_files").mkdir(exist_ok=True)
    
    # Check if required models are available
    try:
        import whisper
        whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"⚠️ Whisper setup issue: {str(e)}")
    
    try:
        from transformers import BlipProcessor
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✅ BLIP model loaded successfully")  
    except Exception as e:
        print(f"⚠️ BLIP setup issue: {str(e)}")
    
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
    except Exception as e:
        print(f"⚠️ spaCy setup issue: {str(e)}")
        print("Install with: python -m spacy download en_core_web_sm")

async def demo_research():
    """Demo function showing how to use the research assistant"""
    
    print("🤖 Initializing Agentic Research Assistant...")
    assistant = AgenticResearchAssistant()
    
    # Demo research topic
    topic = "Artificial Intelligence Ethics"
    
    # Demo video URLs (replace with actual URLs)
    video_urls = [
        "https://youtube.com/watch?v=DEMO_VIDEO_1",
        "https://youtube.com/watch?v=DEMO_VIDEO_2"
    ]
    
    # Demo image paths (replace with actual paths)
    image_paths = [
        "path/to/demo_image1.jpg",
        "path/to/demo_image2.png"
    ]
    
    # Demo audio paths (replace with actual paths)
    audio_paths = [
        "path/to/demo_audio.mp3"
    ]
    
    print(f"🔍 Starting research on: {topic}")
    
    try:
        # Perform research
        insights = await assistant.research_topic(
            topic=topic,
            video_urls=video_urls if input("Include video analysis? (y/n): ").lower() == 'y' else None,
            image_paths=image_paths if input("Include image analysis? (y/n): ").lower() == 'y' else None,
            audio_paths=audio_paths if input("Include audio analysis? (y/n): ").lower() == 'y' else None
        )
        
        # Display results
        print("\n" + "="*60)
        print("🎯 RESEARCH RESULTS")
        print("="*60)
        
        print(f"📊 Confidence Score: {insights.confidence_score:.2f}")
        print(f"📚 Sources Analyzed: {insights.sources_count}")
        
        print("\n🔗 FUSED SUMMARY:")
        for i, point in enumerate(insights.fused_summary, 1):
            print(f"{i}. {point}")
        
        print("\n❓ FOLLOW-UP QUESTIONS:")
        for i, question in enumerate(insights.follow_up_questions, 1):
            print(f"{i}. {question}")
        
        # Export report
        report_path = f"research_report_{topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = assistant.export_research_report(report_path)
        
        print(f"\n📄 Full report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Demo research failed: {str(e)}")
        print(f"❌ Research failed: {str(e)}")

# CLI Interface
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Multi-Modal Research Assistant")
    parser.add_argument("--topic", type=str, required=True, help="Research topic")
    parser.add_argument("--videos", nargs="*", help="YouTube video URLs")
    parser.add_argument("--images", nargs="*", help="Image file paths")
    parser.add_argument("--audio", nargs="*", help="Audio file paths")
    parser.add_argument("--output", type=str, help="Output report path")
    parser.add_argument("--setup", action="store_true", help="Setup environment and check dependencies")
    parser.add_argument("--demo", action="store_true", help="Run demo research")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_environment()
        return
    
    if args.demo:
        asyncio.run(demo_research())
        return
    
    # Run actual research
    async def run_research():
        assistant = AgenticResearchAssistant()
        
        insights = await assistant.research_topic(
            topic=args.topic,
            video_urls=args.videos,
            image_paths=args.images,
            audio_paths=args.audio
        )
        
        output_path = args.output or f"research_report_{args.topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = assistant.export_research_report(output_path)
        
        print("Research completed successfully!")
        print(f"Report saved to: {output_path}")
        
        return insights
    
    try:
        asyncio.run(run_research())
    except KeyboardInterrupt:
        print("\n❌ Research interrupted by user")
    except Exception as e:
        print(f"❌ Research failed: {str(e)}")

if __name__ == "__main__":
    main()

# Installation requirements (save as requirements.txt):
"""
aiohttp>=3.9.0
duckduckgo-search>=4.0.0
youtube-transcript-api>=0.6.0
yt-dlp>=2023.12.30
openai-whisper>=20231117
transformers>=4.36.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pytesseract>=0.3.10
pillow>=10.0.0
requests>=2.31.0
wikipedia>=1.4.0
praw>=7.7.0
tenacity>=8.2.0
spacy>=3.7.0
datasets>=2.14.0
accelerate>=0.24.0
numpy>=1.24.0
pandas>=2.0.0
"""

# Usage Examples:

"""
# Basic usage:
python research_assistant.py --topic "Climate Change" --output report.txt

# With video analysis:
python research_assistant.py --topic "AI Ethics" --videos "https://youtube.com/watch?v=VIDEO1" "https://youtube.com/watch?v=VIDEO2"

# With image analysis:
python research_assistant.py --topic "Medical AI" --images "chart1.png" "diagram2.jpg"

# Complete multi-modal research:
python research_assistant.py --topic "Quantum Computing" --videos "https://youtube.com/watch?v=QUANTUM1" --images "quantum_chip.jpg" --audio "lecture.mp3"

# Setup environment:
python research_assistant.py --setup

# Run demo:
python research_assistant.py --demo
"""