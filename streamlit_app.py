"""
Enhanced Multi-Modal Research Assistant - Streamlit Web Application
Fixed version with better YouTube URL handling and comprehensive error management
"""

import streamlit as st
import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import sys
import tempfile

# Add the current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing nest_asyncio for better async support
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    logger.warning("nest_asyncio not available, async operations may be limited")

# Check if modules are available before importing
try:
    from config import get_config, setup_environment
    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    logger.error(f"Config module not available: {e}")

try:
    from research import AgenticResearchAssistant, ResearchInsight, ResearchSource
    RESEARCH_AVAILABLE = True
except ImportError as e:
    RESEARCH_AVAILABLE = False
    logger.error(f"Research module not available: {e}")

# Initialize configuration
if CONFIG_AVAILABLE:
    try:
        config = get_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        config = None
else:
    config = None

# Page configuration
if config:
    st.set_page_config(
        page_title=config.streamlit_config.page_title,
        page_icon=config.streamlit_config.page_icon,
        layout=config.streamlit_config.layout,
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title="Multi-Modal Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2E86AB;
    }
    .api-status-active {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .api-status-inactive {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .progress-step {
        background-color: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .progress-step.active {
        background-color: #fff3cd;
        border-left-color: #17a2b8;
    }
    .progress-step.completed {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables with better defaults"""
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'research_in_progress' not in st.session_state:
        st.session_state.research_in_progress = False
    if 'research_progress' not in st.session_state:
        st.session_state.research_progress = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {'images': [], 'audio': [], 'video': []}
    if 'config_status' not in st.session_state:
        if CONFIG_AVAILABLE:
            try:
                st.session_state.config_status = setup_environment()
            except Exception as e:
                st.session_state.config_status = {'error': str(e)}
        else:
            st.session_state.config_status = {'error': 'Configuration module not available'}
    if 'research_assistant' not in st.session_state:
        st.session_state.research_assistant = None
    if 'api_status' not in st.session_state:
        if RESEARCH_AVAILABLE:
            try:
                from research import AgenticResearchConfig
                temp_config = AgenticResearchConfig()
                st.session_state.api_status = temp_config.get_api_status()
            except:
                st.session_state.api_status = {}
        else:
            st.session_state.api_status = {}

def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format"""
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, url.strip()):
            return True
    return False

import re

def display_header():
    """Display the enhanced main header"""
    st.markdown('<h1 class="main-header">🤖 Multi-Modal Research Assistant</h1>', unsafe_allow_html=True)
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if CONFIG_AVAILABLE and RESEARCH_AVAILABLE:
            st.success("✅ System Ready")
        else:
            st.error("❌ System Issues")
    
    with col2:
        api_count = sum(1 for status in st.session_state.api_status.values() if status)
        total_apis = len(st.session_state.api_status)
        if api_count > 0:
            st.info(f"🔑 {api_count}/{total_apis} APIs Active")
        else:
            st.warning("⚠️ No APIs Configured")
    
    with col3:
        if st.session_state.research_results:
            sources = st.session_state.research_results.sources_count
            st.success(f"📚 {sources} Sources Analyzed")
        else:
            st.info("📚 Ready for Research")
    
    with col4:
        if st.session_state.research_in_progress:
            st.warning("⏳ Research in Progress")
        else:
            st.info("🚀 Ready to Start")

def display_enhanced_sidebar():
    """Enhanced sidebar with detailed API status and controls"""
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # System Status
        st.subheader("📊 System Status")
        
        system_status = "✅ Ready" if (CONFIG_AVAILABLE and RESEARCH_AVAILABLE) else "❌ Issues"
        st.markdown(f"**System:** {system_status}")
        
        if not CONFIG_AVAILABLE:
            st.error("Config module missing")
        if not RESEARCH_AVAILABLE:
            st.error("Research module missing")
        
        st.divider()
        
        # Enhanced API Status
        st.subheader("🔑 API Integration Status")
        
        api_configs = {
            'google_cse': {'name': 'Google Custom Search', 'icon': '🔍'},
            'serpapi': {'name': 'SerpAPI', 'icon': '🌐'},
            'reddit': {'name': 'Reddit API', 'icon': '🔗'},
            'openai': {'name': 'OpenAI', 'icon': '🤖'},
            'huggingface': {'name': 'Hugging Face', 'icon': '🤗'},
            'assemblyai': {'name': 'AssemblyAI', 'icon': '🎤'}
        }
        
        active_count = 0
        for api_key, api_info in api_configs.items():
            is_active = st.session_state.api_status.get(api_key, False)
            status_class = "api-status-active" if is_active else "api-status-inactive"
            status_icon = "✅" if is_active else "❌"
            
            st.markdown(f"""
            <div class="{status_class}">
                {status_icon} {api_info['icon']} <strong>{api_info['name']}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if is_active:
                active_count += 1
        
        # API Summary
        st.markdown(f"**Active APIs:** {active_count}/{len(api_configs)}")
        
        if active_count == 0:
            st.warning("⚠️ No APIs configured. Using free sources only.")
            st.info("💡 Add API keys to .env or config.json for enhanced features")

def display_research_input():
    """Enhanced research input form with URL validation"""
    st.markdown('<div class="section-header">🔍 Research Configuration</div>', unsafe_allow_html=True)
    
    # Show API capabilities
    active_apis = [name for name, status in st.session_state.api_status.items() if status]
    if active_apis:
        st.success(f"🚀 Enhanced search with: {', '.join(active_apis)}")
    else:
        st.info("🔍 Using free sources: DuckDuckGo, Wikipedia")
    
    with st.form("enhanced_research_form"):
        # Main topic input
        topic = st.text_input(
            "🎯 Research Topic *",
            placeholder="e.g., Artificial Intelligence Ethics, Climate Change Solutions, Quantum Computing",
            help="Enter a specific topic for focused research results"
        )
        
        # Advanced options in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📹 Video Sources**")
            video_urls_text = st.text_area(
                "YouTube Video URLs",
                placeholder="https://youtube.com/watch?v=dQw4w9WgXcQ\nhttps://youtu.be/dQw4w9WgXcQ",
                height=100,
                help="One URL per line - supports various YouTube formats"
            )
            
            # URL validation preview
            if video_urls_text.strip():
                urls = [url.strip() for url in video_urls_text.split('\n') if url.strip()]
                valid_urls = []
                invalid_urls = []
                
                for url in urls:
                    if validate_youtube_url(url):
                        valid_urls.append(url)
                    else:
                        invalid_urls.append(url)
                
                if valid_urls:
                    st.success(f"✅ {len(valid_urls)} valid YouTube URLs")
                
                if invalid_urls:
                    st.error(f"❌ {len(invalid_urls)} invalid URLs:")
                    for invalid_url in invalid_urls:
                        st.write(f"  • {invalid_url}")
            
            st.markdown("**📄 Additional Text**")
            text_sources = st.text_area(
                "Paste Additional Content",
                placeholder="Research papers, articles, or any relevant text...",
                height=120,
                help="Add any text content to include in the analysis"
            )
        
        with col2:
            st.markdown("**📁 File Uploads**")
            
            # Enhanced file upload with size limits
            uploaded_images = st.file_uploader(
                "📸 Images (JPG, PNG, GIF, etc.)",
                type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"],
                accept_multiple_files=True,
                help="Upload images for OCR text extraction and visual analysis (Max: 200MB each)"
            )
            
            uploaded_audio = st.file_uploader(
                "🎵 Audio Files (MP3, WAV, etc.)",
                type=["mp3", "wav", "m4a", "flac", "aac", "ogg"],
                accept_multiple_files=True,
                help="Upload audio files for speech-to-text transcription (Max: 200MB each)"
            )
            
            uploaded_video = st.file_uploader(
                "🎬 Video Files (MP4, AVI, etc.)",
                type=["mp4", "avi", "mov", "mkv", "webm", "flv"],
                accept_multiple_files=True,
                help="Upload video files for audio extraction and transcription (Max: 200MB each)"
            )
            
            # File size validation
            def check_file_sizes(files, file_type):
                if files:
                    total_size = sum(file.size for file in files) / (1024 * 1024)  # MB
                    if total_size > 500:  # 500MB total limit
                        st.warning(f"⚠️ {file_type} files total: {total_size:.1f}MB (consider reducing)")
                    else:
                        st.info(f"📁 {file_type} files: {total_size:.1f}MB")
            
            check_file_sizes(uploaded_images, "Image")
            check_file_sizes(uploaded_audio, "Audio")
            check_file_sizes(uploaded_video, "Video")
        
        # Submit button with validation
        submitted = st.form_submit_button(
            "🚀 Start Comprehensive Research",
            use_container_width=True,
            type="primary"
        )
        
        # Validation and preparation
        if submitted:
            if not topic.strip():
                st.error("❌ Please enter a research topic")
                return None
            
            # Validate YouTube URLs
            video_urls = []
            if video_urls_text.strip():
                urls = [url.strip() for url in video_urls_text.split('\n') if url.strip()]
                for url in urls:
                    if validate_youtube_url(url):
                        video_urls.append(url)
                    else:
                        st.error(f"❌ Invalid YouTube URL: {url}")
                        return None
            
            # Count total inputs
            input_count = 1  # topic
            if video_urls:
                input_count += len(video_urls)
            if text_sources.strip():
                input_count += 1
            if uploaded_images:
                input_count += len(uploaded_images)
            if uploaded_audio:
                input_count += len(uploaded_audio)
            if uploaded_video:
                input_count += len(uploaded_video)
            
            # Show input summary
            st.success(f"✅ Research configured with {input_count} input sources")
            
            return {
                'topic': topic.strip(),
                'video_urls': video_urls,
                'text_sources': text_sources.strip() if text_sources.strip() else None,
                'uploaded_images': uploaded_images or [],
                'uploaded_audio': uploaded_audio or [],
                'uploaded_videos': uploaded_video or []
            }
    
    return None

def save_uploaded_files(uploaded_files: Dict, upload_dir: Path) -> Dict[str, List[str]]:
    """Enhanced file saving with progress tracking and better error handling"""
    saved_files = {'images': [], 'audio': [], 'videos': []}
    
    upload_dir.mkdir(exist_ok=True)
    
    file_mappings = [
        ('images', 'uploaded_images'),
        ('audio', 'uploaded_audio'),
        ('videos', 'uploaded_videos')
    ]
    
    for save_key, upload_key in file_mappings:
        file_list = uploaded_files.get(upload_key, [])
        if file_list:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(file_list):
                try:
                    # Update progress
                    progress = (i + 1) / len(file_list)
                    progress_bar.progress(progress)
                    status_text.text(f"Saving {save_key[:-1]}: {uploaded_file.name}")
                    
                    # Check file size
                    file_size = uploaded_file.size / (1024 * 1024)  # MB
                    if file_size > 200:  # 200MB limit per file
                        st.error(f"❌ File too large: {uploaded_file.name} ({file_size:.1f}MB)")
                        continue
                    
                    # Create unique filename
                    timestamp = int(time.time())
                    safe_filename = re.sub(r'[^\w\-_\.]', '_', uploaded_file.name)
                    file_path = upload_dir / f"{timestamp}_{i}_{safe_filename}"
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    saved_files[save_key].append(str(file_path))
                    logger.info(f"Saved {save_key[:-1]} file: {file_path} ({file_size:.1f}MB)")
                    
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")
                    logger.error(f"File save error: {e}")
            
            progress_bar.empty()
            status_text.empty()
    
    return saved_files

async def run_enhanced_research(research_input: Dict[str, Any]) -> tuple:
    """Enhanced research execution with better progress tracking and error handling"""
    if not RESEARCH_AVAILABLE:
        st.error("❌ Research module not available. Please check installation.")
        return None, None
    
    try:
        # Initialize assistant
        assistant = AgenticResearchAssistant()
        
        # Create enhanced progress tracking
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Research Progress")
            
            progress_steps = [
                ("🔍", "Web Search", "Searching multiple engines for comprehensive coverage..."),
                ("📄", "Text Processing", "Analyzing additional text content..."),
                ("🎥", "Video Analysis", "Extracting and analyzing video transcripts..."),
                ("🎵", "Audio Processing", "Transcribing audio content..."),
                ("🖼️", "Image Analysis", "Processing images with OCR and AI analysis..."),
                ("🧠", "AI Synthesis", "Generating cross-modal insights and summaries...")
            ]
            
            step_containers = {}
            for i, (icon, title, desc) in enumerate(progress_steps):
                step_containers[i] = st.empty()
                step_containers[i].markdown(f"""
                <div class="progress-step">
                    {icon} <strong>{title}</strong>: {desc}
                </div>
                """, unsafe_allow_html=True)
        
        # Save uploaded files with error handling
        upload_dir = Path("uploads")
        try:
            saved_files = save_uploaded_files(research_input, upload_dir)
        except Exception as e:
            st.error(f"❌ Error saving files: {e}")
            return None, None
        
        # Update step 1 - Web Search
        step_containers[0].markdown("""
        <div class="progress-step active">
            🔍 <strong>Web Search</strong>: Searching multiple engines...
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare inputs with validation
        video_urls = research_input.get('video_urls', [])
        image_paths = saved_files.get('images', [])
        audio_paths = saved_files.get('audio', [])
        video_paths = saved_files.get('videos', [])
        text_sources = research_input.get('text_sources')
        
        # Validate video URLs before processing
        valid_video_urls = []
        if video_urls:
            for url in video_urls:
                if validate_youtube_url(url):
                    valid_video_urls.append(url)
                else:
                    st.warning(f"⚠️ Skipping invalid YouTube URL: {url}")
        
        # Run research with enhanced error handling
        try:
            insights = await assistant.research_topic(
                topic=research_input['topic'],
                video_urls=valid_video_urls if valid_video_urls else None,
                image_paths=image_paths if image_paths else None,
                audio_paths=audio_paths + video_paths if audio_paths or video_paths else None,
                text_sources=text_sources
            )
            
            # Mark all steps as completed
            for i in range(len(progress_steps)):
                icon, title, _ = progress_steps[i]
                step_containers[i].markdown(f"""
                <div class="progress-step completed">
                    ✅ <strong>{title}</strong>: Completed successfully
                </div>
                """, unsafe_allow_html=True)
            
            return insights, assistant
            
        except Exception as research_error:
            logger.error(f"Research execution failed: {research_error}")
            
            # Show detailed error information
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Research Error:</strong> {str(research_error)}
                <br><small>Check the logs for more details</small>
            </div>
            """, unsafe_allow_html=True)
            
            return None, None
        
    except Exception as e:
        logger.error(f"Research setup failed: {e}")
        st.error(f"❌ Research setup failed: {str(e)}")
        return None, None

def display_enhanced_results(insights: ResearchInsight):
    """Enhanced results display with better organization and error handling"""
    if not insights:
        return
    
    st.markdown('<div class="section-header">📊 Research Results</div>', unsafe_allow_html=True)
    
    # Enhanced metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📚 Sources Analyzed",
            insights.sources_count,
            help="Total number of sources successfully processed"
        )
    
    with col2:
        confidence_color = "🟢" if insights.confidence_score > 0.8 else "🟡" if insights.confidence_score > 0.6 else "🔴"
        st.metric(
            f"{confidence_color} Confidence Score",
            f"{insights.confidence_score:.1%}",
            help="Average confidence across all sources"
        )
    
    with col3:
        total_insights = len(insights.text_insights) + len(insights.video_insights) + len(insights.image_insights)
        st.metric(
            "💡 Total Insights",
            total_insights,
            help="Number of key insights extracted"
        )
    
    with col4:
        st.metric(
            "❓ Follow-up Questions",
            len(insights.follow_up_questions),
            help="Suggested questions for deeper research"
        )
    
    # Results tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Fused Analysis", 
        "📰 Text Sources", 
        "🎥 Media Sources", 
        "🖼️ Visual Sources", 
        "❓ Next Steps"
    ])
    
    with tab1:
        st.markdown("### 🎯 Cross-Modal Research Summary")
        st.info("This summary combines insights from all analyzed sources")
        
        if insights.fused_summary:
            for i, point in enumerate(insights.fused_summary, 1):
                st.markdown(f"""
                <div class="success-box">
                    <strong>{i}.</strong> {point}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No fused summary available")
    
    with tab2:
        st.markdown("### 📖 Text Source Analysis")
        if insights.text_insights:
            for i, insight in enumerate(insights.text_insights, 1):
                with st.expander(f"📄 Text Insight {i}", expanded=i<=3):
                    st.write(insight)
        else:
            st.info("No text sources were analyzed")
    
    with tab3:
        st.markdown("### 🎬 Video & Audio Analysis")
        if insights.video_insights:
            for i, insight in enumerate(insights.video_insights, 1):
                with st.expander(f"🎥 Media Insight {i}", expanded=i<=2):
                    st.write(insight)
        else:
            st.info("No video or audio sources were analyzed")
    
    with tab4:
        st.markdown("### 🎨 Visual Content Analysis")
        if insights.image_insights:
            for i, insight in enumerate(insights.image_insights, 1):
                with st.expander(f"🖼️ Image Analysis {i}", expanded=i<=2):
                    st.write(insight)
        else:
            st.info("No images were analyzed")
    
    with tab5:
        st.markdown("### 🚀 Suggested Follow-up Research")
        st.info("These questions can guide your next research session")
        
        if insights.follow_up_questions:
            for i, question in enumerate(insights.follow_up_questions, 1):
                st.markdown(f"""
                <div class="info-box">
                    <strong>Q{i}:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No follow-up questions generated")

def display_troubleshooting_section():
    """Display troubleshooting information for common issues"""
    with st.expander("🔧 Troubleshooting & Tips", expanded=False):
        st.markdown("""
        ### 🚨 Common Issues & Solutions
        
        **YouTube URL Issues:**
        - ✅ Valid: `https://youtube.com/watch?v=VIDEO_ID`
        - ✅ Valid: `https://youtu.be/VIDEO_ID`
        - ❌ Invalid: URLs without video IDs or from other platforms
        - 💡 Make sure video has available transcripts (auto-generated or manual)
        
        **API Configuration:**
        - Check your `.env` file has the correct API keys
        - Ensure `config.json` is properly formatted
        - Some features work without APIs (DuckDuckGo, Wikipedia)
        
        **File Upload Issues:**
        - Maximum file size: 200MB per file
        - Supported image formats: JPG, PNG, GIF, BMP, WebP, TIFF
        - Supported audio formats: MP3, WAV, M4A, FLAC, AAC, OGG
        - Supported video formats: MP4, AVI, MOV, MKV, WebM, FLV
        
        **Performance Tips:**
        - Larger files take longer to process
        - Multiple sources provide richer insights
        - Use specific topics for better results
        """)

def main():
    """Enhanced main application function with comprehensive error handling"""
    # Initialize
    initialize_session_state()
    display_header()
    display_enhanced_sidebar()
    
    # Check if core modules are available
    if not CONFIG_AVAILABLE or not RESEARCH_AVAILABLE:
        st.markdown("""
        <div class="error-box">
            <strong>❌ Core modules not available</strong><br>
            Please install required dependencies:<br>
            <code>pip install -r requirements.txt</code><br>
            Then restart the application.
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Run the setup script: `python run_app.py` for automatic setup")
        st.stop()
    
    # Troubleshooting section
    display_troubleshooting_section()
    
    # Main research interface
    research_input = display_research_input()
    
    # Handle research submission
    if research_input and not st.session_state.research_in_progress:
        st.session_state.research_in_progress = True
        
        # Show research initiation
        st.markdown("""
        <div class="info-box">
            <strong>🧠 Starting Multi-Modal Research...</strong><br>
            This may take a few minutes depending on the number of sources.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Run async research with proper error handling
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            insights, assistant = loop.run_until_complete(run_enhanced_research(research_input))
            loop.close()
            
            if insights and insights.sources_count > 0:
                st.session_state.research_results = insights
                st.session_state.research_assistant = assistant
                
                # Success animation and message
                st.balloons()
                st.markdown("""
                <div class="success-box">
                    <strong>🎉 Research completed successfully!</strong><br>
                    Analysis complete with {sources} sources processed.
                </div>
                """.format(sources=insights.sources_count), unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="error-box">
                    <strong>❌ Research returned no results</strong><br>
                    Please try:
                    <ul>
                        <li>Different keywords or more specific topics</li>
                        <li>Valid YouTube URLs with available transcripts</li>
                        <li>Adding text content or uploading files</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Research failed:</strong> {str(e)}<br>
                <small>Please check your inputs and try again. See troubleshooting section below.</small>
            </div>
            """, unsafe_allow_html=True)
            logger.error(f"Research error: {e}", exc_info=True)
        finally:
            st.session_state.research_in_progress = False
            st.rerun()
    
    # Display results if available
    if st.session_state.research_results:
        display_enhanced_results(st.session_state.research_results)
        
        # Export functionality
        st.markdown("### 💾 Export Research Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export JSON", use_container_width=True):
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'topic': st.session_state.research_assistant.current_topic if st.session_state.research_assistant else 'Unknown',
                    'insights': {
                        'fused_summary': st.session_state.research_results.fused_summary,
                        'text_insights': st.session_state.research_results.text_insights,
                        'video_insights': st.session_state.research_results.video_insights,
                        'image_insights': st.session_state.research_results.image_insights,
                        'follow_up_questions': st.session_state.research_results.follow_up_questions
                    },
                    'metadata': {
                        'sources_count': st.session_state.research_results.sources_count,
                        'confidence_score': st.session_state.research_results.confidence_score
                    }
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_str,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📝 Export Text", use_container_width=True):
                results = st.session_state.research_results
                topic = st.session_state.research_assistant.current_topic if st.session_state.research_assistant else 'Unknown'
                
                report = f"""MULTI-MODAL RESEARCH REPORT
==========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Topic: {topic}
Sources: {results.sources_count}
Confidence: {results.confidence_score:.1%}

EXECUTIVE SUMMARY
================
{chr(10).join(f'• {point}' for point in results.fused_summary)}

TEXT SOURCE INSIGHTS
===================
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(results.text_insights))}

MEDIA SOURCE INSIGHTS
====================
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(results.video_insights))}

VISUAL CONTENT INSIGHTS
======================
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(results.image_insights))}

FOLLOW-UP RESEARCH QUESTIONS
===========================
{chr(10).join(f'{i+1}. {question}' for i, question in enumerate(results.follow_up_questions))}

Generated by Multi-Modal Research Assistant
"""
                
                st.download_button(
                    label="📝 Download Text Report",
                    data=report,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("📊 View Statistics", use_container_width=True):
                if st.session_state.research_assistant:
                    stats = st.session_state.research_assistant.get_research_statistics()
                    st.json(stats)
        
        # Action buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🆕 New Research", use_container_width=True, type="primary"):
                st.session_state.research_results = None
                st.session_state.research_assistant = None
                st.rerun()
        
        with col2:
            if st.button("🔍 Research Tips", use_container_width=True):
                st.info("💡 Use the follow-up questions above to refine your research, or try different combinations of text, video, and image sources for richer insights.")
    
    # Footer with enhanced information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem;">
        <h4>🤖 Multi-Modal Research Assistant</h4>
        <p style="margin: 0.5rem 0;">
            <strong>Version 2.1</strong> • Fixed YouTube Integration • Enhanced Error Handling
        </p>
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            🔍 Web Search • 🎥 Video Analysis • 🖼️ Image Processing • 🧠 AI Fusion
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()