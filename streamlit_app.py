"""
Enhanced Multi-Modal Research Assistant - Streamlit Web Application
Improved version with proper API integration and better error handling
"""
# ...existing imports...
import sys
from dotenv import load_dotenv
load_dotenv()
# ...existing code...
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

# Add the current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        background-color: #fffbe6;
        color: #222
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .progress-step.active {
        background-color: #fffbe6;
        color: #222
        border-left-color: #17a2b8;
    }
    .progress-step.completed {
        background-color: #fffbe6;
        color: #222
        border-left-color: #28a745;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #dee2e6;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .source-counter {
        display: inline-block;
        background: #007bff;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        margin: 0.25rem;
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
    
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem;">
        <p style="margin: 0; color: #666; font-size: 1.1rem;">
            🔍 <strong>Conduct comprehensive research</strong> by analyzing text, video, audio, and images from multiple sources
        </p>
    </div>
    """, unsafe_allow_html=True)

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
        
        st.divider()
        
        # Research Settings
        st.subheader("🛠️ Research Settings")
        
        # Model selection
        whisper_model = st.selectbox(
            "Audio Model",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        # Search limits
        max_results = st.slider(
            "Max Results per Source",
            min_value=3,
            max_value=20,
            value=8,
            help="More results = better coverage but slower processing"
        )
        
        # File size limits
        st.info(f"📁 Max file size: 200MB")
        
        st.divider()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.research_results = None
                st.session_state.research_assistant = None
                st.rerun()
        
        # Export current session
        if st.session_state.research_results:
            if st.button("💾 Export Session", use_container_width=True):
                export_session_data()

def display_feature_showcase():
    """Display enhanced feature showcase"""
    st.markdown('<div class="section-header">✨ Research Capabilities</div>', unsafe_allow_html=True)
    
    # Feature grid
    features = [
        {"icon": "🔍", "title": "Multi-Engine Search", "desc": "Google CSE, SerpAPI, DuckDuckGo, Wikipedia"},
        {"icon": "🎥", "title": "Video Analysis", "desc": "YouTube transcripts, audio processing"},
        {"icon": "🖼️", "title": "Image Processing", "desc": "OCR text extraction, AI captions"},
        {"icon": "💬", "title": "Community Insights", "desc": "Reddit discussions and opinions"},
        {"icon": "🧠", "title": "AI-Powered Fusion", "desc": "Cross-modal analysis and insights"},
        {"icon": "📊", "title": "Smart Summarization", "desc": "Key points from all sources"}
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
                <h4 style="margin: 0.5rem 0; color: #2E86AB;">{feature['title']}</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def display_research_input():
    """Enhanced research input form with better validation"""
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
                placeholder="https://youtube.com/watch?v=...\nhttps://youtu.be/...",
                height=100,
                help="One URL per line - supports various YouTube formats"
            )
            
            st.markdown("**📄 Additional Text**")
            text_sources = st.text_area(
                "Paste Additional Content",
                placeholder="Research papers, articles, or any relevant text...",
                height=120,
                help="Add any text content to include in the analysis"
            )
        
        with col2:
            st.markdown("**📁 File Uploads**")
            
            # Enhanced file upload with progress
            uploaded_images = st.file_uploader(
                "📸 Images (JPG, PNG, GIF, etc.)",
                type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"],
                accept_multiple_files=True,
                help="Upload images for OCR text extraction and visual analysis"
            )
            
            uploaded_audio = st.file_uploader(
                "🎵 Audio Files (MP3, WAV, etc.)",
                type=["mp3", "wav", "m4a", "flac", "aac", "ogg"],
                accept_multiple_files=True,
                help="Upload audio files for speech-to-text transcription"
            )
            
            uploaded_video = st.file_uploader(
                "🎬 Video Files (MP4, AVI, etc.)",
                type=["mp4", "avi", "mov", "mkv", "webm", "flv"],
                accept_multiple_files=True,
                help="Upload video files for audio extraction and transcription"
            )
        
        # Research options
        with st.expander("🔧 Advanced Options", expanded=False):
            col3, col4 = st.columns(2)
            
            with col3:
                include_sentiment = st.checkbox(
                    "📊 Include Sentiment Analysis",
                    value=True,
                    help="Analyze sentiment of text sources"
                )
                
                include_entities = st.checkbox(
                    "🏷️ Extract Named Entities",
                    value=True,
                    help="Identify people, organizations, and locations"
                )
            
            with col4:
                max_video_duration = st.slider(
                    "⏱️ Max Video Duration (minutes)",
                    min_value=5,
                    max_value=60,
                    value=10,
                    help="Longer videos take more time to process"
                )
                
                summary_length = st.selectbox(
                    "📝 Summary Detail Level",
                    ["Brief", "Detailed", "Comprehensive"],
                    index=1,
                    help="Choose how detailed you want the summaries"
                )
        
        # Submit button with enhanced styling
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
            
            # Count total inputs
            input_count = 1  # topic
            video_urls = [url.strip() for url in video_urls_text.split('\n') if url.strip()]
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
                'uploaded_videos': uploaded_video or [],
                'options': {
                    'include_sentiment': include_sentiment,
                    'include_entities': include_entities,
                    'max_video_duration': max_video_duration,
                    'summary_length': summary_length
                }
            }
    
    return None

def save_uploaded_files(uploaded_files: Dict, upload_dir: Path) -> Dict[str, List[str]]:
    """Enhanced file saving with progress tracking"""
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
                    
                    # Create unique filename
                    timestamp = int(time.time())
                    file_path = upload_dir / f"{timestamp}_{i}_{uploaded_file.name}"
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    saved_files[save_key].append(str(file_path))
                    logger.info(f"Saved {save_key[:-1]} file: {file_path}")
                    
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")
            
            progress_bar.empty()
            status_text.empty()
    
    return saved_files

async def run_enhanced_research(research_input: Dict[str, Any]) -> tuple:
    """Enhanced research execution with progress tracking"""
    if not RESEARCH_AVAILABLE:
        st.error("❌ Research module not available. Please check installation.")
        return None, None
    
    try:
        # Initialize assistant
        assistant = AgenticResearchAssistant()
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Research Progress")
            
            progress_steps = [
                ("🔍", "Web Search", "Searching multiple engines..."),
                ("📄", "Text Processing", "Processing additional text..."),
                ("🎥", "Video Analysis", "Extracting video transcripts..."),
                ("🎵", "Audio Processing", "Transcribing audio files..."),
                ("🖼️", "Image Analysis", "Analyzing images and extracting text..."),
                ("🧠", "AI Analysis", "Generating insights and summaries...")
            ]
            
            step_containers = {}
            for i, (icon, title, desc) in enumerate(progress_steps):
                step_containers[i] = st.empty()
                step_containers[i].markdown(f"""
                <div class="progress-step">
                    {icon} <strong>{title}</strong>: {desc}
                </div>
                """, unsafe_allow_html=True)
        
        # Save uploaded files
        upload_dir = Path("uploads")
        saved_files = save_uploaded_files(research_input, upload_dir)
        
        # Update step 1 - Web Search
        step_containers[0].markdown("""
        <div class="progress-step active">
            🔍 <strong>Web Search</strong>: Searching multiple engines...
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare inputs
        video_urls = research_input.get('video_urls', [])
        image_paths = saved_files.get('images', [])
        audio_paths = saved_files.get('audio', [])
        video_paths = saved_files.get('videos', [])
        text_sources = research_input.get('text_sources')
        
        # Run research with progress updates
        insights = await assistant.research_topic(
            topic=research_input['topic'],
            video_urls=video_urls if video_urls else None,
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
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        st.error(f"❌ Research failed: {str(e)}")
        return None, None

def display_enhanced_results(insights: ResearchInsight):
    """Enhanced results display with better organization"""
    if not insights:
        return
    
    st.markdown('<div class="section-header">📊 Research Results</div>', unsafe_allow_html=True)
    
    # Enhanced metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📚 Sources Analyzed",
            insights.sources_count,
            help="Total number of sources processed"
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
        
        for i, point in enumerate(insights.fused_summary, 1):
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; border-left: 4px solid #28a745;">
                <strong>{i}.</strong> {point}
            </div>
            """, unsafe_allow_html=True)
    
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
        
        for i, question in enumerate(insights.follow_up_questions, 1):
            st.markdown(f"""
            <div style="background: #fff3cd; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; border-left: 4px solid #ffc107;">
                <strong>Q{i}:</strong> {question}
            </div>
            """, unsafe_allow_html=True)

def export_enhanced_results(insights: ResearchInsight, assistant):
    """Enhanced export functionality with multiple formats"""
    if not insights:
        return
    
    st.markdown("### 💾 Export Research Results")
    
    col1, col2, col3 = st.columns(3)
    
    # Generate export data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'topic': getattr(assistant, 'current_topic', 'Unknown') if assistant else 'Unknown',
        'research_summary': {
            'sources_analyzed': insights.sources_count,
            'confidence_score': insights.confidence_score,
            'api_integrations_used': list(st.session_state.api_status.keys()) if st.session_state.api_status else []
        },
        'insights': {
            'fused_summary': insights.fused_summary,
            'text_insights': insights.text_insights,
            'video_insights': insights.video_insights,
            'image_insights': insights.image_insights,
            'follow_up_questions': insights.follow_up_questions
        }
    }
    
    with col1:
        # JSON Export
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Text Report
        report = f"""
MULTI-MODAL RESEARCH REPORT
==========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Topic: {export_data['topic']}
Sources: {insights.sources_count}
Confidence: {insights.confidence_score:.1%}

EXECUTIVE SUMMARY
================
{chr(10).join(f'• {point}' for point in insights.fused_summary)}

TEXT SOURCE INSIGHTS
===================
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.text_insights))}

MEDIA SOURCE INSIGHTS
====================
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.video_insights))}

VISUAL CONTENT INSIGHTS
======================
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.image_insights))}

FOLLOW-UP RESEARCH QUESTIONS
===========================
{chr(10).join(f'{i+1}. {question}' for i, question in enumerate(insights.follow_up_questions))}

Generated by Multi-Modal Research Assistant
"""
        
        st.download_button(
            label="📝 Download Text Report",
            data=report,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Markdown Export
        markdown_report = f"""# Multi-Modal Research Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Topic:** {export_data['topic']}  
**Sources:** {insights.sources_count}  
**Confidence:** {insights.confidence_score:.1%}

## 🎯 Executive Summary

{chr(10).join(f'- {point}' for point in insights.fused_summary)}

## 📰 Text Source Insights

{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.text_insights))}

## 🎥 Media Source Insights

{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.video_insights))}

## 🖼️ Visual Content Insights

{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.image_insights))}

## ❓ Follow-up Research Questions

{chr(10).join(f'{i+1}. {question}' for i, question in enumerate(insights.follow_up_questions))}

---
*Generated by Multi-Modal Research Assistant*
"""
        
        st.download_button(
            label="📋 Download Markdown",
            data=markdown_report,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

def export_session_data():
    """Export current session data"""
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'api_status': st.session_state.api_status,
        'config_status': st.session_state.config_status,
        'has_results': bool(st.session_state.research_results)
    }
    
    if st.session_state.research_results:
        session_data['last_research'] = {
            'sources_count': st.session_state.research_results.sources_count,
            'confidence': st.session_state.research_results.confidence_score
        }
    
    json_str = json.dumps(session_data, indent=2)
    st.download_button(
        label="💾 Export Session",
        data=json_str,
        file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def main():
    """Enhanced main application function"""
    # Initialize
    initialize_session_state()
    display_header()
    display_enhanced_sidebar()
    
    # Check if core modules are available
    if not CONFIG_AVAILABLE or not RESEARCH_AVAILABLE:
        st.error("❌ Core modules not available. Please install required dependencies:")
        st.code("pip install -r requirements.txt", language="bash")
        st.info("💡 Run the setup script: `python setup.py`")
        st.stop()
    
    # Feature showcase
    display_feature_showcase()
    
    # Example usage section
    with st.expander("📖 How to Use This Tool", expanded=False):
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Enter Your Research Topic** - Be specific for better results
        2. **Add Sources (Optional):**
           - 🎥 YouTube video URLs for transcript analysis
           - 📄 Additional text content or research papers
           - 📸 Images for OCR and visual analysis
           - 🎵 Audio files for transcription
        3. **Configure Options** - Set analysis preferences
        4. **Start Research** - Let the AI analyze everything
        5. **Review Results** - Organized by source type
        6. **Export Reports** - Multiple formats available
        
        ### 💡 Pro Tips
        - Use specific topics like "Machine Learning Ethics in Healthcare"
        - Combine multiple source types for richer insights
        - Check the API status for enhanced search capabilities
        - Use follow-up questions for iterative research
        
        ### 🔧 API Enhancement
        Configure API keys in `.env` or `config.json` for:
        - **Google CSE**: Enhanced search results
        - **SerpAPI**: Professional search data
        - **Reddit API**: Community insights and discussions
        """)
    
    # Main content
    research_input = display_research_input()
    
    # Handle research submission
    if research_input and not st.session_state.research_in_progress:
        st.session_state.research_in_progress = True
        
        with st.spinner("🧠 Conducting comprehensive multi-modal research..."):
            try:
                # Run async research
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                insights, assistant = loop.run_until_complete(run_enhanced_research(research_input))
                loop.close()
                
                if insights:
                    st.session_state.research_results = insights
                    st.session_state.research_assistant = assistant
                    st.balloons()  # Celebration animation
                    st.success("🎉 Research completed successfully!")
                else:
                    st.error("❌ Research returned no results. Please try again with different inputs.")
                
            except Exception as e:
                st.error(f"❌ Research failed: {str(e)}")
                logger.error(f"Research error: {e}", exc_info=True)
            finally:
                st.session_state.research_in_progress = False
                st.rerun()
    
    # Display results if available
    if st.session_state.research_results:
        display_enhanced_results(st.session_state.research_results)
        export_enhanced_results(st.session_state.research_results, st.session_state.research_assistant)
        
        # Action buttons
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🆕 New Research", use_container_width=True, type="primary"):
                st.session_state.research_results = None
                st.session_state.research_assistant = None
                st.rerun()
        
        with col2:
            if st.button("🔍 Refine Search", use_container_width=True):
                st.info("💡 Use the follow-up questions above to refine your research")
        
        with col3:
            if st.button("📊 View Statistics", use_container_width=True):
                if st.session_state.research_assistant:
                    stats = st.session_state.research_assistant.get_research_statistics()
                    st.json(stats)
    
    # Footer with enhanced information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem;">
        <h4>🤖 Multi-Modal Research Assistant</h4>
        <p style="margin: 0.5rem 0;">
            <strong>Version 2.0</strong> • Enhanced with API integrations • 
            Powered by AI models and multiple data sources
        </p>
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            🔍 Web Search • 🎥 Video Analysis • 🖼️ Image Processing • 🧠 AI Fusion
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()