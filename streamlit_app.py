"""
Multi-Modal Research Assistant - Streamlit Web Application
A comprehensive web interface for conducting multi-modal research
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import sys
import os

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

# Custom CSS for better styling
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
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #ffffff;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'research_in_progress' not in st.session_state:
        st.session_state.research_in_progress = False
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

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🤖 Multi-Modal Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Conduct comprehensive research by analyzing text, video, audio, and images from multiple sources
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with configuration and settings"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check module availability
        st.subheader("📦 System Status")
        st.markdown(f"Config Module: {'✅' if CONFIG_AVAILABLE else '❌'}")
        st.markdown(f"Research Module: {'✅' if RESEARCH_AVAILABLE else '❌'}")
        
        if not CONFIG_AVAILABLE or not RESEARCH_AVAILABLE:
            st.error("Core modules not available. Please check installation.")
            return
        
        # Check if config status is available
        if 'error' in st.session_state.config_status:
            st.warning(f"Config Warning: {st.session_state.config_status['error']}")
        
        # API Key Status
        st.subheader("🔑 API Keys Status")
        if 'api_keys' in st.session_state.config_status:
            api_status = st.session_state.config_status['api_keys']
            
            for service, available in api_status.items():
                status_icon = "✅" if available else "⚠️"
                st.markdown(f"{status_icon} **{service.replace('_', ' ').title()}**")
        
        st.divider()
        
        # Model Settings
        st.subheader("🤖 Settings")
        
        if config:
            whisper_model = st.selectbox(
                "Whisper Model",
                ["tiny", "base", "small"],
                index=1,
                help="Larger models are more accurate but slower"
            )
            
            max_results = st.slider(
                "Max Search Results",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of search results to analyze"
            )
        
        st.divider()
        
        # Quick Actions
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

def display_research_input():
    """Display research input form"""
    st.markdown('<div class="section-header">🔍 Research Input</div>', unsafe_allow_html=True)
    
    # Display feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📰 Web Search</h4>
            <p>Multiple sources</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎥 Video Analysis</h4>
            <p>YouTube & audio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🖼️ Image Analysis</h4>
            <p>OCR & captions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 AI Fusion</h4>
            <p>Cross-modal insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.form("research_form"):
        # Topic input
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Artificial Intelligence Ethics, Climate Change Solutions",
            help="Enter the main topic you want to research"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Video URLs
            video_urls_text = st.text_area(
                "YouTube Video URLs (Optional)",
                placeholder="One URL per line",
                height=100,
                help="Paste YouTube video URLs for analysis"
            )
            
            # Additional text sources
            text_sources = st.text_area(
                "Additional Text (Optional)",
                placeholder="Paste any additional text or content",
                height=100,
                help="Add any text content you want to include"
            )
        
        with col2:
            # File uploads
            st.markdown("**📁 File Uploads (Optional)**")
            
            # Simplified file types
            image_types = ["jpg", "jpeg", "png", "gif", "bmp"]
            audio_types = ["mp3", "wav", "m4a"]
            video_types = ["mp4", "avi", "mov"]
            
            # Image upload
            uploaded_images = st.file_uploader(
                "Upload Images",
                type=image_types,
                accept_multiple_files=True,
                help="Upload images for visual analysis"
            )
            
            # Audio upload
            uploaded_audio = st.file_uploader(
                "Upload Audio",
                type=audio_types,
                accept_multiple_files=True,
                help="Upload audio files for transcription"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Start Research",
            use_container_width=True,
            type="primary"
        )
        
        if submitted and topic:
            return {
                'topic': topic,
                'video_urls': [url.strip() for url in video_urls_text.split('\n') if url.strip()],
                'text_sources': text_sources if text_sources.strip() else None,
                'uploaded_images': uploaded_images or [],
                'uploaded_audio': uploaded_audio or [],
                'uploaded_videos': []
            }
    
    return None

def save_uploaded_files(uploaded_files: Dict, upload_dir: Path):
    """Save uploaded files to disk"""
    saved_files = {'images': [], 'audio': [], 'video': []}
    
    # Ensure upload directory exists
    upload_dir.mkdir(exist_ok=True)
    
    for file_type, file_list in [
        ('images', uploaded_files.get('uploaded_images', [])),
        ('audio', uploaded_files.get('uploaded_audio', [])),
        ('video', uploaded_files.get('uploaded_videos', []))
    ]:
        for uploaded_file in file_list:
            if uploaded_file is not None:
                try:
                    # Create file path
                    file_path = upload_dir / f"{int(time.time())}_{uploaded_file.name}"
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    saved_files[file_type].append(str(file_path))
                    logger.info(f"Saved {file_type[:-1]} file: {file_path}")
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")
    
    return saved_files

async def run_research(research_input: Dict[str, Any]):
    """Run the research process"""
    if not RESEARCH_AVAILABLE:
        st.error("Research module not available. Please check installation.")
        return None, None
    
    try:
        assistant = AgenticResearchAssistant()
        
        # Save uploaded files
        upload_dir = Path("uploads")
        saved_files = save_uploaded_files(research_input, upload_dir)
        
        # Prepare inputs
        video_urls = research_input.get('video_urls', [])
        image_paths = saved_files.get('images', [])
        audio_paths = saved_files.get('audio', [])
        text_sources = research_input.get('text_sources')
        
        # Run research
        insights = await assistant.research_topic(
            topic=research_input['topic'],
            video_urls=video_urls if video_urls else None,
            image_paths=image_paths if image_paths else None,
            audio_paths=audio_paths if audio_paths else None,
            text_sources=text_sources
        )
        
        return insights, assistant
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        st.error(f"Research failed: {str(e)}")
        return None, None

def display_research_results(insights):
    """Display research results"""
    if not insights:
        return
    
    st.markdown('<div class="section-header">📊 Research Results</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sources Analyzed", insights.sources_count)
    
    with col2:
        confidence_class = "high" if insights.confidence_score > 0.8 else "medium" if insights.confidence_score > 0.6 else "low"
        st.metric("Confidence", f"{insights.confidence_score:.1%}")
    
    with col3:
        st.metric("Text Insights", len(insights.text_insights))
    
    with col4:
        total_insights = len(insights.video_insights) + len(insights.image_insights)
        st.metric("Media Insights", total_insights)
    
    # Text Insights
    if insights.text_insights:
        st.markdown("### 📰 Text Source Insights")
        for i, insight in enumerate(insights.text_insights, 1):
            st.info(f"**{i}.** {insight}")
    
    # Video Insights
    if insights.video_insights:
        st.markdown("### 🎥 Video/Audio Insights")
        for i, insight in enumerate(insights.video_insights, 1):
            st.info(f"**{i}.** {insight}")
    
    # Image Insights
    if insights.image_insights:
        st.markdown("### 🖼️ Image Insights")
        for i, insight in enumerate(insights.image_insights, 1):
            st.info(f"**{i}.** {insight}")
    
    # Fused Summary
    st.markdown("### 🔗 Cross-Modal Analysis")
    for i, point in enumerate(insights.fused_summary, 1):
        st.success(f"**{i}.** {point}")
    
    # Follow-up Questions
    st.markdown("### ❓ Follow-up Research Questions")
    for i, question in enumerate(insights.follow_up_questions, 1):
        st.warning(f"**{i}.** {question}")

def export_results(insights, assistant):
    """Export research results"""
    if not insights:
        return
    
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'topic': getattr(assistant, 'current_topic', 'Unknown') if assistant else 'Unknown',
            'insights': {
                'text_insights': insights.text_insights,
                'video_insights': insights.video_insights,
                'image_insights': insights.image_insights,
                'fused_summary': insights.fused_summary,
                'follow_up_questions': insights.follow_up_questions,
                'confidence_score': insights.confidence_score,
                'sources_count': insights.sources_count
            }
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as text
        report = f"""
Multi-Modal Research Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Topic: {export_data['topic']}

SUMMARY:
- Sources: {insights.sources_count}
- Confidence: {insights.confidence_score:.1%}

TEXT INSIGHTS:
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.text_insights))}

VIDEO INSIGHTS:
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.video_insights))}

IMAGE INSIGHTS:
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights.image_insights))}

FUSED SUMMARY:
{chr(10).join(f'{i+1}. {point}' for i, point in enumerate(insights.fused_summary))}

FOLLOW-UP QUESTIONS:
{chr(10).join(f'{i+1}. {question}' for i, question in enumerate(insights.follow_up_questions))}
"""
        
        st.download_button(
            label="📝 Download Report",
            data=report,
            file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    display_header()
    display_sidebar()
    
    # Check if core modules are available
    if not CONFIG_AVAILABLE or not RESEARCH_AVAILABLE:
        st.error("❌ Core modules not available. Please install required dependencies:")
        st.code("pip install -r requirements.txt")
        st.stop()
    
    # Example usage
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Enter a research topic** - Be specific for better results
        2. **Add sources (optional):** YouTube URLs, text, or upload files
        3. **Click "Start Research"** to analyze all sources
        4. **Review results** organized by source type
        5. **Export reports** in JSON or text format
        
        **Example Topics:**
        - "Artificial Intelligence in Healthcare"
        - "Climate Change Solutions"
        - "Quantum Computing Applications"
        """)
    
    # Main content
    research_input = display_research_input()
    
    # Handle research submission
    if research_input and not st.session_state.research_in_progress:
        st.session_state.research_in_progress = True
        
        with st.spinner("🚀 Conducting research... This may take a few moments."):
            try:
                # Run async research
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                insights, assistant = loop.run_until_complete(run_research(research_input))
                loop.close()
                
                if insights:
                    st.session_state.research_results = insights
                    st.session_state.research_assistant = assistant
                    st.success("✅ Research completed successfully!")
                else:
                    st.error("Research returned no results. Please try again.")
                
            except Exception as e:
                st.error(f"❌ Research failed: {str(e)}")
                logger.error(f"Research error: {e}", exc_info=True)
            finally:
                st.session_state.research_in_progress = False
                st.rerun()
    
    # Display results if available
    if st.session_state.research_results:
        display_research_results(st.session_state.research_results)
        export_results(st.session_state.research_results, st.session_state.research_assistant)
        
        # Clear results button
        st.divider()
        if st.button("🗑️ Clear Results and Start New Research"):
            st.session_state.research_results = None
            st.session_state.research_assistant = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 Multi-Modal Research Assistant v1.0</p>
        <p>Powered by AI models and multiple data sources</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()