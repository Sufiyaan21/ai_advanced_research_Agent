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
import nest_asyncio

# Apply nest_asyncio to allow running asyncio event loops within Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if modules are available before importing
try:
    from config import get_config, setup_environment
    from research import AgenticResearchAssistant, ResearchInsight, ResearchSource
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Failed to import required modules: {e}")

# Initialize configuration
if MODULES_AVAILABLE:
    try:
        config = get_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        st.stop()
else:
    st.error("Cannot load configuration - required modules not available")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=config.streamlit_config.page_title,
    page_icon=config.streamlit_config.page_icon,
    layout=config.streamlit_config.layout,
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
        color: #495057;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
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
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
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
        try:
            st.session_state.config_status = setup_environment()
        except Exception as e:
            st.session_state.config_status = {'error': str(e)}
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
        
        # Check if config status is available
        if 'error' in st.session_state.config_status:
            st.error(f"Configuration Error: {st.session_state.config_status['error']}")
            return
        
        # API Key Status
        st.subheader("🔑 API Keys Status")
        if 'api_keys' in st.session_state.config_status:
            api_status = st.session_state.config_status['api_keys']
            
            for service, available in api_status.items():
                status_icon = "✅" if available else "❌"
                st.markdown(f"{status_icon} **{service.replace('_', ' ').title()}**: {'Available' if available else 'Missing'}")
            
            # Missing keys warning
            missing_keys = st.session_state.config_status.get('missing_keys', [])
            if missing_keys:
                st.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
                st.info("💡 Set environment variables or update config.json to enable all features")
        
        st.divider()
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower"
        )
        
        max_results = st.slider(
            "Max Search Results",
            min_value=5,
            max_value=20,
            value=config.model_config.max_search_results,
            help="Number of search results to analyze"
        )
        
        st.divider()
        
        # System Status
        st.subheader("🔧 System Status")
        
        # Check dependency availability
        deps_status = st.session_state.config_status
        if 'whisper_available' in deps_status:
            st.markdown(f"🎤 Whisper: {'✅ Available' if deps_status['whisper_available'] else '❌ Not Available'}")
        if 'transformers_available' in deps_status:
            st.markdown(f"🤖 Transformers: {'✅ Available' if deps_status['transformers_available'] else '❌ Not Available'}")
        if 'search_available' in deps_status:
            st.markdown(f"🔍 Search: {'✅ Available' if deps_status['search_available'] else '❌ Not Available'}")
        
        st.divider()
        
        # File Upload Settings
        st.subheader("📁 Upload Settings")
        st.info(f"Max file size: {config.streamlit_config.max_upload_size_mb}MB")
        
        # Display allowed file types
        with st.expander("Allowed File Types"):
            st.write("**Images:**", ", ".join(config.streamlit_config.allowed_image_types))
            st.write("**Audio:**", ", ".join(config.streamlit_config.allowed_audio_types))
            st.write("**Video:**", ", ".join(config.streamlit_config.allowed_video_types))
        
        st.divider()
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        
        if st.button("🔄 Refresh Configuration", use_container_width=True):
            try:
                st.session_state.config_status = setup_environment()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh configuration: {e}")

def display_research_input():
    """Display research input form"""
    st.markdown('<div class="section-header">🔍 Research Input</div>', unsafe_allow_html=True)
    
    # Display feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📰 Web Search</h4>
            <p>DuckDuckGo, Wikipedia, Reddit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎥 Video Analysis</h4>
            <p>YouTube transcripts, audio processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🖼️ Image Analysis</h4>
            <p>OCR, AI captioning, visual content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 AI Fusion</h4>
            <p>Cross-modal analysis, insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.form("research_form"):
        # Topic input
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Artificial Intelligence Ethics, Climate Change Solutions, Quantum Computing Applications",
            help="Enter the main topic you want to research"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Video URLs
            video_urls_text = st.text_area(
                "YouTube Video URLs (Optional)",
                placeholder="Enter one URL per line:\nhttps://youtube.com/watch?v=VIDEO1\nhttps://youtube.com/watch?v=VIDEO2",
                help="Paste YouTube video URLs for analysis"
            )
            
            # Additional text sources
            text_sources = st.text_area(
                "Additional Text Sources (Optional)",
                placeholder="Paste any additional text, articles, or content to analyze",
                help="Add any text content you want to include in the research"
            )
        
        with col2:
            # File uploads
            st.subheader("📁 File Uploads (Optional)")
            
            # Image upload
            uploaded_images = st.file_uploader(
                "Upload Images",
                type=config.streamlit_config.allowed_image_types,
                accept_multiple_files=True,
                help="Upload images for visual analysis"
            )
            
            # Audio upload
            uploaded_audio = st.file_uploader(
                "Upload Audio Files",
                type=config.streamlit_config.allowed_audio_types,
                accept_multiple_files=True,
                help="Upload audio files for transcription and analysis"
            )
            
            # Video upload
            uploaded_videos = st.file_uploader(
                "Upload Video Files",
                type=config.streamlit_config.allowed_video_types,
                accept_multiple_files=True,
                help="Upload video files for analysis"
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
                'text_sources': text_sources if text_sources and text_sources.strip() else None,
                'uploaded_images': uploaded_images or [],
                'uploaded_audio': uploaded_audio or [],
                'uploaded_videos': uploaded_videos or []
            }
    
    return None

def save_uploaded_files(uploaded_files: Dict[str, List], upload_dir: Path):
    """Save uploaded files to disk"""
    saved_files = {'images': [], 'audio': [], 'video': []}
    
    for file_type, file_list in [('images', uploaded_files['uploaded_images']), 
                                  ('audio', uploaded_files['uploaded_audio']), 
                                  ('video', uploaded_files['uploaded_videos'])]:
        for uploaded_file in file_list:
            if uploaded_file is not None:
                try:
                    # Create file path
                    file_path = upload_dir / f"{int(time.time())}_{uploaded_file.name}"
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    saved_files[file_type].append(str(file_path))
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")
    
    return saved_files

async def run_research(research_input: Dict[str, Any]) -> tuple[ResearchInsight, AgenticResearchAssistant]:
    """Run the research process"""
    assistant = AgenticResearchAssistant()
    
    # Save uploaded files
    saved_files = save_uploaded_files(research_input, config.model_config.upload_dir)
    
    # Prepare inputs
    video_urls = research_input['video_urls']
    image_paths = saved_files['images']
    audio_paths = saved_files['audio']
    text_sources = research_input['text_sources']
    
    # Add video files to video_urls (for local processing)
    for video_path in saved_files['video']:
        video_urls.append(f"file://{video_path}")
    
    # Run research
    insights = await assistant.research_topic(
        topic=research_input['topic'],
        video_urls=video_urls if video_urls else None,
        image_paths=image_paths if image_paths else None,
        audio_paths=audio_paths if audio_paths else None,
        text_sources=text_sources
    )
    
    return insights, assistant

def display_research_results(insights: ResearchInsight):
    """Display research results in a formatted way"""
    st.markdown('<div class="section-header">📊 Research Results</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sources Analyzed", insights.sources_count)
    
    with col2:
        confidence_class = "confidence-high" if insights.confidence_score > 0.8 else "confidence-medium" if insights.confidence_score > 0.6 else "confidence-low"
        st.markdown(f'<div class="{confidence_class}">Confidence: {insights.confidence_score:.2f}</div>', unsafe_allow_html=True)
    
    with col3:
        st.metric("Text Insights", len(insights.text_insights))
    
    with col4:
        st.metric("Total Insights", len(insights.video_insights) + len(insights.image_insights))
    
    # Text Insights
    if insights.text_insights:
        st.markdown('<div class="section-header">📰 Text Source Insights</div>', unsafe_allow_html=True)
        for i, insight in enumerate(insights.text_insights, 1):
            st.markdown(f'<div class="insight-box"><strong>{i}.</strong> {insight}</div>', unsafe_allow_html=True)
    
    # Video Insights
    if insights.video_insights:
        st.markdown('<div class="section-header">🎥 Video/Audio Source Insights</div>', unsafe_allow_html=True)
        for i, insight in enumerate(insights.video_insights, 1):
            st.markdown(f'<div class="insight-box"><strong>{i}.</strong> {insight}</div>', unsafe_allow_html=True)
    
    # Image Insights
    if insights.image_insights:
        st.markdown('<div class="section-header">🖼️ Image/Visual Source Insights</div>', unsafe_allow_html=True)
        for i, insight in enumerate(insights.image_insights, 1):
            st.markdown(f'<div class="insight-box"><strong>{i}.</strong> {insight}</div>', unsafe_allow_html=True)
    
    # Fused Summary
    st.markdown('<div class="section-header">🔗 Cross-Modal Fused Analysis</div>', unsafe_allow_html=True)
    for i, point in enumerate(insights.fused_summary, 1):
        st.markdown(f'<div class="insight-box"><strong>{i}.</strong> {point}</div>', unsafe_allow_html=True)
    
    # Follow-up Questions
    st.markdown('<div class="section-header">❓ Recommended Follow-up Research</div>', unsafe_allow_html=True)
    for i, question in enumerate(insights.follow_up_questions, 1):
        st.markdown(f'<div class="insight-box"><strong>{i}.</strong> {question}</div>', unsafe_allow_html=True)

def display_source_details(assistant: AgenticResearchAssistant):
    """Display detailed source information"""
    if not assistant or not hasattr(assistant, 'all_sources') or not assistant.all_sources:
        return
    
    st.markdown('<div class="section-header">📚 Source Details</div>', unsafe_allow_html=True)
    
    # Group sources by type
    source_types = {}
    for source in assistant.all_sources:
        if source.source_type not in source_types:
            source_types[source.source_type] = []
        source_types[source.source_type].append(source)
    
    # Display sources by type
    for source_type, sources in source_types.items():
        with st.expander(f"{source_type.replace('_', ' ').title()} ({len(sources)} sources)"):
            for i, source in enumerate(sources):
                confidence_class = "confidence-high" if source.confidence > 0.8 else "confidence-medium" if source.confidence > 0.6 else "confidence-low"
                
                st.markdown(f"""
                <div class="source-box">
                    <strong>{source.title}</strong><br>
                    <span class="{confidence_class}">Confidence: {source.confidence:.2f}</span><br>
                    <small>URL: {source.url}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show content preview
                with st.expander(f"Preview content"):
                    st.text_area("Content", source.content[:500] + "..." if len(source.content) > 500 else source.content, 
                                 height=100, key=f"content_{source_type}_{i}")

def export_results(insights: ResearchInsight, assistant: Optional[AgenticResearchAssistant] = None):
    """Export research results"""
    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'topic': getattr(assistant, 'current_topic', 'Unknown Topic') if assistant else 'Unknown Topic',
            'insights': {
                'text_insights': insights.text_insights,
                'video_insights': insights.video_insights,
                'image_insights': insights.image_insights,
                'fused_summary': insights.fused_summary,
                'follow_up_questions': insights.follow_up_questions,
                'confidence_score': insights.confidence_score,
                'sources_count': insights.sources_count
            },
            'sources': [
                {
                    'title': source.title,
                    'url': source.url,
                    'source_type': source.source_type,
                    'confidence': source.confidence,
                    'metadata': source.metadata
                }
                for source in (assistant.all_sources if assistant and hasattr(assistant, 'all_sources') else [])
            ]
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as text report
        if assistant and hasattr(assistant, 'export_research_report'):
            try:
                report = assistant.export_research_report()
            except Exception as e:
                report = f"Error generating report: {e}"
        else:
            # Create a simple text report if assistant is not available
            report = f"""
Multi-Modal Research Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Topic: {export_data['topic']}

SUMMARY METRICS:
- Sources Analyzed: {insights.sources_count}
- Confidence Score: {insights.confidence_score:.2f}
- Text Insights: {len(insights.text_insights)}
- Video Insights: {len(insights.video_insights)}
- Image Insights: {len(insights.image_insights)}

TEXT INSIGHTS:
{chr(10).join(f"{i+1}. {insight}" for i, insight in enumerate(insights.text_insights))}

VIDEO INSIGHTS:
{chr(10).join(f"{i+1}. {insight}" for i, insight in enumerate(insights.video_insights))}

IMAGE INSIGHTS:
{chr(10).join(f"{i+1}. {insight}" for i, insight in enumerate(insights.image_insights))}

FUSED SUMMARY:
{chr(10).join(f"{i+1}. {point}" for i, point in enumerate(insights.fused_summary))}

FOLLOW-UP QUESTIONS:
{chr(10).join(f"{i+1}. {question}" for i, question in enumerate(insights.follow_up_questions))}
"""
        
        st.download_button(
            label="📝 Download Text Report",
            data=report,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def display_example_usage():
    """Display example usage guide"""
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        ### 🎯 Quick Start Guide
        
        1. **Enter a research topic** - Be specific for better results
        2. **Add sources (optional):**
           - YouTube video URLs for expert talks, lectures, or discussions
           - Upload images with charts, diagrams, or text
           - Upload audio files of interviews or presentations
           - Paste additional text content
        3. **Click "Start Research"** - The AI will analyze all sources
        4. **Review results** - Get insights organized by source type
        5. **Export reports** - Download JSON or text format
        
        ### 💡 Example Topics
        - "Machine Learning in Healthcare"
        - "Climate Change Solutions"
        - "Quantum Computing Applications"
        - "Renewable Energy Technologies"
        
        ### 🔍 What the AI Does
        - **Web Search**: Finds relevant articles and academic sources
        - **Video Analysis**: Extracts and summarizes spoken content
        - **Image Analysis**: Reads text and describes visual content
        - **Cross-Modal Fusion**: Combines insights from all sources
        """)

async def run_research_wrapper(research_input: Dict[str, Any], progress_placeholder, status_placeholder):
    """A wrapper to run the research and update the UI."""
    # Stage 1: Web Search
    with progress_placeholder.container():
        st.progress(0.2)
    status_placeholder.text("📰 Stage 1: Searching web sources...")
    await asyncio.sleep(0.5)
    
    # Stage 2: Video Processing
    with progress_placeholder.container():
        st.progress(0.4)
    status_placeholder.text("🎥 Stage 2: Processing video and audio content...")
    await asyncio.sleep(0.5)
    
    # Stage 3: Image Processing
    with progress_placeholder.container():
        st.progress(0.6)
    status_placeholder.text("🖼️ Stage 3: Analyzing images...")
    await asyncio.sleep(0.5)
    
    # Stage 4: Fusion
    with progress_placeholder.container():
        st.progress(0.8)
    status_placeholder.text("🧠 Stage 4: Cross-modal fusion and analysis...")
    await asyncio.sleep(0.5)
    
    # Run actual research
    insights, assistant = await run_research(research_input)
    
    return insights, assistant

def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    display_header()
    display_sidebar()
    
    # Example usage guide
    display_example_usage()
    
    # Main content area
    research_input = display_research_input()
    
    # Handle research submission
    if research_input and not st.session_state.research_in_progress:
        st.session_state.research_in_progress = True
        
        # Display progress
        with st.container():
            st.markdown("🚀 Starting research process...")
            
            # Create progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                # Run the async research function
                insights, assistant = asyncio.run(run_research_wrapper(research_input, progress_placeholder, status_placeholder))
                
                with progress_placeholder.container():
                    st.progress(1.0)
                status_placeholder.text("✅ Research completed successfully!")
                
                # Store results
                st.session_state.research_results = insights
                st.session_state.research_assistant = assistant
                st.session_state.research_in_progress = False
                
                # Clear progress indicators after a moment
                time.sleep(1)
                progress_placeholder.empty()
                status_placeholder.empty()
                
                # Trigger rerun to show results
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Research failed: {str(e)}")
                st.session_state.research_in_progress = False
                progress_placeholder.empty()
                status_placeholder.empty()
                
                # Show detailed error for debugging
                with st.expander("Error Details"):
                    st.code(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.research_results:
        display_research_results(st.session_state.research_results)
        
        # Show source details if enabled
        if config.streamlit_config.show_source_details and st.session_state.research_assistant:
            display_source_details(st.session_state.research_assistant)
        
        # Export functionality
        if config.streamlit_config.enable_download:
            export_results(st.session_state.research_results, st.session_state.research_assistant)
        
        # Clear results button
        st.divider()
        if st.button("🗑️ Clear Results and Start New Research", type="secondary"):
            st.session_state.research_results = None
            st.session_state.research_assistant = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Multi-Modal Research Assistant v1.0 | Built with Streamlit</p>
        <p>Powered by AI models: Whisper, BLIP, BART, spaCy | Sources: DuckDuckGo, Wikipedia, Reddit</p>
    </div>
    """, unsafe_allow_html=True)

# Run the async main function
if __name__ == "__main__":
    main()