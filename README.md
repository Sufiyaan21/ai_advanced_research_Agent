# 🤖 Multi-Modal Research Assistant

A comprehensive AI-powered research assistant that analyzes text, video, audio, and images from multiple sources to provide unified research insights.

## ✨ Features

### 🔍 Multi-Source Research
- **Web Search**: DuckDuckGo, Wikipedia, Reddit integration
- **Video Analysis**: YouTube transcript extraction and Whisper transcription
- **Audio Processing**: Speech-to-text with OpenAI Whisper
- **Image Analysis**: OCR text extraction and AI-powered image captioning
- **Cross-Modal Fusion**: Intelligent combination of insights from all sources

### 🎯 AI-Powered Analysis
- **Text Summarization**: BART-based content summarization
- **Sentiment Analysis**: RoBERTa-powered sentiment detection
- **Entity Extraction**: spaCy-based named entity recognition
- **Image Captioning**: BLIP-2 model for visual content understanding
- **OCR Processing**: Tesseract for text extraction from images

### 🌐 Web Interface
- **Streamlit Dashboard**: Modern, intuitive web interface
- **File Upload**: Support for images, audio, and video files
- **Real-time Progress**: Live updates during research process
- **Export Options**: JSON and text report downloads
- **Configuration Management**: Easy API key setup and model configuration

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd agentic-research-assistant

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
python run_app.py
```

### 2. Configuration

#### Option A: Environment Variables (Recommended)
```bash
# Copy the example file
cp env_example.txt .env

# Edit .env with your API keys
nano .env
```

#### Option B: Configuration File
```bash
# Copy the template
cp config_template.json config.json

# Edit config.json with your settings
nano config.json
```

### 3. Launch the Application

```bash
# Using the launcher script (recommended)
python run_app.py

# Or directly with Streamlit
streamlit run streamlit_app.py
```

The application will open at `http://localhost:8501`

## 🔑 API Keys Setup

### Required API Keys

| Service | Purpose | Required | Setup Guide |
|---------|---------|----------|-------------|
| **OpenAI** | Advanced text processing | Optional | [Get API Key](https://platform.openai.com/api-keys) |
| **Google CSE** | Enhanced web search | Optional | [Setup Guide](https://developers.google.com/custom-search/v1/introduction) |
| **Reddit** | Community insights | Optional | [Reddit API](https://www.reddit.com/prefs/apps) |
| **AssemblyAI** | Advanced transcription | Optional | [AssemblyAI](https://www.assemblyai.com/) |
| **SerpAPI** | Enhanced search results | Optional | [SerpAPI](https://serpapi.com/) |
| **Hugging Face** | Model access | Optional | [HF Tokens](https://huggingface.co/settings/tokens) |

### Environment Variables

```bash
# Core APIs
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_CSE_API_KEY=your-google-cse-api-key
GOOGLE_CSE_ID=your-google-cse-id

# Community APIs
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret

# Advanced APIs
ASSEMBLYAI_API_KEY=your-assemblyai-api-key
SERPAPI_KEY=your-serpapi-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

## 📖 Usage Guide

### 1. Basic Research

1. **Enter Topic**: Type your research topic in the main input field
2. **Add Sources**: Optionally add YouTube URLs, text content, or upload files
3. **Start Research**: Click "Start Research" to begin the analysis
4. **View Results**: Review insights organized by source type and fused analysis

### 2. Multi-Modal Research

#### Text Sources
- Paste articles, blog posts, or any text content
- Web search automatically finds relevant sources
- Wikipedia provides authoritative background

#### Video Sources
- Add YouTube video URLs for transcript analysis
- Upload video files for local processing
- Automatic speech-to-text transcription

#### Audio Sources
- Upload audio files (MP3, WAV, M4A, etc.)
- Automatic transcription with Whisper
- Sentiment and content analysis

#### Image Sources
- Upload images for visual analysis
- OCR text extraction from images
- AI-powered image captioning and description

### 3. Understanding Results

#### Source Insights
- **Text Insights**: Summarized findings from web sources
- **Video Insights**: Key points from video/audio content
- **Image Insights**: Visual content analysis and extracted text

#### Fused Analysis
- Cross-modal validation of findings
- Unified summary combining all sources
- Confidence scoring and source attribution

#### Follow-up Questions
- AI-generated questions for deeper research
- Based on gaps in current analysis
- Tailored to your specific topic

## 🛠️ Advanced Configuration

### Model Settings

```json
{
  "model": {
    "whisper_model": "base",        // tiny, base, small, medium, large
    "max_search_results": 10,       // Number of search results
    "max_video_duration": 3600,     // Max video length in seconds
    "max_text_length": 2000,        // Max text processing length
    "max_summary_length": 200,      // Summary length limit
    "min_summary_length": 50        // Minimum summary length
  }
}
```

### File Upload Settings

```json
{
  "streamlit": {
    "max_upload_size_mb": 100,      // Max file size
    "allowed_image_types": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
    "allowed_audio_types": [".mp3", ".wav", ".m4a", ".flac", ".aac"],
    "allowed_video_types": [".mp4", ".avi", ".mov", ".mkv", ".webm"]
  }
}
```

## 📊 Example Research Topics

### Technology & AI
- "Artificial Intelligence Ethics in Healthcare"
- "Quantum Computing Applications in Cryptography"
- "Machine Learning Bias and Fairness"

### Science & Research
- "Climate Change Mitigation Strategies"
- "CRISPR Gene Editing Applications"
- "Space Exploration Technologies"

### Business & Economics
- "Sustainable Business Models"
- "Cryptocurrency Market Trends"
- "Remote Work Impact on Productivity"

## 🔧 Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check specific packages
python -c "import streamlit, torch, transformers; print('All packages installed')"
```

#### 2. spaCy Model Missing
```bash
# Download the model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded')"
```

#### 3. API Key Issues
- Check environment variables: `echo $OPENAI_API_KEY`
- Verify config.json format
- Test API keys individually

#### 4. File Upload Problems
- Check file size limits
- Verify file format support
- Ensure sufficient disk space

### Performance Optimization

#### For Large Files
- Use smaller Whisper models for faster processing
- Limit search results for quicker analysis
- Process files in smaller batches

#### For Better Accuracy
- Use larger Whisper models (medium/large)
- Increase search result limits
- Enable all available API integrations

## 📁 Project Structure

```
agentic-research-assistant/
├── streamlit_app.py          # Main Streamlit application
├── research.py               # Core research engine
├── config.py                 # Configuration management
├── run_app.py               # Application launcher
├── requirements.txt         # Python dependencies
├── config_template.json     # Configuration template
├── env_example.txt          # Environment variables example
├── README.md               # This file
├── temp_files/             # Temporary processing files
├── uploads/                # User uploaded files
└── reports/                # Generated reports
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Hugging Face Transformers** for NLP models
- **Streamlit** for the web interface
- **DuckDuckGo** for search capabilities
- **Wikipedia** for authoritative content
- **Reddit** for community insights

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration guide

---

**Built with ❤️ for researchers, students, and knowledge seekers**
