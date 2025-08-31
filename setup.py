#!/usr/bin/env python3
"""
One-click setup script for Multi-Modal Research Assistant
Handles dependencies, configuration, and initial setup
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(message):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🔧 {message}")
    print("="*60)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📌 {description}...")
    try:
        if isinstance(command, str):
            command = command.split()
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"⚠️ {description} - Warning: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies first"""
    print_header("Installing Core Dependencies")
    
    core_deps = [
        "streamlit>=1.28.0",
        "aiohttp>=3.8.0",
        "nest-asyncio>=1.5.6",
        "requests>=2.28.0",
        "python-dotenv>=0.19.0",
        "duckduckgo-search>=4.0.0",
        "wikipedia-api>=0.5.0",
        "tenacity>=8.2.0",
        "numpy>=1.23.0",
        "Pillow>=9.0.0"
    ]
    
    for dep in core_deps:
        run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep.split('>')[0]}")
    
    return True

def install_optional_dependencies():
    """Install optional dependencies with fallbacks"""
    print_header("Installing Optional Dependencies")
    
    optional_deps = [
        ("youtube-transcript-api>=0.6.0", "YouTube transcript extraction"),
        ("yt-dlp>=2023.1.6", "Video downloading"),
        ("praw>=7.6.0", "Reddit integration"),
        ("opencv-python-headless>=4.7.0", "Image processing"),
        ("pytesseract>=0.3.10", "OCR capabilities"),
        ("spacy>=3.5.0", "NLP processing"),
    ]
    
    for dep, description in optional_deps:
        run_command(f"{sys.executable} -m pip install {dep}", description)
    
    # Try to install transformers (may fail on some systems)
    print("\n📌 Attempting to install AI models (this may take time)...")
    run_command(f"{sys.executable} -m pip install transformers>=4.30.0", "Transformers library")
    
    # Try to install torch (CPU version for compatibility)
    print("\n📌 Attempting to install PyTorch (CPU version)...")
    run_command(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", "PyTorch CPU")
    
    # Try to install Whisper
    print("\n📌 Attempting to install Whisper (audio processing)...")
    run_command(f"{sys.executable} -m pip install openai-whisper", "Whisper model")

def download_spacy_model():
    """Download spaCy language model"""
    print_header("Setting up spaCy Language Model")
    
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("✅ spaCy model already installed")
        except:
            run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Downloading spaCy English model")
    except ImportError:
        print("⚠️ spaCy not installed, skipping model download")

def create_directories():
    """Create required directories"""
    print_header("Creating Required Directories")
    
    directories = ["uploads", "temp_files", "reports"]
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def create_config_file():
    """Create configuration file if it doesn't exist"""
    print_header("Setting up Configuration")
    
    config_file = Path("config.json")
    if not config_file.exists():
        config_data = {
            "api_config": {
                "openai_api_key": None,
                "google_cse_api_key": None,
                "google_cse_search_engine_id": None,
                "reddit_client_id": None,
                "reddit_client_secret": None,
                "reddit_user_agent": "MultiModalResearchAssistant/1.0",
                "assemblyai_api_key": None,
                "serpapi_api_key": None,
                "huggingface_api_key": None
            },
            "model_config": {
                "whisper_model_size": "base",
                "max_search_results": 10,
                "max_text_length": 4000,
                "max_video_duration": 600,
                "upload_dir": "uploads",
                "temp_dir": "temp_files",
                "reports_dir": "reports"
            },
            "streamlit_config": {
                "page_title": "Multi-Modal Research Assistant",
                "page_icon": "🤖",
                "layout": "wide",
                "max_upload_size_mb": 200,
                "allowed_image_types": ["jpg", "jpeg", "png", "gif", "bmp"],
                "allowed_audio_types": ["mp3", "wav", "m4a"],
                "allowed_video_types": ["mp4", "avi", "mov"],
                "show_source_details": True,
                "enable_download": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print("✅ Created config.json")
    else:
        print("✅ config.json already exists")
    
    # Create .env file template
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Multi-Modal Research Assistant - Environment Variables
# Add your API keys here (optional - for enhanced features)

# OpenAI API Key (for advanced text processing)
# OPENAI_API_KEY=your-openai-api-key-here

# Google Custom Search Engine (for enhanced web search)
# GOOGLE_CSE_API_KEY=your-google-cse-api-key
# GOOGLE_CSE_ID=your-google-cse-id

# Reddit API (for community insights)
# REDDIT_CLIENT_ID=your-reddit-client-id
# REDDIT_CLIENT_SECRET=your-reddit-client-secret

# Optional: Custom configuration
# WHISPER_MODEL=base
# MAX_SEARCH_RESULTS=10
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env template")

def test_imports():
    """Test if critical imports work"""
    print_header("Testing Installation")
    
    critical_modules = [
        ("streamlit", "Streamlit (Web Interface)"),
        ("aiohttp", "Async HTTP"),
        ("duckduckgo_search", "Web Search"),
        ("requests", "HTTP Requests"),
    ]
    
    success_count = 0
    total_count = len(critical_modules)
    
    for module_name, description in critical_modules:
        try:
            __import__(module_name)
            print(f"✅ {description} - OK")
            success_count += 1
        except ImportError:
            print(f"❌ {description} - Failed")
    
    print(f"\n📊 Test Result: {success_count}/{total_count} critical modules working")
    return success_count == total_count

def main():
    """Main setup function"""
    print("🤖 Multi-Modal Research Assistant - Setup Script")
    print("="*60)
    print("This script will set up your environment automatically.\n")
    
    # Step 1: Upgrade pip
    print_header("Upgrading pip")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install core dependencies
    install_core_dependencies()
    
    # Step 3: Install optional dependencies
    install_optional_dependencies()
    
    # Step 4: Download spaCy model
    download_spacy_model()
    
    # Step 5: Create directories
    create_directories()
    
    # Step 6: Create config files
    create_config_file()
    
    # Step 7: Test installation
    test_success = test_imports()
    
    # Final summary
    print_header("Setup Complete!")
    
    if test_success:
        print("""
✅ Setup completed successfully!

To start the application:
1. Run: python run_app.py
   OR
2. Run: streamlit run streamlit_app.py

The app will open at: http://localhost:8501

Optional: Add API keys to .env file for enhanced features:
- OpenAI API key for advanced text processing
- Google CSE for better search results
- Reddit API for community insights

Enjoy using the Multi-Modal Research Assistant! 🚀
""")
    else:
        print("""
⚠️ Setup completed with some warnings.

Some features may be limited, but the core app should work.

To start the application:
1. Run: python run_app.py
   OR
2. Run: streamlit run streamlit_app.py

If you encounter issues, try:
- pip install -r requirements.txt
- Check error messages for missing dependencies

The app will still work with limited features! 🚀
""")
    
    return 0 if test_success else 1

if __name__ == "__main__":
    sys.exit(main())