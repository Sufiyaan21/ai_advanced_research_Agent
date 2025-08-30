#!/usr/bin/env python3
"""
Multi-Modal Research Assistant - Application Launcher
This script sets up the environment and launches the Streamlit application
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_package_installed(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'torch': 'torch', 
        'transformers': 'transformers',
        'cv2': 'opencv-python',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'pathlib': None  # Built-in
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        if import_name == 'pathlib':  # Skip built-in modules
            continue
            
        if not check_package_installed(import_name):
            missing_packages.append(package_name or import_name)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False, missing_packages
    
    logger.info("✅ Core dependencies found")
    return True, []

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install requirements
        if Path('requirements.txt').exists():
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        else:
            # Install minimal requirements if requirements.txt is missing
            minimal_reqs = [
                'streamlit>=1.28.0',
                'torch>=2.0.0',
                'transformers>=4.36.0',
                'opencv-python>=4.8.0',
                'requests>=2.31.0',
                'aiohttp>=3.9.0',
                'duckduckgo-search>=4.0.0',
                'wikipedia>=1.4.0',
                'python-dotenv>=1.0.0',
                'nest-asyncio>=1.5.6'
            ]
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + minimal_reqs)
        
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def download_spacy_model():
    """Download required spaCy model"""
    logger.info("📥 Checking spaCy model...")
    
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model already available")
            return True
        except OSError:
            logger.info("📥 Downloading spaCy model...")
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            logger.info("✅ spaCy model downloaded successfully")
            return True
    except ImportError:
        logger.warning("⚠️ spaCy not installed, skipping model download")
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Failed to download spaCy model: {e}")
        logger.info("💡 Some NLP features may be limited")
        return False

def setup_environment():
    """Setup the environment for the application"""
    logger.info("🔧 Setting up environment...")
    
    # Create required directories
    directories = ['temp_files', 'uploads', 'reports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    # Check for configuration files
    config_file = Path('config.json')
    config_template = Path('config_template.json')
    
    if not config_file.exists():
        if config_template.exists():
            logger.info("📋 Creating config.json from template...")
            import shutil
            shutil.copy(config_template, config_file)
            logger.info("✅ config.json created. Update it with your API keys if needed")
        else:
            # Create a basic config file
            logger.info("📋 Creating basic config.json...")
            basic_config = {
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
                    "allowed_image_types": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                    "allowed_audio_types": ["mp3", "wav", "m4a", "flac", "aac"],
                    "allowed_video_types": ["mp4", "avi", "mov", "mkv", "webm"],
                    "show_source_details": True,
                    "enable_download": True
                }
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(basic_config, f, indent=2)
            logger.info("✅ Basic config.json created")
    
    # Check for .env file
    env_file = Path('.env')
    env_example = Path('env_example.txt')
    
    if not env_file.exists() and env_example.exists():
        logger.warning("⚠️ .env file not found. Copy env_example.txt to .env and add your API keys")
        logger.info("💡 You can still run the app, but some features may be limited")

def check_streamlit_app():
    """Check if streamlit_app.py exists and is valid"""
    app_file = Path('streamlit_app.py')
    if not app_file.exists():
        logger.error("❌ streamlit_app.py not found")
        return False

    # Basic syntax check
    try:
        with open(app_file, 'r', encoding='utf-8') as f:  # <-- specify encoding
            content = f.read()
            compile(content, app_file, 'exec')
        logger.info("✅ Streamlit app file is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in streamlit_app.py: {e}")
        return False
    except UnicodeDecodeError as e:
        logger.error(f"❌ Unicode decode error in streamlit_app.py: {e}")
        return False
def launch_streamlit():
    """Launch the Streamlit application"""
    logger.info("🚀 Launching Streamlit application...")
    
    try:
        # Set Streamlit configuration
        env_vars = {
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_SERVER_ADDRESS': 'localhost',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Launch the app
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ]
        
        logger.info("🌐 Application will be available at: http://localhost:8501")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch application: {e}")
        logger.info("💡 Try running manually: streamlit run streamlit_app.py")

def show_startup_info():
    """Show startup information and instructions"""
    print("🤖 Multi-Modal Research Assistant")
    print("=" * 50)
    print("🔧 Setting up your research environment...")
    print("")
    
    # Check system requirements
    if not check_python_version():
        print("❌ Please install Python 3.8 or higher")
        return False
    
    # Check if we're in the right directory
    if not Path('streamlit_app.py').exists():
        print("❌ streamlit_app.py not found")
        print("💡 Please run this script from the project directory")
        return False
    
    return True

def main():
    """Main function"""
    if not show_startup_info():
        sys.exit(1)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"📦 Missing dependencies: {', '.join(missing)}")
        print("🔄 Attempting to install...")
        
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            print("💡 Please install manually:")
            print("   pip install -r requirements.txt")
            sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Download spaCy model (optional)
    download_spacy_model()
    
    # Check streamlit app
    if not check_streamlit_app():
        print("❌ Streamlit app file has issues")
        sys.exit(1)
    
    # Launch application
    print("\n🎉 Setup complete! Launching application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    print("")
    
    launch_streamlit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)
