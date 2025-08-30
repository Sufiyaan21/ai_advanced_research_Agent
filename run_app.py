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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import torch
        import transformers
        logger.info("✅ Core dependencies found")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def setup_environment():
    """Setup the environment for the application"""
    logger.info("🔧 Setting up environment...")
    
    # Create required directories
    directories = ['temp_files', 'uploads', 'reports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    # Check for .env file
    env_file = Path('.env')
    env_example = Path('env_example.txt')
    
    if not env_file.exists() and env_example.exists():
        logger.warning("⚠️ .env file not found. Copy env_example.txt to .env and add your API keys")
        logger.info("💡 You can still run the app, but some features may be limited")
    
    # Check for config.json
    config_file = Path('config.json')
    config_template = Path('config_template.json')
    
    if not config_file.exists() and config_template.exists():
        logger.info("📋 Creating config.json from template...")
        import shutil
        shutil.copy(config_template, config_file)
        logger.info("✅ config.json created. Update it with your API keys")

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def download_spacy_model():
    """Download required spaCy model"""
    logger.info("📥 Downloading spaCy model...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        logger.info("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Failed to download spaCy model: {e}")
        logger.info("💡 Some NLP features may be limited")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    logger.info("🚀 Launching Streamlit application...")
    
    try:
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        # Launch the app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch application: {e}")

def main():
    """Main function"""
    print("🤖 Multi-Modal Research Assistant")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('streamlit_app.py').exists():
        logger.error("❌ streamlit_app.py not found. Please run this script from the project directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.info("📦 Installing missing dependencies...")
        if not install_dependencies():
            logger.error("❌ Failed to install dependencies. Please install manually:")
            logger.error("   pip install -r requirements.txt")
            sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Download spaCy model
    download_spacy_model()
    
    # Launch application
    print("\n🎉 Setup complete! Launching application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    launch_streamlit()

if __name__ == "__main__":
    main()
