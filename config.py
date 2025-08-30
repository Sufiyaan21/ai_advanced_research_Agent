"""
Configuration management for the Multi-Modal Research Assistant
Handles API keys, model settings, and application configuration
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration settings"""
    openai_api_key: Optional[str] = None
    google_cse_api_key: Optional[str] = None
    google_cse_search_engine_id: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: Optional[str] = None
    assemblyai_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

@dataclass
class ModelConfig:
    """Model configuration settings"""
    whisper_model_size: str = "base"
    max_search_results: int = 10
    max_text_length: int = 4000
    max_video_duration: int = 600  # 10 minutes
    upload_dir: Path = Path("uploads")
    temp_dir: Path = Path("temp_files")
    reports_dir: Path = Path("reports")

@dataclass
class StreamlitConfig:
    """Streamlit application configuration"""
    page_title: str = "Multi-Modal Research Assistant"
    page_icon: str = "🤖"
    layout: str = "wide"
    max_upload_size_mb: int = 200
    allowed_image_types: list = None
    allowed_audio_types: list = None
    allowed_video_types: list = None
    show_source_details: bool = True
    enable_download: bool = True
    
    def __post_init__(self):
        if self.allowed_image_types is None:
            self.allowed_image_types = ["jpg", "jpeg", "png", "gif", "bmp", "webp"]
        if self.allowed_audio_types is None:
            self.allowed_audio_types = ["mp3", "wav", "m4a", "flac", "aac"]
        if self.allowed_video_types is None:
            self.allowed_video_types = ["mp4", "avi", "mov", "mkv", "webm"]

@dataclass
class AppConfig:
    """Main application configuration"""
    api_config: APIConfig
    model_config: ModelConfig
    streamlit_config: StreamlitConfig

def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}

def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {}
    
    # API Keys
    api_keys = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'google_cse_api_key': os.getenv('GOOGLE_CSE_API_KEY'),
        'google_cse_search_engine_id': os.getenv('GOOGLE_CSE_SEARCH_ENGINE_ID'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'reddit_user_agent': os.getenv('REDDIT_USER_AGENT'),
        'assemblyai_api_key': os.getenv('ASSEMBLYAI_API_KEY'),
        'serpapi_api_key': os.getenv('SERPAPI_API_KEY'),
        'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
    }
    
    # Model settings
    model_settings = {
        'whisper_model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),
        'max_search_results': int(os.getenv('MAX_SEARCH_RESULTS', '10')),
        'max_text_length': int(os.getenv('MAX_TEXT_LENGTH', '4000')),
        'max_video_duration': int(os.getenv('MAX_VIDEO_DURATION', '600')),
    }
    
    # Streamlit settings
    streamlit_settings = {
        'page_title': os.getenv('STREAMLIT_PAGE_TITLE', 'Multi-Modal Research Assistant'),
        'page_icon': os.getenv('STREAMLIT_PAGE_ICON', '🤖'),
        'layout': os.getenv('STREAMLIT_LAYOUT', 'wide'),
        'max_upload_size_mb': int(os.getenv('STREAMLIT_MAX_UPLOAD_SIZE_MB', '200')),
    }
    
    config.update({
        'api_config': api_keys,
        'model_config': model_settings,
        'streamlit_config': streamlit_settings
    })
    
    return config

def get_config() -> AppConfig:
    """Get the complete application configuration"""
    # Load from environment variables first
    env_config = load_config_from_env()
    
    # Load from config file
    config_path = Path("config.json")
    file_config = load_config_from_file(config_path)
    
    # Merge configurations (env takes precedence)
    merged_config = {}
    
    # Merge API config
    api_config = {**file_config.get('api_config', {}), **env_config.get('api_config', {})}
    merged_config['api_config'] = APIConfig(**api_config)
    
    # Merge model config
    model_config = {**file_config.get('model_config', {}), **env_config.get('model_config', {})}
    merged_config['model_config'] = ModelConfig(**model_config)
    
    # Merge streamlit config
    streamlit_config = {**file_config.get('streamlit_config', {}), **env_config.get('streamlit_config', {})}
    merged_config['streamlit_config'] = StreamlitConfig(**streamlit_config)
    
    return AppConfig(**merged_config)

def setup_environment() -> Dict[str, Any]:
    """Setup the environment and return status information"""
    config = get_config()
    
    # Create necessary directories
    config.model_config.upload_dir.mkdir(exist_ok=True)
    config.model_config.temp_dir.mkdir(exist_ok=True)
    config.model_config.reports_dir.mkdir(exist_ok=True)
    
    # Check API key availability
    api_status = {
        'openai': bool(config.api_config.openai_api_key),
        'google_cse': bool(config.api_config.google_cse_api_key and config.api_config.google_cse_search_engine_id),
        'reddit': bool(config.api_config.reddit_client_id and config.api_config.reddit_client_secret),
        'assemblyai': bool(config.api_config.assemblyai_api_key),
        'serpapi': bool(config.api_config.serpapi_api_key),
        'huggingface': bool(config.api_config.huggingface_api_key),
    }
    
    missing_keys = [key for key, available in api_status.items() if not available]
    
    return {
        'api_keys': api_status,
        'missing_keys': missing_keys,
        'config_loaded': True,
        'directories_created': True
    }

def save_config(config: AppConfig, config_path: Path = Path("config.json")):
    """Save configuration to JSON file"""
    try:
        config_dict = {
            'api_config': asdict(config.api_config),
            'model_config': asdict(config.model_config),
            'streamlit_config': asdict(config.streamlit_config)
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"API Keys available: {setup_environment()['api_keys']}")
