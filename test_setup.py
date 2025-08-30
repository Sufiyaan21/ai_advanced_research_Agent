#!/usr/bin/env python3
"""
Test script to verify the Multi-Modal Research Assistant setup
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'torch',
        'transformers',
        'aiohttp',
        'duckduckgo_search',
        'youtube_transcript_api',
        'yt_dlp',
        'whisper',
        'cv2',
        'pytesseract',
        'PIL',
        'requests',
        'wikipedia',
        'praw',
        'tenacity',
        'spacy',
        'numpy',
        'pandas'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("💡 Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required modules imported successfully!")
        return True

def test_config():
    """Test configuration system"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import get_config, setup_environment
        config = get_config()
        
        print(f"  ✅ Configuration loaded")
        print(f"  📁 Temp directory: {config.model_config.temp_dir}")
        print(f"  📁 Upload directory: {config.model_config.upload_dir}")
        
        # Test API key validation
        api_status = config.api_config.validate_required_keys()
        print(f"  🔑 API Keys Status:")
        for service, available in api_status.items():
            status = "✅" if available else "❌"
            print(f"    {status} {service}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_spacy_model():
    """Test spaCy model"""
    print("\n🧠 Testing spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test with sample text
        doc = nlp("This is a test sentence for spaCy.")
        entities = [ent.text for ent in doc.ents]
        
        print(f"  ✅ spaCy model loaded successfully")
        print(f"  📝 Test entities: {entities}")
        return True
        
    except OSError:
        print(f"  ❌ spaCy model not found")
        print(f"  💡 Install with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"  ❌ spaCy error: {e}")
        return False

def test_whisper():
    """Test Whisper model loading"""
    print("\n🎤 Testing Whisper model...")
    
    try:
        import whisper
        model = whisper.load_model("base")
        print(f"  ✅ Whisper model loaded successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Whisper error: {e}")
        return False

def test_transformers():
    """Test Transformers models"""
    print("\n🤖 Testing Transformers models...")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
        
        # Test BLIP model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print(f"  ✅ BLIP model loaded successfully")
        
        # Test summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print(f"  ✅ BART summarization model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Transformers error: {e}")
        return False

def test_research_assistant():
    """Test research assistant initialization"""
    print("\n🔬 Testing Research Assistant...")
    
    try:
        from research import AgenticResearchAssistant
        
        assistant = AgenticResearchAssistant()
        print(f"  ✅ Research Assistant initialized successfully")
        print(f"  📊 Current topic: {assistant.current_topic}")
        print(f"  📚 Sources count: {len(assistant.all_sources)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Research Assistant error: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app structure"""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        # Check if streamlit_app.py exists
        app_file = Path("streamlit_app.py")
        if not app_file.exists():
            print(f"  ❌ streamlit_app.py not found")
            return False
        
        print(f"  ✅ streamlit_app.py found")
        
        # Try to import the app (without running it)
        import streamlit_app
        print(f"  ✅ Streamlit app imports successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit app error: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 Multi-Modal Research Assistant - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("spaCy Model Test", test_spacy_model),
        ("Whisper Test", test_whisper),
        ("Transformers Test", test_transformers),
        ("Research Assistant Test", test_research_assistant),
        ("Streamlit App Test", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready to go!")
        print("🚀 Run 'python run_app.py' to start the application")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Most issues can be resolved by:")
        print("   - Installing missing packages: pip install -r requirements.txt")
        print("   - Downloading spaCy model: python -m spacy download en_core_web_sm")
        print("   - Setting up API keys in config.json or .env file")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
