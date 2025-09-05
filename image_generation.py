"""
Stable Diffusion image generation utilities via diffusers.

Generates:
- Conceptual illustration based on summary
- Diagram/infographic style visual (prompt engineered)
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

try:
    import torch
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from diffusers import DDIMScheduler
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers library loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import diffusers: {e}")
    DIFFUSERS_AVAILABLE = False


@dataclass
class GeneratedImage:
    path: str
    prompt: str


class StableDiffusionGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        self.model_id = model_id
        
        # Device selection with better logic
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"  # For Apple Silicon Macs
            print("🍎 Using Apple Metal Performance Shaders (MPS)")
        else:
            self.device = "cpu"
            print("⚠️  Using CPU (this will be slow)")
        
        self._pipe: Optional[StableDiffusionPipeline] = None

    def _get_pipeline(self) -> Optional[StableDiffusionPipeline]:
        """Load and cache the pipeline"""
        if not DIFFUSERS_AVAILABLE:
            print("❌ Diffusers not available")
            return None
            
        if self._pipe is None:
            try:
                print(f"📥 Loading Stable Diffusion pipeline: {self.model_id}")
                print("This may take a few minutes on first run...")
                
                # Configure based on device
                if self.device == "cuda":
                    # Optimized for CUDA
                    self._pipe = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,  # Use half precision for memory efficiency
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_auth_token=False  # Set to True if using gated models
                    )
                    self._pipe = self._pipe.to(self.device)
                    
                    # Enable memory efficient attention if available
                    if hasattr(self._pipe, 'enable_attention_slicing'):
                        self._pipe.enable_attention_slicing()
                    if hasattr(self._pipe, 'enable_memory_efficient_attention'):
                        try:
                            self._pipe.enable_memory_efficient_attention()
                        except:
                            pass  # Not all versions support this
                            
                elif self.device == "mps":
                    # Optimized for Apple Silicon
                    self._pipe = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,  # MPS works better with float32
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    self._pipe = self._pipe.to(self.device)
                    
                else:  # CPU
                    self._pipe = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                
                print("✅ Pipeline loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load pipeline: {e}")
                logger.error(f"Pipeline loading failed: {e}")
                return None
                
        return self._pipe

    def test_generation(self) -> bool:
        """Test if image generation is working"""
        print("🧪 Testing image generation...")
        pipe = self._get_pipeline()
        if pipe is None:
            return False
        
        try:
            # Simple test prompt
            test_prompt = "a red apple on a white background"
            print(f"Testing with prompt: '{test_prompt}'")
            
            # Generate with minimal parameters
            with torch.no_grad():
                result = pipe(
                    test_prompt,
                    num_inference_steps=10,  # Reduced for testing
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                
            if result and result.images:
                print("✅ Test generation successful!")
                return True
            else:
                print("❌ Test generation failed - no image returned")
                return False
                
        except Exception as e:
            print(f"❌ Test generation failed with error: {e}")
            logger.error(f"Test generation error: {e}")
            return False

    def generate(self, prompt: str, out_dir: Path, file_prefix: str, 
                 num_steps: int = 28, guidance: float = 7.5) -> Optional[GeneratedImage]:
        """Generate a single image"""
        
        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers not available; skipping image generation")
            return None
            
        pipe = self._get_pipeline()
        if pipe is None:
            return None
            
        try:
            print(f"🎨 Generating image: {file_prefix}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Device: {self.device}")
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Generate image with error handling
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    height=512,
                    width=512
                )
            
            if not result or not result.images:
                print("❌ No image generated")
                return None
                
            image = result.images[0]
            path = out_dir / f"{file_prefix}.png"
            
            # Save image
            image.save(path)
            print(f"✅ Image saved: {path}")
            
            return GeneratedImage(str(path), prompt)
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU out of memory! Try reducing image size or using CPU")
            logger.error("CUDA out of memory during image generation")
            return None
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            logger.error(f"Image generation error: {e}")
            return None

    def generate_for_summary(self, summary_text: str, out_dir: Path, 
                           base_name: str = "image") -> List[GeneratedImage]:
        """Generate multiple images for a summary"""
        results: List[GeneratedImage] = []
        
        if not summary_text or len(summary_text.strip()) == 0:
            print("⚠️  No summary text provided")
            return results

        # Clean and truncate summary for prompts
        clean_summary = summary_text.strip()[:200]
        
        # Enhanced prompts
        prompts = [
            {
                "name": "concept",
                "prompt": f"Digital art, concept illustration, detailed, professional, high quality: {clean_summary}"
            },
            {
                "name": "diagram", 
                "prompt": f"Clean infographic, diagram style, educational, vector art, clear visualization: {clean_summary}"
            }
        ]
        
        print(f"📚 Generating {len(prompts)} images for summary...")
        
        for prompt_data in prompts:
            print(f"\n🎯 Generating {prompt_data['name']} image...")
            img = self.generate(
                prompt_data["prompt"], 
                out_dir, 
                f"{base_name}_{prompt_data['name']}",
                num_steps=20,  # Slightly faster
                guidance=7.5
            )
            if img:
                results.append(img)
                print(f"✅ {prompt_data['name']} image completed")
            else:
                print(f"❌ {prompt_data['name']} image failed")
        
        print(f"\n🏁 Generated {len(results)}/{len(prompts)} images successfully")
        return results

    def cleanup(self):
        """Clean up GPU memory"""
        if self._pipe is not None and self.device == "cuda":
            del self._pipe
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")


# Utility function for testing
def test_image_generation():
    """Test the image generation setup"""
    print("🔍 Testing image generation setup...")
    
    # Check if required packages are available
    if not DIFFUSERS_AVAILABLE:
        print("❌ Diffusers library not available")
        return False
    
    # Test device availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Create generator and test
    generator = StableDiffusionGenerator()
    success = generator.test_generation()
    
    if success:
        print("🎉 Image generation is working!")
    else:
        print("❌ Image generation is not working")
    
    return success


if __name__ == "__main__":
    # Run GPU-focused test when script is executed directly
    print("🚀 RUNNING GPU-OPTIMIZED IMAGE GENERATION TEST")
    print("=" * 50)
    success = test_image_generation()
    if success:
        print("🎯 Ready for GPU-accelerated image generation!")
    else:
        print("⚠️  Consider running with CPU fallback: test_image_generation(force_gpu=False)")