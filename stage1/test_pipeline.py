#!/usr/bin/env python3
"""
Quick Test Script for Fashion Wardrobe Pipeline

This script helps you test the pipeline with sample images.
Run with: python test_pipeline.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_UPLOADS_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def _resolve_uploads_dir() -> Path:
    env_dir = os.getenv("UPLOADS_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("/mnt/user-data/uploads")
    return fallback if fallback.exists() else DEFAULT_UPLOADS_DIR


def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check for Gemini API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        issues.append("⚠ GEMINI_API_KEY not set (Stage 3 will use fallback metadata)")
        print("   Set with: export GEMINI_API_KEY='your-key-here' for full Gemini metadata")
    else:
        print("✓ Gemini API key found")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    # Check for uploaded images
    uploads_dir = _resolve_uploads_dir()
    if uploads_dir.exists():
        images = [p for p in uploads_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        if images:
            print(f"✓ Found {len(images)} image(s) in {uploads_dir}")
        else:
            issues.append(f"⚠ No images found in {uploads_dir}")
    else:
        issues.append(f"⚠ Uploads directory not found: {uploads_dir}")
    
    if issues:
        print("\n⚠ Issues detected:")
        for issue in issues:
            print(f"  {issue}")
        
    return True


def download_sample_image():
    """Download a sample fashion image if no images available"""
    print("\n📥 Downloading sample image...")
    
    import urllib.request
    
    # Sample fashion image URL (public domain)
    sample_url = "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=800"
    output_path = str(DEFAULT_OUTPUT_DIR / "sample_outfit.jpg")
    
    try:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(sample_url, output_path)
        print(f"✓ Downloaded sample image to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download sample: {e}")
        return None


def run_quick_test():
    """Run a quick test of the pipeline"""
    print("\n" + "="*60)
    print("RUNNING QUICK TEST")
    print("="*60 + "\n")
    
    from pipeline import FashionWardrobePipeline
    
    # Find test image
    uploads_dir = _resolve_uploads_dir()
    
    if uploads_dir.exists():
        images = [p for p in uploads_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        if images:
            test_image = str(images[0])
        else:
            test_image = download_sample_image()
    else:
        test_image = download_sample_image()
    
    if not test_image:
        print("❌ No image available for testing")
        return
    
    print(f"📸 Testing with: {test_image}\n")
    
    try:
        # Initialize pipeline
        pipeline = FashionWardrobePipeline()
        
        # Process single image
        print("⏳ Processing (this may take a few minutes)...\n")
        items = pipeline.process_image(test_image, visualize=True)
        
        if items:
            print("\n" + "="*60)
            print("✅ TEST SUCCESSFUL!")
            print("="*60 + "\n")
            
            print(f"Found {len(items)} item(s):\n")
            
            for idx, item in enumerate(items, 1):
                meta = item.metadata
                print(f"{idx}. {meta.style_attributes.subcategory}")
                print(f"   Category: {meta.main_category}")
                print(f"   Colors: {', '.join([c.name for c in meta.colors[:3]])}")
                print(f"   Style: {', '.join(meta.style_attributes.style_tags[:3])}")
                print(f"   Vibe: {meta.vibe_description}")
                print(f"   Confidence: {item.detection.confidence:.2%}")
                print()
            
            # Save and test search
            pipeline.save_search_index()
            
            print("\n🔍 Testing semantic search...")
            results = pipeline.search("casual everyday wear", top_k=3)
            
            if results:
                print("\nSearch results for 'casual everyday wear':")
                for idx, result in enumerate(results, 1):
                    print(f"  {idx}. {result.item_metadata.style_attributes.subcategory}")
                    print(f"     Similarity: {result.similarity_score:.3f}")
            
            print("\n" + "="*60)
            print(f"All outputs saved to: {DEFAULT_OUTPUT_DIR}")
            print("="*60)
            
        else:
            print("⚠ No items detected in the image")
            print("Try with a different image or lower the confidence threshold")
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def run_full_demo():
    """Run the full demo with all uploaded images"""
    print("\n" + "="*60)
    print("RUNNING FULL DEMO")
    print("="*60 + "\n")
    
    from pipeline import run_demo_pipeline
    from pathlib import Path
    
    # Get all images
    uploads_dir = _resolve_uploads_dir()
    
    if not uploads_dir.exists():
        print(f"❌ No uploads directory found: {uploads_dir}")
        return
    
    image_paths = [
        str(f) for f in uploads_dir.glob("*")
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
    ]
    
    if not image_paths:
        print("❌ No images found in uploads")
        test_image = download_sample_image()
        if test_image:
            image_paths = [test_image]
        else:
            return
    
    print(f"Found {len(image_paths)} image(s) to process\n")
    
    try:
        # Run full pipeline
        pipeline, all_items = run_demo_pipeline(image_paths)
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETE!")
        print("="*60)
        print(f"\nProcessed {len(all_items)} total items")
        print(f"Outputs saved to: {DEFAULT_OUTPUT_DIR}")
        
        # Interactive search demo
        print("\n🔍 Try some searches:")
        
        queries = [
            "something cozy",
            "formal business wear",
            "casual weekend outfit"
        ]
        
        for query in queries:
            print(f"\n📝 '{query}':")
            results = pipeline.search(query, top_k=3)
            
            for idx, result in enumerate(results, 1):
                print(f"  {idx}. {result.item_metadata.style_attributes.subcategory}")
                print(f"     Score: {result.similarity_score:.3f}")
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()


def interactive_menu():
    """Interactive menu for testing"""
    print("\n" + "🎨 "*20)
    print("FASHION WARDROBE PIPELINE - TEST MENU")
    print("🎨 "*20 + "\n")
    
    if not check_environment():
        print("\n⚠ Please fix environment issues before proceeding")
        return
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Quick test (single image)")
        print("2. Full demo (all uploaded images)")
        print("3. Check environment")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_quick_test()
        elif choice == "2":
            run_full_demo()
        elif choice == "3":
            check_environment()
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    """
    Run the test script
    """
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            check_environment()
            run_quick_test()
        elif sys.argv[1] == "full":
            check_environment()
            run_full_demo()
        elif sys.argv[1] == "check":
            check_environment()
        else:
            print("Usage: python test_pipeline.py [quick|full|check]")
    else:
        # Interactive mode
        interactive_menu()
