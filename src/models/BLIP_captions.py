#!/usr/bin/env python3

"""
Enrich metadata with BLIP-generated captions.


Usage:
    python scripts/BLIP_captions.py
"""

import json
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")


def generate_caption_blip(image_url: str) -> str:
    """
    Generate caption using BLIP model.
    Requires: pip install transformers
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        print("BLIP requires: pip install transformers")
        return ""
    
    # Load model (cached after first call)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Generate caption
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_length=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error captioning {image_url}: {e}")
        return ""


def get_image_url(item: dict) -> str:
    """Get the best URL for captioning."""
    if 'thumbnail' in item and item['thumbnail']:
        return item['thumbnail']
    elif 'thumb' in item and item['thumb']:
        return item['thumb']
    elif 'url' in item:
        return item['url']
    return ""


def main():
    print("=== Enriching Metadata with BLIP Captions ===")
    print("Note: This can take a long time. Consider running overnight.")
    print("You can skip this step - the search will still work without captions.\n")
    
    # Check for BLIP
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP model loaded!\n")
    except ImportError:
        print("BLIP not available. Install with: pip install transformers")
        return
    
    # Process images
    images_path = DATA_DIR / "images.json"
    if images_path.exists():
        with open(images_path) as f:
            images = json.load(f)
        
        print(f"Processing {len(images)} images...")
        for i, img in enumerate(images):
            if img.get('blip_caption'):
                continue  # Skip if already captioned
            
            url = get_image_url(img)
            if url:
                try:
                    response = requests.get(url, timeout=10)
                    pil_image = Image.open(BytesIO(response.content)).convert('RGB')
                    
                    inputs = processor(pil_image, return_tensors="pt")
                    output = model.generate(**inputs, max_length=50)
                    caption = processor.decode(output[0], skip_special_tokens=True)
                    
                    img['blip_caption'] = caption
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Captioned {i + 1}/{len(images)} images")
                        # Save progress
                        with open(images_path, 'w') as f:
                            json.dump(images, f, indent=2)
                
                except Exception as e:
                    print(f"  Error with image {i}: {e}")
        
        with open(images_path, 'w') as f:
            json.dump(images, f, indent=2)
        print(f"Saved {len(images)} images with captions")
    
    # Process videos (using thumbnails)
    videos_path = DATA_DIR / "videos.json"
    if videos_path.exists():
        with open(videos_path) as f:
            videos = json.load(f)
        
        print(f"\nProcessing {len(videos)} videos...")
        for i, vid in enumerate(videos):
            if vid.get('blip_caption'):
                continue
            
            url = get_image_url(vid)
            if url:
                try:
                    response = requests.get(url, timeout=10)
                    pil_image = Image.open(BytesIO(response.content)).convert('RGB')
                    
                    inputs = processor(pil_image, return_tensors="pt")
                    output = model.generate(**inputs, max_length=50)
                    caption = processor.decode(output[0], skip_special_tokens=True)
                    
                    vid['blip_caption'] = caption
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Captioned {i + 1}/{len(videos)} videos")
                        with open(videos_path, 'w') as f:
                            json.dump(videos, f, indent=2)
                
                except Exception as e:
                    print(f"  Error with video {i}: {e}")
        
        with open(videos_path, 'w') as f:
            json.dump(videos, f, indent=2)
        print(f"Saved {len(videos)} videos with captions")
    
    print("\n=== Enrichment Complete ===")


if __name__ == "__main__":
    main()
