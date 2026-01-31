#!/usr/bin/env python3
"""
Build FAISS index from collected images and videos.
Generates CLIP embeddings for all items.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --batch-size 16
"""

import json
import argparse
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.clip_encoder import EmbeddingModel
from src.search_engine import build_index

DATA_DIR = Path("data")


def get_image_url(item: dict) -> str:
    """Get the thumbnail/preview URL for embedding generation."""
    if 'thumbnail' in item and item['thumbnail']:
        # Video
        return item['thumbnail']
    elif 'thumb' in item and item['thumb']:
        # Image (medium size is faster to download)
        return item['thumb']
    elif 'url' in item:
        return item['url']
    return ""


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    args = parser.parse_args()
    
    # Load metadata
    all_items = []
    
    images_path = DATA_DIR / "images.json"
    if images_path.exists():
        with open(images_path) as f:
            images = json.load(f)
            for img in images:
                img['type'] = 'image'
            all_items.extend(images)
            print(f"Loaded {len(images)} images")
    
    videos_path = DATA_DIR / "videos.json"
    if videos_path.exists():
        with open(videos_path) as f:
            videos = json.load(f)
            for vid in videos:
                vid['type'] = 'video'
            all_items.extend(videos)
            print(f"Loaded {len(videos)} videos")
    
    if not all_items:
        print("No data found. Run collect_data.py first.")
        return
    
    print(f"\nTotal items to index: {len(all_items)}")
    
    # Get URLs for embedding
    urls = [get_image_url(item) for item in all_items]
    
    # Filter out items without URLs
    valid_indices = [i for i, url in enumerate(urls) if url]
    valid_urls = [urls[i] for i in valid_indices]
    valid_items = [all_items[i] for i in valid_indices]
    
    print(f"Valid items with URLs: {len(valid_items)}")
    
    if len(valid_items) < len(all_items):
        print(f"Warning: {len(all_items) - len(valid_items)} items skipped (no URL)")
    
    # Generate embeddings
    print("\n=== Generating CLIP Embeddings ===")
    model = EmbeddingModel()
    embeddings = model.encode_batch(valid_urls, batch_size=args.batch_size)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    embeddings_path = DATA_DIR / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Update metadata files with only valid items
    # (maintains alignment with embeddings)
    valid_images = [item for item in valid_items if item['type'] == 'image']
    valid_videos = [item for item in valid_items if item['type'] == 'video']
    
    with open(DATA_DIR / "images.json", "w") as f:
        json.dump(valid_images, f, indent=2)
    
    with open(DATA_DIR / "videos.json", "w") as f:
        json.dump(valid_videos, f, indent=2)
    
    print(f"Updated metadata: {len(valid_images)} images, {len(valid_videos)} videos")
    
    # Build FAISS index
    print("\n=== Building FAISS Index ===")
    build_index(embeddings, str(DATA_DIR / "faiss_index.bin"))
    
    print("\n=== Index Build Complete ===")
    print(f"Index contains {len(valid_items)} searchable items")


if __name__ == "__main__":
    main()
