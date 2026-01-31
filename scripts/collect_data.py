#!/usr/bin/env python3
"""
Pexels API Fetcher
Purpose: Download image/video metadata from Pexels API.
Run this once to build the metadata dataset.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --images-only
    python scripts/collect_data.py --videos-only
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("PEXELS_API_KEY")
HEADERS = {"Authorization": API_KEY}

# VFX-relevant search queries
SEARCH_QUERIES = [
    # Environments
    "cinematic landscape",
    "city skyline night",
    "aerial view mountains",
    "desert sunset",
    "ocean waves dramatic",
    "forest moody fog",
    "urban street rain",
    "futuristic city",
    "abandoned building",
    "arctic ice",
    
    # Atmosphere & Weather
    "dramatic clouds",
    "storm lightning",
    "fog mist",
    "rain drops",
    "snow falling",
    "dust particles",
    
    # Elements
    "fire flames",
    "explosion",
    "smoke",
    "water splash",
    "sparks",
    "lava",
    
    # Lighting
    "golden hour",
    "neon lights",
    "silhouette sunset",
    "dramatic shadows",
    "backlit",
    "light rays",
    
    # Motion (especially good for video)
    "slow motion",
    "timelapse",
    "drone shot",
    "tracking shot",
]

DATA_DIR = Path("data")


def fetch_images(query: str, per_page: int = 30, pages: int = 3) -> list:
    """Fetch images from Pexels for a search query."""
    images = []
    
    for page in range(1, pages + 1):
        url = "https://api.pexels.com/v1/search"
        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
            "orientation": "landscape"  # Better for VFX reference
        }
        
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for photo in data.get("photos", []):
                images.append({
                    "id": photo["id"],
                    "url": photo["src"]["large"],
                    "thumb": photo["src"]["medium"],
                    "original": photo["src"]["original"],
                    "photographer": photo["photographer"],
                    "photographer_url": photo["photographer_url"],
                    "alt": photo.get("alt", ""),
                    "width": photo["width"],
                    "height": photo["height"],
                    "avg_color": photo.get("avg_color", ""),
                    "query": query,
                    "source": "pexels"
                })
            
            print(f"  Images: {query} page {page} - got {len(data.get('photos', []))} results")
            
        except Exception as e:
            print(f"  Error fetching images for '{query}' page {page}: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return images


def fetch_videos(query: str, per_page: int = 15, pages: int = 2) -> list:
    """Fetch videos from Pexels for a search query."""
    videos = []
    
    for page in range(1, pages + 1):
        url = "https://api.pexels.com/videos/search"
        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
            "orientation": "landscape"
        }
        
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for video in data.get("videos", []):
                # Get thumbnail from video pictures
                thumbnail = ""
                if video.get("video_pictures"):
                    thumbnail = video["video_pictures"][0].get("picture", "")
                
                # Get video file URLs
                video_files = []
                for vf in video.get("video_files", []):
                    video_files.append({
                        "quality": vf.get("quality"),
                        "width": vf.get("width"),
                        "height": vf.get("height"),
                        "link": vf.get("link")
                    })
                
                videos.append({
                    "id": video["id"],
                    "thumbnail": thumbnail,
                    "video_pictures": video.get("video_pictures", []),
                    "video_files": video_files,
                    "duration": video.get("duration"),
                    "width": video.get("width"),
                    "height": video.get("height"),
                    "user": video.get("user", {}).get("name", ""),
                    "user_url": video.get("user", {}).get("url", ""),
                    "url": video.get("url", ""),
                    "query": query,
                    "source": "pexels"
                })
            
            print(f"  Videos: {query} page {page} - got {len(data.get('videos', []))} results")
            
        except Exception as e:
            print(f"  Error fetching videos for '{query}' page {page}: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return videos


def deduplicate(items: list, key: str = "id") -> list:
    """Remove duplicate items based on a key."""
    seen = set()
    unique = []
    for item in items:
        if item[key] not in seen:
            seen.add(item[key])
            unique.append(item)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Collect data from Pexels API")
    parser.add_argument("--images-only", action="store_true", help="Only collect images")
    parser.add_argument("--videos-only", action="store_true", help="Only collect videos")
    parser.add_argument("--queries", type=str, help="Comma-separated custom queries")
    args = parser.parse_args()
    
    if not API_KEY:
        print("Error: PEXELS_API_KEY not found in environment")
        print("Please create a .env file with your API key")
        print("Get a free key at: https://www.pexels.com/api/")
        return
    
    # Use custom queries if provided
    queries = args.queries.split(",") if args.queries else SEARCH_QUERIES
    
    DATA_DIR.mkdir(exist_ok=True)
    
    collect_images = not args.videos_only
    collect_videos = not args.images_only
    
    # Collect images
    if collect_images:
        print("\n=== Collecting Images ===")
        all_images = []
        for query in queries:
            images = fetch_images(query)
            all_images.extend(images)
        
        all_images = deduplicate(all_images)
        print(f"\nTotal unique images: {len(all_images)}")
        
        with open(DATA_DIR / "images.json", "w") as f:
            json.dump(all_images, f, indent=2)
        print(f"Saved to {DATA_DIR}/images.json")
    
    # Collect videos
    if collect_videos:
        print("\n=== Collecting Videos ===")
        all_videos = []
        for query in queries:
            videos = fetch_videos(query)
            all_videos.extend(videos)
        
        all_videos = deduplicate(all_videos)
        print(f"\nTotal unique videos: {len(all_videos)}")
        
        with open(DATA_DIR / "videos.json", "w") as f:
            json.dump(all_videos, f, indent=2)
        print(f"Saved to {DATA_DIR}/videos.json")
    
    print("\n=== Collection Complete ===")
    if collect_images:
        print(f"Images: {len(all_images) if 'all_images' in locals() else 'N/A'}")
    if collect_videos:
        print(f"Videos: {len(all_videos) if 'all_videos' in locals() else 'N/A'}")


if __name__ == "__main__":
    main()
