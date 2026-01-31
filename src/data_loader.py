



"""
Data loading utilities for the search engine.
Helper functions for loading/saving data.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


def load_images(data_dir: str = "data") -> List[Dict]:
    """Load image metadata from JSON file."""
    path = Path(data_dir) / "images.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_videos(data_dir: str = "data") -> List[Dict]:
    """Load video metadata from JSON file."""
    path = Path(data_dir) / "videos.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_all_media(data_dir: str = "data") -> List[Dict]:
    """Load all media (images + videos) with type labels."""
    media = []
    
    for img in load_images(data_dir):
        img['type'] = 'image'
        media.append(img)
    
    for vid in load_videos(data_dir):
        vid['type'] = 'video'
        media.append(vid)
    
    return media


def save_images(images: List[Dict], data_dir: str = "data"):
    """Save image metadata to JSON file."""
    path = Path(data_dir) / "images.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(images, f, indent=2)


def save_videos(videos: List[Dict], data_dir: str = "data"):
    """Save video metadata to JSON file."""
    path = Path(data_dir) / "videos.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(videos, f, indent=2)


def get_thumbnail_url(item: Dict) -> Optional[str]:
    """Get the best thumbnail URL for an item."""
    if item.get('type') == 'video':
        # Videos have thumbnail in video_pictures
        pictures = item.get('video_pictures', [])
        if pictures:
            return pictures[0].get('picture')
        return item.get('thumbnail')
    else:
        # Images - prefer smaller size for grid display
        return item.get('thumb') or item.get('url')


def get_preview_url(item: Dict) -> str:
    """Get the preview URL (larger size) for an item."""
    if item.get('type') == 'video':
        # Return video URL for playback
        video_files = item.get('video_files', [])
        # Prefer HD quality
        for vf in video_files:
            if vf.get('quality') == 'hd':
                return vf.get('link')
        # Fallback to first available
        if video_files:
            return video_files[0].get('link')
        return item.get('url', '')
    else:
        # Images - return regular size
        return item.get('url', item.get('thumb', ''))
