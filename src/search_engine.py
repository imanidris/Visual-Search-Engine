# The Search Engine Main Module 

"""
Core search engine using FAISS for similarity search.
Supports text-to-image and image-to-image search.
"""

import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image

from .models.clip_encoder import get_model


class SearchEngine:
    """Semantic search engine for images and videos."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize search engine with pre-built index.
        
        Args:
            data_dir: Directory containing index and metadata files
        """
        self.data_dir = Path(data_dir)
        self.model = get_model()
        
        # Load FAISS index
        index_path = self.data_dir / "faiss_index.bin"
        if index_path.exists():
            print("Loading FAISS index...")
            self.index = faiss.read_index(str(index_path))
            print(f"Index loaded with {self.index.ntotal} items")
        else:
            print("No index found. Run build_index.py first.")
            self.index = None
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> List[Dict]:
        """Load combined metadata from images and videos."""
        metadata = []
        
        # Load images
        images_path = self.data_dir / "images.json"
        if images_path.exists():
            with open(images_path) as f:
                images = json.load(f)
                for img in images:
                    img['type'] = 'image'
                metadata.extend(images)
        
        # Load videos
        videos_path = self.data_dir / "videos.json"
        if videos_path.exists():
            with open(videos_path) as f:
                videos = json.load(f)
                for vid in videos:
                    vid['type'] = 'video'
                metadata.extend(videos)
        
        print(f"Loaded metadata for {len(metadata)} items")
        return metadata
    
    def search_by_text(self, query: str, k: int = 12, 
                       media_type: Optional[str] = None) -> List[Dict]:
        """
        Search for images/videos matching a text description.
        
        Args:
            query: Natural language description
            k: Number of results to return
            media_type: Filter by 'image' or 'video' (None for both)
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode_text(query)
        
        # Search
        return self._search(query_embedding, k, media_type)
    
    def search_by_image(self, image: Union[Image.Image, str], k: int = 12,
                        media_type: Optional[str] = None) -> List[Dict]:
        """
        Search for similar images/videos given a reference image.
        
        Args:
            image: PIL Image, URL, or file path
            k: Number of results to return
            media_type: Filter by 'image' or 'video' (None for both)
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        if self.index is None:
            return []
        
        # Encode image
        image_embedding = self.model.encode_image(image)
        
        # Search
        return self._search(image_embedding, k, media_type)
    
    def _search(self, embedding: np.ndarray, k: int, 
                media_type: Optional[str]) -> List[Dict]:
        """
        Internal search function.
        
        Args:
            embedding: Query embedding vector
            k: Number of results
            media_type: Optional filter
            
        Returns:
            List of results with metadata and similarity scores
        """
        # Fetch more results if filtering
        fetch_k = k * 3 if media_type else k
        
        # FAISS search
        distances, indices = self.index.search(
            embedding.astype('float32'), 
            min(fetch_k, self.index.ntotal)
        )
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            item = self.metadata[idx].copy()
            
            # Filter by type if specified
            if media_type and item.get('type') != media_type:
                continue
            
            # Convert distance to similarity score (0-1, higher is better)
            # FAISS returns L2 distance, convert to cosine similarity approximation
            item['score'] = float(1 / (1 + dist))
            results.append(item)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_hybrid(self, text_query: str = None, 
                      image_query: Union[Image.Image, str] = None,
                      k: int = 12, 
                      text_weight: float = 0.5) -> List[Dict]:
        """
        Combined text and image search.
        
        Args:
            text_query: Text description
            image_query: Reference image
            k: Number of results
            text_weight: Weight for text query (0-1), image gets (1 - text_weight)
            
        Returns:
            List of results combining both queries
        """
        if text_query is None and image_query is None:
            return []
        
        if self.index is None:
            return []
        
        # Get embeddings
        combined_embedding = np.zeros((1, self.model.embedding_dim), dtype='float32')
        
        if text_query:
            text_emb = self.model.encode_text(text_query)
            combined_embedding += text_weight * text_emb
        
        if image_query:
            image_emb = self.model.encode_image(image_query)
            combined_embedding += (1 - text_weight) * image_emb
        
        # Normalize
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        return self._search(combined_embedding, k, None)


def build_index(embeddings: np.ndarray, save_path: str = "data/faiss_index.bin"):
    """
    Build and save a FAISS index from embeddings.
    
    Args:
        embeddings: numpy array of shape (n, dim)
        save_path: Where to save the index
    """
    print(f"Building FAISS index for {len(embeddings)} items...")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index (Inner Product after normalization = Cosine Similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    # Add embeddings
    index.add(embeddings.astype('float32'))
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, save_path)
    print(f"Index saved to {save_path}")
    
    return index
