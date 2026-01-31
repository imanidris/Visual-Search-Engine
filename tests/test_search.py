# testing search functionality 


"""
Basic tests for the search engine.

Usage:
    python -m pytest tests/test_search.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEmbeddings:
    """Test embedding generation."""
    
    def test_model_loads(self):
        """Test that CLIP model loads successfully."""
        from src.embeddings import EmbeddingModel
        model = EmbeddingModel()
        assert model.embedding_dim == 512
    
    def test_text_encoding(self):
        """Test text encoding produces correct shape."""
        from src.embeddings import EmbeddingModel
        model = EmbeddingModel()
        
        embedding = model.encode_text("a beautiful sunset")
        assert embedding.shape == (1, 512)
        
        embeddings = model.encode_text(["sunset", "ocean", "city"])
        assert embeddings.shape == (3, 512)
    
    def test_embeddings_normalized(self):
        """Test that embeddings are roughly unit vectors."""
        from src.embeddings import EmbeddingModel
        model = EmbeddingModel()
        
        embedding = model.encode_text("test query")
        norm = np.linalg.norm(embedding)
        # CLIP embeddings should be normalized (close to 1)
        assert 0.9 < norm < 1.1


class TestSearchEngine:
    """Test search functionality."""
    
    def test_engine_initialization(self):
        """Test search engine can be initialized."""
        from src.search_engine import SearchEngine
        
        # Will print warning if no index, but shouldn't crash
        engine = SearchEngine()
        assert engine is not None


class TestDataLoader:
    """Test data loading utilities."""
    
    def test_get_thumbnail_url(self):
        """Test thumbnail URL extraction."""
        from src.data_loader import get_thumbnail_url
        
        # Image item
        image_item = {
            'type': 'image',
            'thumb': 'https://example.com/thumb.jpg',
            'url': 'https://example.com/full.jpg'
        }
        assert get_thumbnail_url(image_item) == 'https://example.com/thumb.jpg'
        
        # Video item
        video_item = {
            'type': 'video',
            'thumbnail': 'https://example.com/video_thumb.jpg',
            'video_pictures': [{'picture': 'https://example.com/frame.jpg'}]
        }
        assert get_thumbnail_url(video_item) == 'https://example.com/frame.jpg'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
