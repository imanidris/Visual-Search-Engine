# CLIP model used for the visual search engine


"""
Embedding generation using CLIP model.
Handles both text and image encoding for semantic search.
"""

from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import Union, List
import requests
from io import BytesIO


class EmbeddingModel:
    """Wrapper for CLIP model to generate text and image embeddings."""
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the CLIP model.
        
        Args:
            model_name: SentenceTransformer model name. 
                        Options: clip-ViT-B-32, clip-ViT-L-14
        """
        print(f"Loading CLIP model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 512  # CLIP ViT-B-32 dimension
        print("Model loaded successfully!")
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text query into embedding vector(s).
        
        Args:
            text: Single string or list of strings
            
        Returns:
            numpy array of shape (n, 512) for n texts
        """
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_image(self, image: Union[Image.Image, str, List]) -> np.ndarray:
        """
        Encode image(s) into embedding vector(s).
        
        Args:
            image: PIL Image, URL string, file path, or list of these
            
        Returns:
            numpy array of shape (n, 512) for n images
        """
        if not isinstance(image, list):
            image = [image]
        
        pil_images = []
        for img in image:
            if isinstance(img, str):
                if img.startswith(('http://', 'https://')):
                    # Load from URL
                    response = requests.get(img, timeout=10)
                    pil_images.append(Image.open(BytesIO(response.content)))
                else:
                    # Load from file path
                    pil_images.append(Image.open(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        return self.model.encode(pil_images, convert_to_numpy=True)
    
    def encode_batch(self, images: List, batch_size: int = 32, 
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode a large batch of images efficiently.
        
        Args:
            images: List of PIL Images or URLs
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n, 512)
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Convert URLs to PIL Images
            pil_batch = []
            for img in batch:
                if isinstance(img, str) and img.startswith(('http://', 'https://')):
                    try:
                        response = requests.get(img, timeout=10)
                        pil_batch.append(Image.open(BytesIO(response.content)))
                    except Exception as e:
                        print(f"Error loading {img}: {e}")
                        # Use a blank image as placeholder
                        pil_batch.append(Image.new('RGB', (224, 224), color='gray'))
                elif isinstance(img, Image.Image):
                    pil_batch.append(img)
                else:
                    pil_batch.append(Image.open(img))
            
            embeddings = self.model.encode(pil_batch, convert_to_numpy=True)
            all_embeddings.append(embeddings)
            
            if show_progress:
                print(f"Processed {min(i + batch_size, len(images))}/{len(images)} images")
        
        return np.vstack(all_embeddings)


# Singleton instance for reuse
_model_instance = None

def get_model() -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance
