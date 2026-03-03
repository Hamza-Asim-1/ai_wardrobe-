"""
Stage 4: Fashion-CLIP Embeddings and Semantic Search
Creates visual embeddings for semantic similarity search
"""

import os
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json
import hashlib

from config import FashionCLIPConfig
from Schemas import EmbeddingResult, SearchQuery, SearchResult, FashionItemMetadata


class FashionCLIPEncoder:
    """
    Fashion-CLIP encoder for creating semantic embeddings.
    Enables natural language search over fashion items.
    """
    
    def __init__(self, config: FashionCLIPConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    def initialize(self):
        """Load Fashion-CLIP model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print(f"Loading Fashion-CLIP model on {self.device}...")
            
            self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
            
            self.model.eval()  # Set to evaluation mode
            
            print(f"✓ Fashion-CLIP initialized (embedding dim: {self.config.embedding_dimension})")
            
        except Exception as e:
            print(f"✗ Fashion-CLIP initialization failed: {e}")
            raise
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Create embedding vector from an image.
        
        Args:
            image_path: Path to the fashion item image
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            self.initialize()
        
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize the embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy()[0]
        
        return embedding
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Create embedding vector from text query.
        
        Args:
            text: Natural language query
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            self.initialize()
        
        # Process text
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = text_features.cpu().numpy()[0]
        
        return embedding


    
    def create_embedding_result(
        self,
        image_path: str,
        item_id: str
    ) -> EmbeddingResult:
        """
        Create EmbeddingResult for an item.
        
        Args:
            image_path: Path to item image
            item_id: Unique identifier
            
        Returns:
            EmbeddingResult object
        """
        embedding = self.encode_image(image_path)
        
        return EmbeddingResult(
            item_id=item_id,
            embedding_vector=embedding.tolist(),
            embedding_dimension=len(embedding)
        )
    
    def batch_encode_images(
        self,
        image_paths: List[str],
        item_ids: List[str]
    ) -> Dict[str, EmbeddingResult]:
        """
        Encode multiple images.
        
        Args:
            image_paths: List of image paths
            item_ids: Corresponding item IDs
            
        Returns:
            Dictionary mapping item_id to EmbeddingResult
        """
        results = {}
        
        total = len(image_paths)
        
        for idx, (image_path, item_id) in enumerate(zip(image_paths, item_ids)):
            print(f"Encoding {idx + 1}/{total}: {item_id}")
            
            try:
                result = self.create_embedding_result(image_path, item_id)
                results[item_id] = result
            except Exception as e:
                print(f"✗ Failed to encode {item_id}: {e}")
                continue
        
        return results
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Formula: cos(θ) = (A · B) / (||A|| ||B||)
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        similarity = dot_product / (norm1 * norm2)
        # Map cosine from [-1, 1] to [0, 1] for schema compatibility.
        similarity = (similarity + 1.0) / 2.0

        return float(similarity)
    
    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        return float(np.linalg.norm(vec1 - vec2))
    
    @staticmethod
    def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate dot product similarity"""
        return float(np.dot(vec1, vec2))


class SemanticSearchEngine:
    """
    Semantic search engine for fashion items using embeddings.
    """
    
    def __init__(
        self,
        encoder: FashionCLIPEncoder,
        config: Optional[FashionCLIPConfig] = None
    ):
        self.encoder = encoder
        self.config = config or FashionCLIPConfig()
        
        # Storage for embeddings and metadata
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, FashionItemMetadata] = {}
        self.image_paths: Dict[str, str] = {}
    
    def add_item(
        self,
        item_id: str,
        embedding: np.ndarray,
        metadata: FashionItemMetadata,
        image_path: str
    ):
        """Add an item to the search index"""
        self.embeddings[item_id] = embedding
        self.metadata[item_id] = metadata
        self.image_paths[item_id] = image_path
    
    def add_items_batch(
        self,
        items: Dict[str, Dict]
    ):
        """
        Add multiple items to the search index.
        
        Args:
            items: Dict mapping item_id to dict with 'embedding', 'metadata', 'image_path'
        """
        for item_id, data in items.items():
            self.add_item(
                item_id=item_id,
                embedding=np.array(data['embedding']),
                metadata=data['metadata'],
                image_path=data['image_path']
            )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_category: Optional[str] = None,
        filter_style: Optional[List[str]] = None,
        filter_season: Optional[List[str]] = None,
        filter_formality: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for fashion items using natural language.
        
        Args:
            query: Natural language search query (e.g., "cozy sweater for rainy days")
            top_k: Number of results to return
            filter_category: Filter by main category
            filter_style: Filter by style tags
            filter_season: Filter by season
            filter_formality: Filter by formality level
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Encode the query
        query_embedding = self.encoder.encode_text(query)
        
        # Calculate similarities
        results = []
        
        for item_id, item_embedding in self.embeddings.items():
            item_metadata = self.metadata[item_id]
            
            # Apply filters
            if filter_category and item_metadata.main_category != filter_category:
                continue
            
            if filter_style:
                if not any(style in item_metadata.style_attributes.style_tags for style in filter_style):
                    continue
            
            if filter_season:
                if not any(season in item_metadata.context.season_suitability for season in filter_season):
                    continue
            
            if filter_formality and item_metadata.context.formality_level != filter_formality:
                continue
            
            # Calculate similarity
            if self.config.similarity_metric == "cosine":
                similarity = self.encoder.cosine_similarity(query_embedding, item_embedding)
            elif self.config.similarity_metric == "dot_product":
                similarity = self.encoder.dot_product(query_embedding, item_embedding)
            else:  # euclidean
                # Convert distance to similarity (inverse)
                distance = self.encoder.euclidean_distance(query_embedding, item_embedding)
                similarity = 1.0 / (1.0 + distance)
            
            result = SearchResult(
                item_id=item_id,
                similarity_score=similarity,
                item_metadata=item_metadata,
                image_path=self.image_paths[item_id]
            )
            
            results.append(result)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Return top-k
        return results[:top_k]
    
    def save_index(self, output_dir: str):
        """
        Save the search index to disk.
        
        Args:
            output_dir: Directory to save index files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_file = os.path.join(output_dir, "embeddings.npz")
        np.savez(
            embeddings_file,
            **{item_id: emb for item_id, emb in self.embeddings.items()}
        )
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.json")
        metadata_dict = {
            item_id: meta.model_dump() 
            for item_id, meta in self.metadata.items()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Save image paths
        paths_file = os.path.join(output_dir, "image_paths.json")
        with open(paths_file, 'w') as f:
            json.dump(self.image_paths, f, indent=2)
        
        print(f"✓ Search index saved to {output_dir}")
    
    def load_index(self, index_dir: str):
        """
        Load search index from disk.
        
        Args:
            index_dir: Directory containing index files
        """
        # Load embeddings
        embeddings_file = os.path.join(index_dir, "embeddings.npz")
        embeddings_data = np.load(embeddings_file)
        self.embeddings = {key: embeddings_data[key] for key in embeddings_data.files}
        
        # Load metadata
        metadata_file = os.path.join(index_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        self.metadata = {
            item_id: FashionItemMetadata(**meta_dict)
            for item_id, meta_dict in metadata_dict.items()
        }
        
        # Load image paths
        paths_file = os.path.join(index_dir, "image_paths.json")
        with open(paths_file, 'r') as f:
            self.image_paths = json.load(f)
        
        print(f"✓ Loaded {len(self.embeddings)} items from index")


# Convenience functions
def create_embedding(
    image_path: str,
    item_id: str,
    config: Optional[FashionCLIPConfig] = None
) -> EmbeddingResult:
    """Create embedding for a single image"""
    if config is None:
        config = FashionCLIPConfig()
    
    encoder = FashionCLIPEncoder(config)
    return encoder.create_embedding_result(image_path, item_id)


def search_wardrobe(
    query: str,
    search_engine: SemanticSearchEngine,
    top_k: int = 10
) -> List[SearchResult]:
    """Perform semantic search on wardrobe"""
    return search_engine.search(query, top_k=top_k)
