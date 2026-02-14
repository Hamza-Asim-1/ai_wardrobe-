"""
Configuration file for the Fashion Wardrobe Pipeline
All parameters are centralized here - no hardcoding in the main pipeline
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class YOLOConfig(BaseModel):
    """Configuration for YOLO-World object detection"""
    model_id: str = "wondervictor/YOLO-World"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    device: str = "cuda"  # or "cpu"
    
    # Fashion-specific categories to detect
    target_categories: List[str] = [
        "shirt", "t-shirt", "jacket", "coat", "sweater", "hoodie",
        "jeans", "pants", "shorts", "skirt", "dress",
        "shoes", "sneakers", "boots", "sandals",
        "hat", "cap", "scarf", "bag", "backpack",
        "belt", "watch", "sunglasses"
    ]


class SAMConfig(BaseModel):
    """Configuration for Segment Anything Model 2"""
    model_type: str = "vit_h"  # Options: vit_h, vit_l, vit_b
    checkpoint_path: str = "sam2_hiera_large.pt"
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    output_format: str = "png"  # PNG with transparency


class GeminiConfig(BaseModel):
    """Configuration for Gemini/LLM metadata extraction"""
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Extraction schema configuration
    extract_subcategory: bool = True
    extract_material: bool = True
    extract_colors: bool = True
    extract_patterns: bool = True
    extract_style_tags: bool = True
    extract_season: bool = True
    extract_formality: bool = True
    
    # Style vocabularies
    style_tags: List[str] = [
        "Dark Academia", "Streetwear", "Minimalism", "Bohemian",
        "Preppy", "Grunge", "Athleisure", "Vintage", "Modern",
        "Casual", "Formal", "Business Casual", "Smart Casual"
    ]
    
    season_tags: List[str] = [
        "Spring", "Summer", "Fall", "Winter", "All-Season"
    ]
    
    formality_levels: List[str] = [
        "Casual", "Smart Casual", "Business Casual", "Business Formal", "Black Tie"
    ]


class FashionCLIPConfig(BaseModel):
    """Configuration for Fashion-CLIP embeddings"""
    model_name: str = "patrickjohncyh/fashion-clip"
    device: str = "cuda"
    embedding_dimension: int = 512
    
    # Vector search configuration
    similarity_metric: str = "cosine"  # Options: cosine, euclidean, dot_product
    top_k_results: int = 10


class PipelineConfig(BaseModel):
    """Master configuration for the entire pipeline"""
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    fashion_clip: FashionCLIPConfig = Field(default_factory=FashionCLIPConfig)
    
    # Output paths
    output_dir: str = str(Path(__file__).resolve().parent / "outputs")
    segmented_images_dir: str = "segmented"
    embeddings_dir: str = "embeddings"
    metadata_dir: str = "metadata"
    
    # Processing options
    batch_size: int = 1
    save_intermediate_results: bool = True
    verbose: bool = True


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
