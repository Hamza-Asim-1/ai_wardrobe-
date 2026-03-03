"""
Pydantic models for structured fashion metadata extraction
These schemas enforce strict JSON output from the LLM
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class ColorInfo(BaseModel):
    """Detailed color information"""
    name: str = Field(description="Human-readable color name (e.g., 'Navy Blue', 'Charcoal Gray')")
    hex_code: str = Field(description="Hex color code (e.g., '#1A1A2E')", pattern=r"^#[0-9A-Fa-f]{6}$")
    percentage: float = Field(description="Approximate percentage of this color in the garment (0-100)", ge=0, le=100)
    
    @field_validator('hex_code')
    @classmethod
    def validate_hex(cls, v: str) -> str:
        if not v.startswith('#'):
            v = f"#{v}"
        return v.upper()


class MaterialComposition(BaseModel):
    """Material and fabric information"""
    primary_material: str = Field(description="Main fabric type (e.g., 'Cotton', 'Denim', 'Wool', 'Polyester')")
    secondary_materials: List[str] = Field(
        default_factory=list,
        description="Additional materials present"
    )
    weight: Optional[str] = Field(
        None,
        description="Fabric weight classification (e.g., 'Lightweight', 'Medium-weight', 'Heavyweight')"
    )
    texture: Optional[str] = Field(
        None,
        description="Fabric texture (e.g., 'Smooth', 'Ribbed', 'Textured', 'Brushed')"
    )
    finish: Optional[str] = Field(
        None,
        description="Special finish or treatment (e.g., 'Washed', 'Raw', 'Distressed', 'Coated')"
    )


class StyleAttributes(BaseModel):
    """Style and aesthetic information"""
    subcategory: str = Field(
        description="Specific garment type (e.g., 'Oversized Flannel', 'Slim Fit Chinos', 'Chelsea Boots')"
    )
    style_tags: List[str] = Field(
        description="Style aesthetics that apply (e.g., ['Dark Academia', 'Vintage', 'Casual'])",
        min_length=1,
        max_length=5
    )
    fit: Optional[str] = Field(
        None,
        description="Fit type (e.g., 'Oversized', 'Slim', 'Regular', 'Relaxed', 'Tailored')"
    )
    length: Optional[str] = Field(
        None,
        description="Length descriptor (e.g., 'Cropped', 'Regular', 'Long', 'Ankle', 'Full')"
    )
    neckline: Optional[str] = Field(
        None,
        description="Neckline type for tops (e.g., 'Crew', 'V-neck', 'Turtleneck', 'Collar')"
    )
    pattern: Optional[str] = Field(
        None,
        description="Pattern or print (e.g., 'Solid', 'Striped', 'Plaid', 'Floral', 'Graphic')"
    )


class ContextualInfo(BaseModel):
    """Contextual usage information"""
    season_suitability: List[str] = Field(
        description="Appropriate seasons (e.g., ['Fall', 'Winter'])",
        min_length=1
    )
    formality_level: str = Field(
        description="Formality classification (e.g., 'Casual', 'Smart Casual', 'Business Formal')"
    )
    occasions: List[str] = Field(
        description="Suitable occasions (e.g., ['Office', 'Date Night', 'Outdoor Activities'])",
        min_length=1,
        max_length=5
    )
    weather_conditions: Optional[List[str]] = Field(
        None,
        description="Ideal weather (e.g., ['Rainy', 'Cold', 'Mild'])"
    )


class FashionItemMetadata(BaseModel):
    """Complete metadata schema for a fashion item"""
    # Core classification
    main_category: str = Field(
        description="Broad category (e.g., 'Tops', 'Bottoms', 'Outerwear', 'Footwear', 'Accessories')"
    )
    style_attributes: StyleAttributes
    
    # Physical properties
    colors: List[ColorInfo] = Field(
        description="Color palette of the item",
        min_length=1,
        max_length=5
    )
    materials: MaterialComposition
    
    # Contextual information
    context: ContextualInfo
    
    # Additional details
    brand: Optional[str] = Field(None, description="Brand name if visible or identifiable")
    condition: Optional[str] = Field(
        None,
        description="Condition assessment (e.g., 'New', 'Excellent', 'Good', 'Fair')"
    )
    special_features: List[str] = Field(
        default_factory=list,
        description="Unique features (e.g., 'Pockets', 'Zipper details', 'Logo placement')"
    )
    care_instructions: Optional[str] = Field(
        None,
        description="Recommended care (e.g., 'Machine Wash Cold', 'Dry Clean Only')"
    )
    
    # Search-friendly fields
    search_keywords: List[str] = Field(
        description="Keywords for text search (auto-generated from other fields)",
        min_length=3
    )
    vibe_description: str = Field(
        description="A one-sentence description of the overall aesthetic and feeling of this item"
    )


class DetectionResult(BaseModel):
    """Result from YOLO-World detection"""
    item_id: str = Field(description="Unique identifier for this detection")
    category: str = Field(description="Detected category label")
    confidence: float = Field(description="Detection confidence score", ge=0, le=1)
    bounding_box: Dict[str, float] = Field(
        description="Bounding box coordinates {x_min, y_min, x_max, y_max}"
    )


class SegmentationResult(BaseModel):
    """Result from SAM 2 segmentation"""
    item_id: str = Field(description="Unique identifier matching detection")
    segmented_image_path: str = Field(description="Path to the segmented PNG with transparency")
    mask_area: int = Field(description="Number of pixels in the segmented mask")
    success: bool = Field(description="Whether segmentation was successful")


class EmbeddingResult(BaseModel):
    """Result from Fashion-CLIP embedding"""
    item_id: str = Field(description="Unique identifier")
    embedding_vector: List[float] = Field(description="Fashion-CLIP embedding vector")
    embedding_dimension: int = Field(description="Dimensionality of the embedding")


class WardrobeItem(BaseModel):
    """Complete wardrobe item with all pipeline outputs"""
    item_id: str
    original_image_path: str
    detection: DetectionResult
    segmentation: SegmentationResult
    metadata: FashionItemMetadata
    embedding: EmbeddingResult
    processing_timestamp: str


class SearchQuery(BaseModel):
    """Schema for semantic search queries"""
    query_text: str = Field(description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filter_category: Optional[str] = Field(None, description="Filter by main category")
    filter_style: Optional[List[str]] = Field(None, description="Filter by style tags")
    filter_season: Optional[List[str]] = Field(None, description="Filter by season")
    filter_formality: Optional[str] = Field(None, description="Filter by formality level")


class SearchResult(BaseModel):
    """Result from semantic search"""
    item_id: str
    similarity_score: float = Field(ge=0, le=1)
    item_metadata: FashionItemMetadata
    image_path: str