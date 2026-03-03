"""
Stage 3: Metadata Extraction using Gemini with Strict JSON Schema
Extracts detailed fashion metadata from segmented images
"""

import os
import json
import base64
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import google.generativeai as genai

from config import GeminiConfig
from Schemas import (
    ColorInfo,
    ContextualInfo,
    FashionItemMetadata,
    MaterialComposition,
    StyleAttributes,
)


class GeminiExtractor:
    """
    Gemini-based metadata extractor for fashion items.
    Uses strict JSON schema to ensure structured, consistent outputs.
    """
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.model = None
        self._configure_api()
    
    def _configure_api(self):
        """Configure Google Gemini API"""
        # Prioritize config key, then env var
        api_key = getattr(self.config, 'api_key', None) or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("✗ Error: Gemini API key not found in config or environment")
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        print("✓ Gemini API configured")
    
    def initialize(self):
        """Initialize the Gemini model"""

        try:
            # Configure generation settings
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "response_mime_type": "application/json",
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config
            )
            
            print(f"✓ Gemini {self.config.model_name} initialized")
            
        except Exception as e:
            print(f"✗ Failed to initialize Gemini: {e}")
            raise
    
    def _create_extraction_prompt(self) -> str:
        """
        Create the system prompt for metadata extraction.
        This guides the model to extract all required fields.
        """
        prompt = f"""You are an expert fashion analyst. Analyze the provided fashion item image and extract detailed metadata.

**CRITICAL**: Respond with ONLY valid JSON. Use EXACTLY these snake_case field names:

{{
  "main_category": "Tops|Bottoms|Outerwear|Footwear|Accessories|Dresses",
  "style_attributes": {{
    "subcategory": "e.g. Oversized Flannel Shirt",
    "style_tags": ["tag1", "tag2"],
    "fit": "Oversized|Slim|Regular|Relaxed|Tailored",
    "length": "Cropped|Regular|Long|Ankle|Full",
    "neckline": "Crew|V-neck|Turtleneck|Collar",
    "pattern": "Solid|Striped|Plaid|Floral|Graphic"
  }},
  "colors": [
    {{"name": "Navy Blue", "hex_code": "#1A1A2E", "percentage": 80}}
  ],
  "materials": {{
    "primary_material": "Cotton|Denim|Wool|Polyester|Leather",
    "secondary_materials": [],
    "weight": "Lightweight|Medium-weight|Heavyweight",
    "texture": "Smooth|Ribbed|Textured|Brushed",
    "finish": "Washed|Raw|Distressed|Coated"
  }},
  "context": {{
    "season_suitability": {json.dumps(self.config.season_tags[:2])},
    "formality_level": "{self.config.formality_levels[0]}",
    "occasions": ["Office", "Casual Outing"],
    "weather_conditions": ["Mild"]
  }},
  "brand": null,
  "condition": "New|Excellent|Good|Fair",
  "special_features": [],
  "care_instructions": null,
  "search_keywords": ["keyword1", "keyword2", "keyword3"],
  "vibe_description": "One sentence describing the aesthetic"
}}

Style tags must be from: {', '.join(self.config.style_tags)}
Seasons must be from: {', '.join(self.config.season_tags)}
Formality must be one of: {', '.join(self.config.formality_levels)}

Respond with ONLY the JSON object using the EXACT field names shown above. No markdown, no explanation."""
        
        return prompt
    
    @staticmethod
    def _normalize_keys(obj):
        """Recursively convert JSON keys from Title Case / camelCase to snake_case."""
        import re
        def to_snake(key):
            # "Main Category" -> "main_category"
            key = key.strip().replace(" ", "_")
            # "styleAttributes" -> "style_attributes"
            key = re.sub(r'([a-z])([A-Z])', r'\1_\2', key)
            return key.lower()
        
        if isinstance(obj, dict):
            return {to_snake(k): GeminiExtractor._normalize_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [GeminiExtractor._normalize_keys(item) for item in obj]
        return obj
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_metadata(
        self,
        image_path: str,
        item_id: str,
        category_hint: Optional[str] = None
    ) -> FashionItemMetadata:
        """
        Extract structured metadata from a fashion item image.
        
        Args:
            image_path: Path to the segmented item image
            item_id: Unique identifier for this item
            category_hint: Optional category from detection (e.g., "jacket")
            
        Returns:
            FashionItemMetadata object
        """
        if self.model is None:
            self.initialize()

        
        # Prepare prompt
        system_prompt = self._create_extraction_prompt()
        
        # Add category hint if available
        user_message = "Analyze this fashion item and extract all metadata."
        if category_hint:
            user_message += f"\n\nDetected category: {category_hint}"
        
        # Load and prepare image
        from PIL import Image
        image = Image.open(image_path)
        
        # Generate response
        try:
            response = self.model.generate_content(
                [system_prompt, image, user_message]
            )
            
            # Parse JSON response
            json_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]
            
            # Parse and normalize keys
            metadata_dict = json.loads(json_text)
            metadata_dict = self._normalize_keys(metadata_dict)
            metadata = FashionItemMetadata(**metadata_dict)
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON response: {e}")
            print(f"Response was: {response.text[:500]}...")
            raise
        except Exception as e:
            print(f"✗ Metadata extraction failed: {e}")
            raise


    
    def extract_with_schema_validation(
        self,
        image_path: str,
        item_id: str,
        category_hint: Optional[str] = None,
        max_retries: int = 3
    ) -> FashionItemMetadata:
        """
        Extract metadata with automatic retry on validation failures.
        
        Args:
            image_path: Path to item image
            item_id: Unique ID
            category_hint: Optional category hint
            max_retries: Maximum retry attempts
            
        Returns:
            Validated FashionItemMetadata
        """
        for attempt in range(max_retries):
            try:
                metadata = self.extract_metadata(image_path, item_id, category_hint)
                print(f"✓ Metadata extracted successfully for {item_id}")
                return metadata
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt == max_retries - 1:
                    print(f"✗ All extraction attempts failed for {item_id}")
                    raise
        
    def batch_extract(
        self,
        image_paths: list,
        item_ids: list,
        category_hints: Optional[list] = None
    ) -> Dict[str, FashionItemMetadata]:
        """
        Extract metadata for multiple items.
        
        Args:
            image_paths: List of image paths
            item_ids: List of corresponding item IDs
            category_hints: Optional list of category hints
            
        Returns:
            Dictionary mapping item_id to FashionItemMetadata
        """
        results = {}
        
        if category_hints is None:
            category_hints = [None] * len(image_paths)
        
        total = len(image_paths)
        
        for idx, (image_path, item_id, hint) in enumerate(zip(image_paths, item_ids, category_hints)):
            print(f"\nExtracting metadata {idx + 1}/{total}: {item_id}")
            
            try:
                metadata = self.extract_with_schema_validation(
                    image_path=image_path,
                    item_id=item_id,
                    category_hint=hint
                )
                results[item_id] = metadata
                
            except Exception as e:
                print(f"✗ Skipping {item_id} due to error: {e}")
                continue
        
        return results
    
    def save_metadata(
        self,
        metadata: FashionItemMetadata,
        output_path: str
    ):
        """Save metadata to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                metadata.model_dump(),
                f,
                indent=2,
                ensure_ascii=False
            )
        
        print(f"✓ Metadata saved to {output_path}")


# Convenience function
def extract_fashion_metadata(
    image_path: str,
    item_id: str,
    category_hint: Optional[str] = None,
    config: Optional[GeminiConfig] = None
) -> FashionItemMetadata:
    """
    Extract metadata from a fashion item image.
    
    Args:
        image_path: Path to the item image
        item_id: Unique identifier
        category_hint: Optional category hint
        config: Optional GeminiConfig
        
    Returns:
        FashionItemMetadata object
    """
    if config is None:
        config = GeminiConfig()
    
    extractor = GeminiExtractor(config)
    return extractor.extract_metadata(image_path, item_id, category_hint)
