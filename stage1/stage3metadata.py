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
        self.api_enabled = False
        self._configure_api()
    
    def _configure_api(self):
        """Configure Google Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("⚠ Warning: GEMINI_API_KEY not found in environment")
            print("Running Stage 3 in fallback mode (heuristic metadata).")
        else:
            genai.configure(api_key=api_key)
            self.api_enabled = True
            print("✓ Gemini API configured")
    
    def initialize(self):
        """Initialize the Gemini model"""
        if not self.api_enabled:
            return

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
        prompt = f"""You are an expert fashion analyst and stylist. Analyze the provided fashion item image and extract detailed metadata.

**CRITICAL**: You must respond with ONLY valid JSON matching this exact schema. No additional text, explanations, or markdown.

Extract the following information:

1. **Main Category**: Broad classification (Tops, Bottoms, Outerwear, Footwear, Accessories, Dresses)

2. **Style Attributes**:
   - subcategory: Specific type (e.g., "Oversized Flannel Shirt", "Slim-Fit Chinos", "Chelsea Boots")
   - style_tags: 1-5 aesthetic tags from: {', '.join(self.config.style_tags)}
   - fit: Silhouette (e.g., Oversized, Slim, Regular, Relaxed, Tailored)
   - length: Length descriptor if applicable (Cropped, Regular, Long, Ankle, Full)
   - neckline: For tops (Crew, V-neck, Turtleneck, Collar, etc.)
   - pattern: Pattern type (Solid, Striped, Plaid, Floral, Graphic, etc.)

3. **Colors**: 1-5 dominant colors with:
   - name: Human-readable color name
   - hex_code: Accurate 6-digit hex code (e.g., #1A1A2E)
   - percentage: Approximate percentage of this color (0-100)

4. **Materials**:
   - primary_material: Main fabric (Cotton, Denim, Wool, Polyester, Leather, etc.)
   - secondary_materials: List of additional materials
   - weight: Fabric weight (Lightweight, Medium-weight, Heavyweight)
   - texture: Surface texture (Smooth, Ribbed, Textured, Brushed)
   - finish: Special treatment (Washed, Raw, Distressed, Coated, etc.)

5. **Context**:
   - season_suitability: Appropriate seasons from: {', '.join(self.config.season_tags)}
   - formality_level: One of: {', '.join(self.config.formality_levels)}
   - occasions: 1-5 suitable occasions (Office, Date Night, Casual Outing, etc.)
   - weather_conditions: Ideal weather (Rainy, Cold, Hot, Mild, etc.)

6. **Additional Details**:
   - brand: Brand name if visible/identifiable (or null)
   - condition: Condition assessment (New, Excellent, Good, Fair)
   - special_features: Unique features (Pockets, Zipper details, Logo, etc.)
   - care_instructions: Recommended care (or null)
   - search_keywords: 5-10 searchable keywords
   - vibe_description: One sentence describing the overall aesthetic and feeling

**Important Guidelines**:
- Be specific and detailed in descriptions
- Use precise color names and accurate hex codes
- Consider the actual visual appearance, not assumptions
- For search_keywords, include synonyms and related terms
- The vibe_description should capture the essence and mood of the item

Respond with ONLY the JSON object, nothing else."""
        
        return prompt
    
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

        if not self.api_enabled:
            return self._fallback_metadata(image_path, category_hint)
        
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
            
            # Parse into Pydantic model
            metadata_dict = json.loads(json_text)
            metadata = FashionItemMetadata(**metadata_dict)
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON response: {e}")
            print(f"Response was: {response.text[:500]}...")
            raise
        except Exception as e:
            print(f"✗ Metadata extraction failed: {e}")
            raise

    def _fallback_metadata(
        self,
        image_path: str,
        category_hint: Optional[str],
    ) -> FashionItemMetadata:
        """Generate minimal deterministic metadata when Gemini is unavailable."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        pixels = np.array(image).reshape(-1, 3)
        avg = pixels.mean(axis=0)
        hex_code = "#{:02X}{:02X}{:02X}".format(int(avg[0]), int(avg[1]), int(avg[2]))

        category_map = {
            "shirt": ("Tops", "Shirt"),
            "t-shirt": ("Tops", "T-Shirt"),
            "jacket": ("Outerwear", "Jacket"),
            "coat": ("Outerwear", "Coat"),
            "sweater": ("Tops", "Sweater"),
            "hoodie": ("Tops", "Hoodie"),
            "jeans": ("Bottoms", "Jeans"),
            "pants": ("Bottoms", "Pants"),
            "shorts": ("Bottoms", "Shorts"),
            "skirt": ("Bottoms", "Skirt"),
            "dress": ("Dresses", "Dress"),
            "shoes": ("Footwear", "Shoes"),
            "sneakers": ("Footwear", "Sneakers"),
            "boots": ("Footwear", "Boots"),
            "sandals": ("Footwear", "Sandals"),
            "hat": ("Accessories", "Hat"),
            "cap": ("Accessories", "Cap"),
            "scarf": ("Accessories", "Scarf"),
            "bag": ("Accessories", "Bag"),
            "backpack": ("Accessories", "Backpack"),
            "belt": ("Accessories", "Belt"),
            "watch": ("Accessories", "Watch"),
            "sunglasses": ("Accessories", "Sunglasses"),
        }
        main_category, subcategory = category_map.get(
            (category_hint or "").lower(), ("Accessories", "Fashion Item")
        )
        keyword = (category_hint or "fashion-item").lower().replace(" ", "-")

        return FashionItemMetadata(
            main_category=main_category,
            style_attributes=StyleAttributes(
                subcategory=subcategory,
                style_tags=["Casual"],
                fit="Regular",
                length="Regular",
                neckline=None,
                pattern="Solid",
            ),
            colors=[
                ColorInfo(
                    name="Dominant Tone",
                    hex_code=hex_code,
                    percentage=100.0,
                )
            ],
            materials=MaterialComposition(
                primary_material="Unknown",
                secondary_materials=[],
                weight="Medium-weight",
                texture="Smooth",
                finish=None,
            ),
            context=ContextualInfo(
                season_suitability=["All-Season"],
                formality_level="Casual",
                occasions=["Casual Outing"],
                weather_conditions=["Mild"],
            ),
            brand=None,
            condition="Good",
            special_features=[],
            care_instructions=None,
            search_keywords=[keyword, "wardrobe", "fashion"],
            vibe_description=f"Fallback metadata generated for {subcategory.lower()}.",
        )
    
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
