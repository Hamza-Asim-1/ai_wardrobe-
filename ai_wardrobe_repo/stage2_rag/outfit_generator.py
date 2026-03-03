"""
Outfit Generator: Uses Gemini LLM to compose coherent outfits
from retrieved wardrobe item candidates.
"""

import json
from typing import List, Dict, Optional

import google.generativeai as genai

from .config import OutfitGeneratorConfig


class OutfitGenerator:
    """LLM-powered outfit composition from retrieved candidates."""

    def __init__(self, config: OutfitGeneratorConfig = None):
        self.config = config or OutfitGeneratorConfig()
        self.model = None
        self._initialize()

    def _initialize(self):
        """Set up Gemini model."""
        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            },
        )
        print(f"✓ Outfit Generator initialized ({self.config.model_name})")

    def extract_intent(self, user_query: str) -> Dict:
        """
        Use LLM to extract structured intent from natural language query.
        E.g., "I want something casual for a date" →
              {"style": "Casual", "occasion": "Date Night", "formality": "Casual"}
        """
        prompt = f"""Extract the fashion intent from this user query. Respond with ONLY valid JSON.

User query: "{user_query}"

Extract these fields (use null if not mentioned):
{{
  "style": "one of: Casual, Streetwear, Dark Academia, Minimalism, Bohemian, Vintage, Modern, Athleisure, Preppy, Grunge, Y2K, Cottagecore, Techwear",
  "occasion": "e.g. Date Night, Office, Casual Outing, Workout, Party, Everyday Wear",
  "formality": "one of: Casual, Smart Casual, Semi-Formal, Formal",
  "season": "one of: Spring, Summer, Fall, Winter, All-Season",
  "color_preference": "any color preference mentioned, or null",
  "category": "specific category if mentioned: Tops, Bottoms, Outerwear, Footwear, Accessories, Dresses, or null",
  "mood": "overall mood/vibe: relaxed, edgy, elegant, sporty, etc."
}}"""

        response = self.model.generate_content(prompt)
        text = response.text.strip()

        # Clean markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text)

    def generate_outfit(
        self,
        user_query: str,
        candidate_items: List[Dict],
        intent: Dict,
    ) -> Dict:
        """
        Use LLM to compose a coherent outfit from candidate items.

        Args:
            user_query: Original user prompt
            candidate_items: List of item dicts with metadata
            intent: Extracted user intent

        Returns:
            Outfit recommendation dict
        """
        # Format items for the prompt
        items_description = []
        for item in candidate_items:
            meta = item.get("metadata", {})
            # Handle both flat (ChromaDB) and nested metadata formats
            colors = meta.get("colors", "[]")
            if isinstance(colors, str):
                colors = json.loads(colors)
            style_tags = meta.get("style_tags", "[]")
            if isinstance(style_tags, str):
                style_tags = json.loads(style_tags)

            items_description.append({
                "item_id": item.get("item_id", item.get("id", "")),
                "category": meta.get("main_category", ""),
                "subcategory": meta.get("subcategory", ""),
                "colors": colors,
                "style_tags": style_tags,
                "formality": meta.get("formality_level", meta.get("formality", "")),
                "pattern": meta.get("pattern", ""),
                "material": meta.get("primary_material", meta.get("material", "")),
                "vibe": meta.get("vibe_description", ""),
                "image_path": meta.get("image_path", ""),
            })

        prompt = f"""You are an expert fashion stylist. Create an outfit recommendation.

USER REQUEST: "{user_query}"

EXTRACTED INTENT:
{json.dumps(intent, indent=2)}

AVAILABLE WARDROBE ITEMS:
{json.dumps(items_description, indent=2)}

RULES:
1. Select {self.config.min_outfit_items}-{self.config.max_outfit_items} items that form a cohesive outfit
2. A complete outfit MUST include items from different categories (e.g., Tops + Bottoms)
3. Items should match in style, formality, and color harmony
4. Match the user's requested occasion/mood/style
5. If there aren't enough items for a full outfit, recommend what's available and note what's missing

Respond with ONLY valid JSON:
{{
  "outfit_name": "a creative descriptive name for this outfit",
  "selected_items": [
    {{
      "item_id": "the item's id",
      "category": "item category",
      "subcategory": "item subcategory",
      "role": "what role this plays in the outfit (e.g., 'base layer', 'bottom', 'statement piece')"
    }}
  ],
  "styling_notes": "2-3 sentences explaining why these items work together",
  "color_story": "brief description of the color palette and harmony",
  "missing_pieces": ["any categories that would complete the outfit but aren't in the wardrobe"],
  "confidence": 0.0-1.0
}}"""

        response = self.model.generate_content(prompt)
        text = response.text.strip()

        # Clean markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text)
