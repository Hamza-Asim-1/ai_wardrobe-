"""
Vector Store: ChromaDB wrapper for Fashion-CLIP embeddings.
Provides semantic similarity search over wardrobe items.
"""

import json
from typing import List, Dict, Optional, Any
import chromadb

from .config import VectorStoreConfig


class VectorStore:
    """ChromaDB-backed vector store for wardrobe item embeddings."""

    def __init__(self, config: VectorStoreConfig = None):
        self.config = config or VectorStoreConfig()
        self.client = chromadb.PersistentClient(path=self.config.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )
        print(f"✓ ChromaDB initialized at {self.config.persist_dir}")
        print(f"  Collection '{self.config.collection_name}': {self.collection.count()} items")

    def add_item(
        self,
        item_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        image_path: str,
    ):
        """Add or update a single wardrobe item."""
        flat_meta = self._flatten_metadata(metadata, image_path)
        self.collection.upsert(
            ids=[item_id],
            embeddings=[embedding],
            metadatas=[flat_meta],
        )

    def add_items_batch(
        self,
        item_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        image_paths: List[str],
    ):
        """Add multiple items at once."""
        flat_metas = [
            self._flatten_metadata(m, p) for m, p in zip(metadatas, image_paths)
        ]
        self.collection.upsert(
            ids=item_ids,
            embeddings=embeddings,
            metadatas=flat_metas,
        )
        print(f"✓ Upserted {len(item_ids)} items into ChromaDB")

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> Dict:
        """Search by embedding vector."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
        }
        if where_filter:
            kwargs["where"] = where_filter
        return self.collection.query(**kwargs)

    def search_by_metadata(
        self, where_filter: Dict, top_k: int = 10
    ) -> Dict:
        """Filter items by metadata fields."""
        return self.collection.get(where=where_filter, limit=top_k)

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Retrieve a single item by ID."""
        result = self.collection.get(ids=[item_id], include=["metadatas", "embeddings"])
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "metadata": result["metadatas"][0],
                "embedding": result["embeddings"][0] if result["embeddings"] else None,
            }
        return None

    def get_all_items(self) -> Dict:
        """Get all items in the collection."""
        return self.collection.get(include=["metadatas", "embeddings"])

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _flatten_metadata(metadata: Dict[str, Any], image_path: str) -> Dict[str, str]:
        """Flatten nested metadata into ChromaDB-compatible flat dict (strings only)."""
        style = metadata.get("style_attributes", {})
        context = metadata.get("context", {})
        materials = metadata.get("materials", {})
        colors = metadata.get("colors", [])

        return {
            "main_category": metadata.get("main_category", ""),
            "subcategory": style.get("subcategory", ""),
            "style_tags": json.dumps(style.get("style_tags", [])),
            "fit": style.get("fit", ""),
            "pattern": style.get("pattern", ""),
            "formality_level": context.get("formality_level", ""),
            "seasons": json.dumps(context.get("season_suitability", [])),
            "occasions": json.dumps(context.get("occasions", [])),
            "colors": json.dumps([c.get("name", "") for c in colors]),
            "primary_color": colors[0].get("name", "") if colors else "",
            "primary_material": materials.get("primary_material", ""),
            "vibe_description": metadata.get("vibe_description", ""),
            "search_keywords": json.dumps(metadata.get("search_keywords", [])),
            "image_path": image_path,
        }
