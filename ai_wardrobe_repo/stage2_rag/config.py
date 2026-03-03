"""
Configuration for the Hybrid Graph RAG system.
"""

from pydantic import BaseModel, Field
from pathlib import Path


class VectorStoreConfig(BaseModel):
    """ChromaDB vector store configuration."""
    persist_dir: str = str(Path(__file__).resolve().parent.parent / "wardrobe_db")
    collection_name: str = "wardrobe_items"
    distance_metric: str = "cosine"  # cosine, l2, ip


class GraphStoreConfig(BaseModel):
    """Graph store configuration."""
    # For lightweight mode (NetworkX) — no server needed
    persist_path: str = str(Path(__file__).resolve().parent.parent / "wardrobe_db" / "graph.json")


class OutfitGeneratorConfig(BaseModel):
    """Outfit generation configuration."""
    model_name: str = "gemini-2.5-flash"
    api_key: str = "AIzaSyC5rQc4snfwmntu9p5piYdf7D2iYkhENKI"
    temperature: float = 0.7
    max_output_tokens: int = 4000
    min_outfit_items: int = 2
    max_outfit_items: int = 5


class RAGConfig(BaseModel):
    """Master config for the RAG system."""
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    outfit_generator: OutfitGeneratorConfig = Field(default_factory=OutfitGeneratorConfig)
    top_k_retrieval: int = 10


DEFAULT_RAG_CONFIG = RAGConfig()
