"""
Ingestion Pipeline: Reads Stage 1-4 outputs and stores them in Vector + Graph databases.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .config import RAGConfig, DEFAULT_RAG_CONFIG
from .vector_store import VectorStore
from .graph_store import GraphStore


def ingest_from_outputs(
    outputs_dir: str,
    config: RAGConfig = None,
) -> tuple:
    """
    Read Stage 3 metadata + Stage 4 embeddings from pipeline outputs
    and populate both ChromaDB and the knowledge graph.

    Args:
        outputs_dir: Path to the pipeline outputs directory (e.g., outputs_test/)
        config: Optional RAG configuration

    Returns:
        (vector_store, graph_store) tuple
    """
    config = config or DEFAULT_RAG_CONFIG
    outputs = Path(outputs_dir)
    stage3_dir = outputs / "stage3"
    stage4_dir = outputs / "stage4"
    stage2_dir = outputs / "stage2"

    if not stage3_dir.exists():
        raise FileNotFoundError(f"Stage 3 metadata directory not found: {stage3_dir}")
    if not stage4_dir.exists():
        raise FileNotFoundError(f"Stage 4 embeddings directory not found: {stage4_dir}")

    # Initialize stores
    vector_store = VectorStore(config.vector_store)
    graph_store = GraphStore(config.graph_store)

    # Collect items
    item_ids = []
    embeddings = []
    metadatas = []
    image_paths = []

    metadata_files = list(stage3_dir.glob("*.json"))
    print(f"\nIngesting {len(metadata_files)} items from {outputs_dir}")

    for meta_file in metadata_files:
        item_id = meta_file.stem

        # Load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load embedding
        emb_file = stage4_dir / f"{item_id}.json"
        if not emb_file.exists():
            print(f"  ⚠ Skipping {item_id}: no embedding file")
            continue
        with open(emb_file, "r", encoding="utf-8") as f:
            emb_data = json.load(f)
        embedding = emb_data["embedding_vector"]

        # Segmented image path
        seg_image = str(stage2_dir / f"{item_id}.png")

        item_ids.append(item_id)
        embeddings.append(embedding)
        metadatas.append(metadata)
        image_paths.append(seg_image)

        # Add to graph
        graph_store.add_item(item_id, metadata)
        print(f"  ✓ {item_id[:8]}... → {metadata.get('main_category', '?')} / {metadata.get('style_attributes', {}).get('subcategory', '?')}")

    # Batch upsert into ChromaDB
    if item_ids:
        vector_store.add_items_batch(item_ids, embeddings, metadatas, image_paths)

    # Build compatibility edges in the graph
    print("\nBuilding outfit compatibility graph...")
    graph_store.build_compatibility_edges()
    graph_store.save()

    # Print stats
    stats = graph_store.stats()
    print(f"\n--- Ingestion Complete ---")
    print(f"  ChromaDB: {vector_store.count()} items")
    print(f"  Graph:    {stats['items']} items, {stats['compatibility_edges']} compatibility edges")
    print(f"            {stats['categories']} categories, {stats['styles']} styles")

    return vector_store, graph_store


if __name__ == "__main__":
    import sys
    outputs_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "outputs_test"
    )
    ingest_from_outputs(outputs_dir)
