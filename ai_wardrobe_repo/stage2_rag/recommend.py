"""
Recommend: End-to-end script to ingest wardrobe data and run outfit queries.

Usage:
    python recommend.py                          # interactive mode
    python recommend.py "I want something casual" # single query
"""

import sys
import json
from pathlib import Path

# Add parent to path so we can import stage1 modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1"))

from stage2_rag.config import DEFAULT_RAG_CONFIG
from stage2_rag.ingest import ingest_from_outputs
from stage2_rag.hybrid_retriever import HybridRetriever
from stage2_rag.outfit_generator import OutfitGenerator


def encode_query_text(query: str) -> list:
    """
    Encode user query text into a Fashion-CLIP embedding.
    Uses the same encoder as Stage 4.
    """
    from stage1.config import DEFAULT_CONFIG
    from stage1.Stage4embeddings import FashionCLIPEncoder

    encoder = FashionCLIPEncoder(DEFAULT_CONFIG.fashion_clip)
    encoder.initialize()
    embedding = encoder.encode_text(query)
    return embedding.tolist()


def main():
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs_test"
    config = DEFAULT_RAG_CONFIG

    # ── Step 1: Ingest pipeline outputs into databases ──
    print("=" * 60)
    print("STEP 1: Ingesting pipeline outputs into databases")
    print("=" * 60)
    vector_store, graph_store = ingest_from_outputs(str(outputs_dir), config)

    # ── Step 2: Initialize components ──
    print("\n" + "=" * 60)
    print("STEP 2: Initializing RAG components")
    print("=" * 60)
    retriever = HybridRetriever(vector_store, graph_store)
    generator = OutfitGenerator(config.outfit_generator)

    # ── Step 3: Process query ──
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        process_query(query, retriever, generator)
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("OUTFIT RECOMMENDER — Interactive Mode")
        print("Type your outfit request (or 'quit' to exit)")
        print("=" * 60)
        while True:
            query = input("\n> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            process_query(query, retriever, generator)


def process_query(query: str, retriever: HybridRetriever, generator: OutfitGenerator):
    """Process a single outfit query through the full pipeline."""
    print(f"\n{'─' * 50}")
    print(f"Query: \"{query}\"")
    print(f"{'─' * 50}")

    # 1) Extract intent from query
    print("\n[1/4] Extracting intent...")
    try:
        intent = generator.extract_intent(query)
        print(f"  Style:     {intent.get('style')}")
        print(f"  Occasion:  {intent.get('occasion')}")
        print(f"  Formality: {intent.get('formality')}")
        print(f"  Season:    {intent.get('season')}")
        print(f"  Mood:      {intent.get('mood')}")
    except Exception as e:
        print(f"  Warning: Intent extraction failed ({e}), using vector search only")
        intent = {}

    # 2) Encode query to embedding
    print("\n[2/4] Encoding query with Fashion-CLIP...")
    query_embedding = encode_query_text(query)
    print(f"  Embedding dim: {len(query_embedding)}")

    # 3) Hybrid retrieval
    print("\n[3/4] Retrieving candidates (vector + graph)...")
    candidates = retriever.retrieve(
        query_embedding=query_embedding,
        intent=intent,
        top_k=10,
    )
    print(f"  Found {len(candidates)} candidate items:")
    for c in candidates:
        meta = c["metadata"]
        cat = meta.get("main_category", "?")
        sub = meta.get("subcategory", "?")
        sources = ", ".join(c.get("sources", []))
        print(f"    [{c['score']:.2f}] {cat}/{sub} ({sources})")

    # 4) Generate outfit
    print("\n[4/4] Generating outfit recommendation...")
    try:
        outfit = generator.generate_outfit(query, candidates, intent)
        print(f"\n{'*' * 50}")
        print(f"  OUTFIT: {outfit.get('outfit_name', 'Untitled')}")
        print(f"  CONFIDENCE: {outfit.get('confidence', '?')}")
        print(f"{'*' * 50}")

        print(f"\n  Selected Items:")
        for item in outfit.get("selected_items", []):
            print(f"    - [{item.get('category')}] {item.get('subcategory', '')} → {item.get('role', '')}")

        print(f"\n  Styling Notes:")
        print(f"    {outfit.get('styling_notes', '')}")

        print(f"\n  Color Story:")
        print(f"    {outfit.get('color_story', '')}")

        missing = outfit.get("missing_pieces", [])
        if missing:
            print(f"\n  Missing Pieces:")
            for m in missing:
                print(f"    - {m}")

    except Exception as e:
        print(f"  Outfit generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
