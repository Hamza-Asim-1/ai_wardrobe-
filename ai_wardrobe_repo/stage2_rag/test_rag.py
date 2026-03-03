"""Quick test of the RAG pipeline — outputs results to a clean text file."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1"))

from stage2_rag.config import DEFAULT_RAG_CONFIG
from stage2_rag.ingest import ingest_from_outputs
from stage2_rag.hybrid_retriever import HybridRetriever
from stage2_rag.outfit_generator import OutfitGenerator

def main():
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs_test"
    result_file = base_dir / "rag_result.json"

    # Ingest
    print("Ingesting...")
    vector_store, graph_store = ingest_from_outputs(str(outputs_dir))

    # Init components
    retriever = HybridRetriever(vector_store, graph_store)
    generator = OutfitGenerator(DEFAULT_RAG_CONFIG.outfit_generator)

    # Encode query
    from stage1.config import DEFAULT_CONFIG
    from stage1.Stage4embeddings import FashionCLIPEncoder
    encoder = FashionCLIPEncoder(DEFAULT_CONFIG.fashion_clip)
    encoder.initialize()

    query = "I want to wear something casual"
    print(f"Query: {query}")

    # Extract intent
    intent = generator.extract_intent(query)
    print(f"Intent: {json.dumps(intent)}")

    # Encode
    query_emb = encoder.encode_text(query).tolist()
    print(f"Embedding dim: {len(query_emb)}")

    # Retrieve
    candidates = retriever.retrieve(query_emb, intent, top_k=10)
    print(f"Candidates: {len(candidates)}")

    # Generate outfit
    outfit = generator.generate_outfit(query, candidates, intent)
    print(f"Outfit: {outfit.get('outfit_name', '?')}")

    # Save full result
    result = {
        "query": query,
        "intent": intent,
        "num_candidates": len(candidates),
        "candidates": [
            {"item_id": c["item_id"][:8], "score": round(c["score"], 3), "sources": c.get("sources", [])}
            for c in candidates
        ],
        "outfit": outfit,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Result saved to {result_file}")

if __name__ == "__main__":
    main()
