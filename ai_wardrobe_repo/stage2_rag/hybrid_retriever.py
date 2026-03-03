"""
Hybrid Retriever: Combines vector similarity search (ChromaDB)
with graph traversal (NetworkX) for outfit candidate generation.
"""

import json
from typing import List, Dict, Optional

from .vector_store import VectorStore
from .graph_store import GraphStore


class HybridRetriever:
    """
    Combines two retrieval strategies:
    1. Vector search — semantic similarity via Fashion-CLIP embeddings
    2. Graph traversal — compatibility relationships + style/occasion filtering
    """

    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve(
        self,
        query_embedding: List[float],
        intent: Optional[Dict] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Hybrid retrieval: vector search + graph expansion.

        Args:
            query_embedding: Fashion-CLIP embedding of the user's query text
            intent: Extracted intent from LLM (occasion, style, formality, etc.)
            top_k: Max number of items to return

        Returns:
            Ranked list of candidate items with metadata and sources
        """
        candidates = {}

        # ── Strategy 1: Vector Similarity Search ──
        vector_results = self.vector_store.search_by_embedding(
            query_embedding, top_k=top_k
        )
        if vector_results and vector_results["ids"]:
            for i, item_id in enumerate(vector_results["ids"][0]):
                meta = vector_results["metadatas"][0][i] if vector_results["metadatas"] else {}
                dist = vector_results["distances"][0][i] if vector_results["distances"] else 1.0
                similarity = 1.0 - dist  # cosine distance → similarity
                candidates[item_id] = {
                    "item_id": item_id,
                    "metadata": meta,
                    "score": similarity,
                    "sources": ["vector_search"],
                }

        # ── Strategy 2: Graph-Based Filtering ──
        if intent:
            graph_items = self._graph_filter(intent)
            for item_id in graph_items:
                if item_id in candidates:
                    candidates[item_id]["score"] += 0.3  # boost items found by both
                    candidates[item_id]["sources"].append("graph_filter")
                else:
                    node_data = self.graph_store.graph.nodes.get(item_id, {})
                    candidates[item_id] = {
                        "item_id": item_id,
                        "metadata": dict(node_data),
                        "score": 0.3,
                        "sources": ["graph_filter"],
                    }

        # ── Strategy 3: Graph Expansion ──
        # For top vector matches, find compatible items from the graph
        top_vector_ids = list(candidates.keys())[:5]
        for item_id in top_vector_ids:
            compatible = self.graph_store.get_compatible_items(item_id, min_score=2)
            for comp in compatible:
                cid = comp["item_id"]
                if cid in candidates:
                    candidates[cid]["score"] += 0.1 * comp["compatibility_score"]
                    if "graph_expansion" not in candidates[cid]["sources"]:
                        candidates[cid]["sources"].append("graph_expansion")
                else:
                    node_data = self.graph_store.graph.nodes.get(cid, {})
                    candidates[cid] = {
                        "item_id": cid,
                        "metadata": dict(node_data),
                        "score": 0.1 * comp["compatibility_score"],
                        "sources": ["graph_expansion"],
                        "compatible_with": item_id,
                    }

        # Rank by score descending
        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _graph_filter(self, intent: Dict) -> List[str]:
        """Use extracted intent to filter items from the graph."""
        item_sets = []

        # Filter by occasion
        if intent.get("occasion"):
            items = self.graph_store.get_items_by_occasion(intent["occasion"])
            if items:
                item_sets.append(set(items))

        # Filter by style
        if intent.get("style"):
            items = self.graph_store.get_items_by_style(intent["style"])
            if items:
                item_sets.append(set(items))

        # Filter by category
        if intent.get("category"):
            items = self.graph_store.get_items_by_category(intent["category"])
            if items:
                item_sets.append(set(items))

        if not item_sets:
            return []

        # Union of all matching sets (broad matching)
        result = set()
        for s in item_sets:
            result |= s
        return list(result)

    def get_item_details(self, item_ids: List[str]) -> List[Dict]:
        """Fetch full metadata for a list of item IDs from ChromaDB."""
        details = []
        for item_id in item_ids:
            item = self.vector_store.get_item(item_id)
            if item:
                details.append(item)
        return details
