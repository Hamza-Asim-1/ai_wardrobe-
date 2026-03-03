"""
Graph Store: Lightweight in-process fashion knowledge graph.
Uses NetworkX for outfit compatibility relationships.
Can be upgraded to Neo4j for production.
"""

import json
import os
import networkx as nx
from typing import List, Dict, Optional, Set, Tuple

from .config import GraphStoreConfig


# Color wheel groups for complementary color matching
COLOR_GROUPS = {
    "neutral": ["black", "white", "gray", "grey", "beige", "cream", "ivory", "charcoal", "taupe", "nude"],
    "warm": ["red", "orange", "yellow", "coral", "burgundy", "maroon", "rust", "terracotta", "gold", "amber"],
    "cool": ["blue", "navy", "teal", "cyan", "aqua", "turquoise", "indigo", "cobalt"],
    "earth": ["brown", "tan", "olive", "khaki", "camel", "chocolate", "coffee", "mocha"],
    "jewel": ["emerald", "green", "purple", "violet", "magenta", "plum", "sapphire", "ruby"],
    "pastel": ["pink", "lavender", "mint", "peach", "lilac", "baby blue", "blush", "rose"],
}

# Categories that naturally pair together
COMPLEMENTARY_CATEGORIES = [
    ("Tops", "Bottoms"),
    ("Tops", "Outerwear"),
    ("Dresses", "Outerwear"),
    ("Dresses", "Footwear"),
    ("Tops", "Footwear"),
    ("Bottoms", "Footwear"),
    ("Tops", "Accessories"),
    ("Outerwear", "Accessories"),
]


class GraphStore:
    """NetworkX-based fashion knowledge graph for outfit compatibility."""

    def __init__(self, config: GraphStoreConfig = None):
        self.config = config or GraphStoreConfig()
        self.graph = nx.Graph()
        self._load_if_exists()

    def _load_if_exists(self):
        """Load graph from disk if it exists."""
        if os.path.exists(self.config.persist_path):
            with open(self.config.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            print(f"✓ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        else:
            print("✓ New graph initialized")

    def save(self):
        """Persist graph to disk."""
        os.makedirs(os.path.dirname(self.config.persist_path), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.config.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Graph saved: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def add_item(self, item_id: str, metadata: Dict):
        """Add a wardrobe item node with its properties and category/style links."""
        style = metadata.get("style_attributes", {})
        context = metadata.get("context", {})
        colors = metadata.get("colors", [])

        # Add item node
        self.graph.add_node(
            item_id,
            node_type="item",
            main_category=metadata.get("main_category", ""),
            subcategory=style.get("subcategory", ""),
            formality=context.get("formality_level", ""),
            style_tags=json.dumps(style.get("style_tags", [])),
            colors=json.dumps([c.get("name", "").lower() for c in colors]),
            seasons=json.dumps(context.get("season_suitability", [])),
            occasions=json.dumps(context.get("occasions", [])),
            pattern=style.get("pattern", ""),
            material=metadata.get("materials", {}).get("primary_material", ""),
        )

        # Add category node + edge
        cat = metadata.get("main_category", "")
        if cat:
            self.graph.add_node(cat, node_type="category")
            self.graph.add_edge(item_id, cat, relation="BELONGS_TO_CATEGORY")

        # Add style nodes + edges
        for tag in style.get("style_tags", []):
            style_id = f"style:{tag}"
            self.graph.add_node(style_id, node_type="style", name=tag)
            self.graph.add_edge(item_id, style_id, relation="HAS_STYLE")

        # Add occasion nodes + edges
        for occ in context.get("occasions", []):
            occ_id = f"occasion:{occ}"
            self.graph.add_node(occ_id, node_type="occasion", name=occ)
            self.graph.add_edge(item_id, occ_id, relation="SUITABLE_FOR")

    def build_compatibility_edges(self):
        """Build PAIRS_WITH edges between items based on fashion rules."""
        items = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "item"]
        new_edges = 0

        for i, a in enumerate(items):
            for b in items[i + 1:]:
                reasons = self._compute_compatibility(a, b)
                if reasons:
                    self.graph.add_edge(
                        a, b,
                        relation="PAIRS_WITH",
                        reasons=json.dumps(reasons),
                        compatibility_score=len(reasons),
                    )
                    new_edges += 1

        print(f"✓ Built {new_edges} compatibility edges")

    def _compute_compatibility(self, item_a: str, item_b: str) -> List[str]:
        """Determine if two items are compatible and return reasons."""
        a = self.graph.nodes[item_a]
        b = self.graph.nodes[item_b]
        reasons = []

        cat_a = a.get("main_category", "")
        cat_b = b.get("main_category", "")

        # Rule 1: Must be different categories (no two tops together)
        if cat_a == cat_b:
            return []

        # Rule 2: Complementary categories
        pair = tuple(sorted([cat_a, cat_b]))
        if pair in [(c1, c2) if c1 < c2 else (c2, c1) for c1, c2 in COMPLEMENTARY_CATEGORIES]:
            reasons.append("complementary_categories")

        # Rule 3: Shared style tags
        styles_a = set(json.loads(a.get("style_tags", "[]")))
        styles_b = set(json.loads(b.get("style_tags", "[]")))
        shared_styles = styles_a & styles_b
        if shared_styles:
            reasons.append(f"shared_styles:{','.join(shared_styles)}")

        # Rule 4: Same formality level
        if a.get("formality", "") == b.get("formality", "") and a.get("formality"):
            reasons.append("same_formality")

        # Rule 5: Compatible colors (neutrals go with everything, same group pairs)
        colors_a = set(json.loads(a.get("colors", "[]")))
        colors_b = set(json.loads(b.get("colors", "[]")))
        if self._colors_compatible(colors_a, colors_b):
            reasons.append("compatible_colors")

        # Rule 6: Shared seasons
        seasons_a = set(json.loads(a.get("seasons", "[]")))
        seasons_b = set(json.loads(b.get("seasons", "[]")))
        if seasons_a & seasons_b or "All-Season" in seasons_a | seasons_b:
            reasons.append("shared_season")

        return reasons

    @staticmethod
    def _colors_compatible(colors_a: Set[str], colors_b: Set[str]) -> bool:
        """Check if two color sets are fashion-compatible."""
        def get_groups(colors):
            groups = set()
            for c in colors:
                c_lower = c.lower()
                for group, members in COLOR_GROUPS.items():
                    if any(m in c_lower for m in members):
                        groups.add(group)
            return groups

        groups_a = get_groups(colors_a)
        groups_b = get_groups(colors_b)

        # Neutrals pair with everything
        if "neutral" in groups_a or "neutral" in groups_b:
            return True
        # Same color group pairs well
        if groups_a & groups_b:
            return True
        # Earth + warm, cool + jewel, pastel + neutral are good combos
        good_combos = [{"earth", "warm"}, {"cool", "jewel"}, {"pastel", "cool"}]
        combined = groups_a | groups_b
        for combo in good_combos:
            if combo.issubset(combined):
                return True
        return False

    def get_compatible_items(self, item_id: str, min_score: int = 2) -> List[Dict]:
        """Get items compatible with the given item, sorted by compatibility score."""
        if item_id not in self.graph:
            return []

        compatible = []
        for neighbor in self.graph.neighbors(item_id):
            edge = self.graph.edges[item_id, neighbor]
            if edge.get("relation") == "PAIRS_WITH":
                score = edge.get("compatibility_score", 0)
                if score >= min_score:
                    compatible.append({
                        "item_id": neighbor,
                        "compatibility_score": score,
                        "reasons": json.loads(edge.get("reasons", "[]")),
                        "metadata": dict(self.graph.nodes[neighbor]),
                    })

        compatible.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return compatible

    def get_items_by_style(self, style_tag: str) -> List[str]:
        """Get all item IDs that have a given style tag."""
        style_node = f"style:{style_tag}"
        if style_node not in self.graph:
            return []
        return [
            n for n in self.graph.neighbors(style_node)
            if self.graph.nodes[n].get("node_type") == "item"
        ]

    def get_items_by_occasion(self, occasion: str) -> List[str]:
        """Get all item IDs suitable for an occasion."""
        occ_node = f"occasion:{occasion}"
        if occ_node not in self.graph:
            return []
        return [
            n for n in self.graph.neighbors(occ_node)
            if self.graph.nodes[n].get("node_type") == "item"
        ]

    def get_items_by_category(self, category: str) -> List[str]:
        """Get all item IDs in a category."""
        if category not in self.graph:
            return []
        return [
            n for n in self.graph.neighbors(category)
            if self.graph.nodes[n].get("node_type") == "item"
        ]

    def get_all_items(self) -> List[Dict]:
        """Get all item nodes with their properties."""
        return [
            {"item_id": n, **dict(d)}
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "item"
        ]

    def stats(self) -> Dict:
        """Return graph statistics."""
        nodes = dict(self.graph.nodes(data=True))
        items = [n for n, d in nodes.items() if d.get("node_type") == "item"]
        pairs_with = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("relation") == "PAIRS_WITH"]
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "items": len(items),
            "compatibility_edges": len(pairs_with),
            "categories": len([n for n, d in nodes.items() if d.get("node_type") == "category"]),
            "styles": len([n for n, d in nodes.items() if d.get("node_type") == "style"]),
        }
