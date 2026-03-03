"""
End-to-end fashion wardrobe pipeline orchestration.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DEFAULT_CONFIG, PipelineConfig
from Schemas import WardrobeItem
from Stage1detection import YOLODetector
from Stage2segmentation import SAMSegmenter
from stage3metadata import GeminiExtractor
from Stage4embeddings import FashionCLIPEncoder, SemanticSearchEngine


class FashionWardrobePipeline:
    """Orchestrates detection, segmentation, metadata, and embeddings."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.output_root = Path(self.config.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.stage1_dir = self.output_root / "stage1"
        self.stage2_dir = self.output_root / "stage2"
        self.stage3_dir = self.output_root / "stage3"
        self.stage4_dir = self.output_root / "stage4"
        for d in [self.stage1_dir, self.stage2_dir, self.stage3_dir, self.stage4_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.detector = YOLODetector(self.config.yolo)
        self.segmenter = SAMSegmenter(self.config.sam)
        self.metadata_extractor = GeminiExtractor(self.config.gemini)
        self.encoder = FashionCLIPEncoder(self.config.fashion_clip)
        self.search_engine = SemanticSearchEngine(self.encoder, self.config.fashion_clip)

        self.items: Dict[str, WardrobeItem] = {}

    def process_image(self, image_path: str, visualize: bool = False) -> List[WardrobeItem]:
        """Run all phases for a single image and return processed items."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        detections = self.detector.detect_items(str(path))
        if not detections:
            return []

        if visualize:
            vis_path = self.stage1_dir / f"detected_{path.stem}.png"
            self.detector.visualize_detections(str(path), detections, str(vis_path))

        segmentation_results = self.segmenter.batch_segment(
            image_path=str(path),
            detections=detections,
            output_dir=str(self.stage2_dir),
        )
        segmentation_by_id = {s.item_id: s for s in segmentation_results}

        processed_items: List[WardrobeItem] = []
        for detection in detections:
            seg = segmentation_by_id.get(detection.item_id)
            if not seg or not seg.success:
                continue

            metadata = self.metadata_extractor.extract_with_schema_validation(
                image_path=seg.segmented_image_path,
                item_id=detection.item_id,
                category_hint=detection.category,
            )
            metadata_path = self.stage3_dir / f"{detection.item_id}.json"
            self.metadata_extractor.save_metadata(metadata, str(metadata_path))

            embedding = self.encoder.create_embedding_result(
                image_path=seg.segmented_image_path,
                item_id=detection.item_id,
            )
            self.search_engine.add_item(
                item_id=detection.item_id,
                embedding=self._to_numpy(embedding.embedding_vector),
                metadata=metadata,
                image_path=seg.segmented_image_path,
            )

            item = WardrobeItem(
                item_id=detection.item_id,
                original_image_path=str(path),
                detection=detection,
                segmentation=seg,
                metadata=metadata,
                embedding=embedding,
                processing_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.items[item.item_id] = item
            processed_items.append(item)

        return processed_items

    def process_images(self, image_paths: List[str], visualize: bool = False) -> List[WardrobeItem]:
        """Run all phases for a list of images."""
        all_items: List[WardrobeItem] = []
        for image_path in image_paths:
            all_items.extend(self.process_image(image_path, visualize=visualize))
        return all_items

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_category: Optional[str] = None,
        filter_style: Optional[List[str]] = None,
        filter_season: Optional[List[str]] = None,
        filter_formality: Optional[str] = None,
    ):
        """Search indexed wardrobe items."""
        return self.search_engine.search(
            query=query,
            top_k=top_k,
            filter_category=filter_category,
            filter_style=filter_style,
            filter_season=filter_season,
            filter_formality=filter_formality,
        )

    def save_search_index(self, output_dir: Optional[str] = None) -> str:
        """Persist embeddings + metadata index to disk."""
        index_dir = output_dir or str(self.stage4_dir / "index")
        self.search_engine.save_index(index_dir)
        return index_dir

    @staticmethod
    def _to_numpy(values: List[float]):
        import numpy as np

        return np.array(values, dtype=float)


def run_demo_pipeline(
    image_paths: List[str],
    config: Optional[PipelineConfig] = None,
    visualize: bool = True,
) -> Tuple[FashionWardrobePipeline, List[WardrobeItem]]:
    """Convenience helper used by the test script."""
    pipeline = FashionWardrobePipeline(config=config)
    items = pipeline.process_images(image_paths, visualize=visualize)
    pipeline.save_search_index()
    return pipeline, items
