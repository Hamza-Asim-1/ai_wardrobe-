#!/usr/bin/env python3
"""
Test script: Runs all 4 pipeline stages on user images with detailed per-stage logging.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Ensure stage1 is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PipelineConfig, DEFAULT_CONFIG
from Stage1detection import YOLODetector
from Stage2segmentation import SAMSegmenter
from stage3metadata import GeminiExtractor
from Stage4embeddings import FashionCLIPEncoder


def main():
    # --- Paths ---
    base_dir = Path(__file__).resolve().parent          # stage1/
    images_dir = base_dir.parent.parent / "images"      # alta/images
    if not images_dir.exists():
        images_dir = Path(r"c:\Users\dell\Documents\ai_outfit\alta\images")
    output_dir = base_dir.parent / "outputs_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = output_dir / "stage1"
    stage2_dir = output_dir / "stage2"
    stage3_dir = output_dir / "stage3"
    stage4_dir = output_dir / "stage4"
    for d in [stage1_dir, stage2_dir, stage3_dir, stage4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Discover images ---
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in exts])
    print(f"\n{'='*60}")
    print(f"Found {len(image_paths)} images in {images_dir}")
    for p in image_paths:
        print(f"  • {p.name}")
    print(f"{'='*60}\n")

    if not image_paths:
        print("No images found. Exiting.")
        return

    # --- Initialize models ---
    config = DEFAULT_CONFIG

    print("── INITIALIZING MODELS ──")
    detector = YOLODetector(config.yolo)
    detector.initialize()

    segmenter = SAMSegmenter(config.sam)
    segmenter.initialize()

    extractor = GeminiExtractor(config.gemini)
    extractor.initialize()

    encoder = FashionCLIPEncoder(config.fashion_clip)
    encoder.initialize()
    print("── ALL MODELS READY ──\n")

    # --- Process each image through all 4 stages ---
    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"IMAGE: {img_path.name}")
        print(f"{'='*60}")

        # ── STAGE 1: Detection ──
        print("\n── STAGE 1: Detection ──")
        try:
            detections = detector.detect_items(str(img_path))
            print(f"  ✓ Detected {len(detections)} item(s)")
            for i, det in enumerate(detections):
                print(f"    [{i+1}] {det.category} (conf: {det.confidence:.2f})")

            # Save visualization
            vis_path = stage1_dir / f"detected_{img_path.stem}.png"
            detector.visualize_detections(str(img_path), detections, str(vis_path))
            print(f"  ✓ Visualization saved: {vis_path.name}")
        except Exception as e:
            print(f"  ✗ Detection failed: {e}")
            traceback.print_exc()
            continue

        if not detections:
            print("  ⚠ No items detected – skipping remaining stages")
            continue

        # ── STAGE 2: Segmentation ──
        print("\n── STAGE 2: Segmentation ──")
        seg_results = segmenter.batch_segment(
            image_path=str(img_path),
            detections=detections,
            output_dir=str(stage2_dir),
        )
        successful_segs = [s for s in seg_results if s.success]
        print(f"  ✓ Segmented {len(successful_segs)}/{len(detections)} item(s)")
        for s in seg_results:
            status = "✓" if s.success else "✗"
            print(f"    {status} {s.item_id[:8]}... mask_area={s.mask_area}")

        # ── STAGE 3: Metadata Extraction ──
        print("\n── STAGE 3: Metadata Extraction (Gemini) ──")
        for seg in successful_segs:
            det = next(d for d in detections if d.item_id == seg.item_id)
            try:
                metadata = extractor.extract_with_schema_validation(
                    image_path=seg.segmented_image_path,
                    item_id=seg.item_id,
                    category_hint=det.category,
                )
                # Save metadata JSON
                meta_path = stage3_dir / f"{seg.item_id}.json"
                extractor.save_metadata(metadata, str(meta_path))

                print(f"  ✓ {seg.item_id[:8]}...")
                print(f"    Category: {metadata.main_category} / {metadata.style_attributes.subcategory}")
                print(f"    Colors:   {', '.join(c.name for c in metadata.colors)}")
                print(f"    Material: {metadata.materials.primary_material}")
                print(f"    Style:    {', '.join(metadata.style_attributes.style_tags)}")
                print(f"    Vibe:     {metadata.vibe_description}")
            except Exception as e:
                print(f"  ✗ Metadata failed for {seg.item_id[:8]}...: {e}")
                traceback.print_exc()

        # ── STAGE 4: Embeddings ──
        print("\n── STAGE 4: Embeddings (Fashion-CLIP) ──")
        all_embeddings = {}
        for seg in successful_segs:
            try:
                emb_result = encoder.create_embedding_result(
                    image_path=seg.segmented_image_path,
                    item_id=seg.item_id,
                )
                # Save individual embedding as JSON
                import json as json_mod
                emb_json_path = stage4_dir / f"{seg.item_id}.json"
                with open(emb_json_path, 'w') as ef:
                    json_mod.dump({
                        "item_id": emb_result.item_id,
                        "embedding_dimension": emb_result.embedding_dimension,
                        "embedding_vector": emb_result.embedding_vector,
                    }, ef, indent=2)
                
                all_embeddings[seg.item_id] = emb_result.embedding_vector
                print(f"  ✓ {seg.item_id[:8]}... dim={emb_result.embedding_dimension} → {emb_json_path.name}")
            except Exception as e:
                print(f"  ✗ Embedding failed for {seg.item_id[:8]}...: {e}")
                traceback.print_exc()
        
        # Save all embeddings as a single NPZ file for easy loading
        if all_embeddings:
            import numpy as np
            npz_path = stage4_dir / "all_embeddings.npz"
            np.savez(str(npz_path), **{k: np.array(v) for k, v in all_embeddings.items()})
            print(f"  ✓ All embeddings saved to {npz_path.name}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"Outputs saved to: {output_dir}")
    for d in [stage1_dir, stage2_dir, stage3_dir, stage4_dir]:
        files = list(d.glob("*"))
        print(f"  {d.name}: {len(files)} file(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
