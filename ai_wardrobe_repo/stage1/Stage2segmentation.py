"""
Stage 2: Image Segmentation using rembg (background removal)
Extracts clean, background-removed images of detected items
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from pathlib import Path

from config import SAMConfig
from Schemas import DetectionResult, SegmentationResult


class SAMSegmenter:
    """
    Segmenter for extracting clean item images.
    Uses rembg for background removal and bbox cropping.
    """

    def __init__(self, config: SAMConfig):
        self.config = config
        self.rembg_session = None

    def initialize(self):
        """Load the rembg model"""
        try:
            from rembg import new_session
            print("Loading rembg background removal model...")
            self.rembg_session = new_session("u2net")
            print("✓ rembg initialized successfully")
        except ImportError:
            print("✗ rembg not installed. Please run: pip install rembg")
            raise
        except Exception as e:
            print(f"✗ rembg initialization failed: {e}")
            raise

    def segment_from_box(
        self,
        image_path: str,
        detection: DetectionResult,
        output_path: str
    ) -> SegmentationResult:
        """
        Segment an item from the image using bounding box + background removal.

        Args:
            image_path: Path to original image
            detection: DetectionResult with bounding box
            output_path: Where to save the segmented PNG

        Returns:
            SegmentationResult object
        """
        if self.rembg_session is None:
            self.initialize()

        # Load image
        image = Image.open(image_path).convert("RGBA")

        # Check if image already has transparency (transparent-background PNG)
        alpha = np.array(image)[:, :, 3]
        has_transparency = np.any(alpha < 255)

        # Crop to bounding box with padding
        bbox = detection.bounding_box
        padding = 10
        x_min = max(0, int(bbox["x_min"]) - padding)
        y_min = max(0, int(bbox["y_min"]) - padding)
        x_max = min(image.width, int(bbox["x_max"]) + padding)
        y_max = min(image.height, int(bbox["y_max"]) + padding)

        cropped = image.crop((x_min, y_min, x_max, y_max))

        if has_transparency:
            # Image already has transparent background — just save the crop
            segmented = cropped
            print(f"  → Image already has transparency, using direct crop")
        else:
            # Use rembg to remove background
            from rembg import remove
            cropped_rgb = cropped.convert("RGB")
            segmented = remove(cropped_rgb, session=self.rembg_session)
            print(f"  → Background removed with rembg")

        # Save as PNG with transparency
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        segmented.save(output_path, "PNG")

        # Calculate mask area from alpha channel
        seg_array = np.array(segmented)
        mask_area = int(np.sum(seg_array[:, :, 3] > 128)) if seg_array.shape[2] == 4 else seg_array.shape[0] * seg_array.shape[1]

        result = SegmentationResult(
            item_id=detection.item_id,
            segmented_image_path=output_path,
            mask_area=mask_area,
            success=True
        )

        return result

    def batch_segment(
        self,
        image_path: str,
        detections: list,
        output_dir: str
    ) -> list:
        """
        Segment multiple items from the same image.

        Args:
            image_path: Path to original image
            detections: List of DetectionResult objects
            output_dir: Directory to save segmented images

        Returns:
            List of SegmentationResult objects
        """
        results = []

        for detection in detections:
            output_path = os.path.join(
                output_dir,
                f"{detection.item_id}.png"
            )

            try:
                result = self.segment_from_box(image_path, detection, output_path)
                results.append(result)
            except Exception as e:
                print(f"✗ Segmentation failed for {detection.item_id}: {e}")
                import traceback
                traceback.print_exc()
                result = SegmentationResult(
                    item_id=detection.item_id,
                    segmented_image_path="",
                    mask_area=0,
                    success=False
                )
                results.append(result)

        return results


# Convenience function
def segment_fashion_item(
    image_path: str,
    detection: DetectionResult,
    output_path: str,
    config: Optional[SAMConfig] = None
) -> SegmentationResult:
    """
    Segment a single fashion item.

    Args:
        image_path: Original image path
        detection: Detection result with bounding box
        output_path: Where to save segmented image
        config: Optional SAMConfig

    Returns:
        SegmentationResult
    """
    if config is None:
        config = SAMConfig()

    segmenter = SAMSegmenter(config)
    return segmenter.segment_from_box(image_path, detection, output_path)
