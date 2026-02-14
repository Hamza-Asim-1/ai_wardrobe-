"""
Stage 2: Image Segmentation using SAM 2 (Segment Anything Model 2)
Extracts clean, background-removed images of detected items
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import torch
from pathlib import Path

from config import SAMConfig
from Schemas import DetectionResult, SegmentationResult


class SAMSegmenter:
    """
    SAM 2 segmenter for extracting clean item images.
    Removes backgrounds and creates professional PNG cutouts.
    """
    
    def __init__(self, config: SAMConfig):
        self.config = config
        self.model = None
        self.predictor = None
        self.fallback_mode = False
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    def initialize(self):
        """Load SAM 2 model"""
        try:
            # Try importing SAM 2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            print(f"Loading SAM 2 model ({self.config.model_type}) on {self.device}...")
            
            # Download model if needed
            checkpoint_path = self._download_checkpoint()
            config_file = self._get_config_file()
            
            # Build model
            sam2_model = build_sam2(config_file, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            print("✓ SAM 2 initialized successfully")
            
        except ImportError as e:
            print(
                "⚠ SAM 2 not installed. Falling back to bbox-based alpha mask segmentation."
            )
            self.fallback_mode = True
    
    def _download_checkpoint(self) -> str:
        """Download SAM 2 checkpoint if needed"""
        # Use the user's home directory for model storage
        import os
        home_dir = os.path.expanduser("~")
        checkpoint_dir = Path(home_dir) / ".cache" / "sam2"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / self.config.checkpoint_path
        
        if not checkpoint_file.exists():
            print(f"Downloading SAM 2 checkpoint: {self.config.checkpoint_path}")
            
            # Map model types to download URLs
            model_urls = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
            }
            
            url = model_urls.get(self.config.model_type, model_urls["vit_h"])
            
            import urllib.request
            urllib.request.urlretrieve(url, checkpoint_file)
            print(f"✓ Downloaded checkpoint to {checkpoint_file}")
        
        return str(checkpoint_file)
    
    def _get_config_file(self) -> str:
        """Get the appropriate config file for the model type"""
        # SAM 2 uses specific config files
        config_map = {
            "vit_h": "sam2_hiera_l.yaml",
            "vit_l": "sam2_hiera_l.yaml", 
            "vit_b": "sam2_hiera_b+.yaml"
        }
        return config_map.get(self.config.model_type, "sam2_hiera_l.yaml")
    
    def segment_from_box(
        self,
        image_path: str,
        detection: DetectionResult,
        output_path: str
    ) -> SegmentationResult:
        """
        Segment an item from the image using the bounding box.
        
        Args:
            image_path: Path to original image
            detection: DetectionResult with bounding box
            output_path: Where to save the segmented PNG
            
        Returns:
            SegmentationResult object
        """
        if self.predictor is None:
            self.initialize()

        if self.fallback_mode:
            return self._segment_bbox_fallback(image_path, detection, output_path)
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM
        self.predictor.set_image(image_rgb)
        
        # Extract bounding box
        bbox = detection.bounding_box
        input_box = np.array([
            bbox["x_min"],
            bbox["y_min"],
            bbox["x_max"],
            bbox["y_max"]
        ])
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        
        # Get the best mask
        mask = masks[0]
        
        # Create RGBA image with transparency
        segmented_image = self._apply_mask_with_transparency(image_rgb, mask)
        
        # Save as PNG
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA))
        
        # Calculate mask area
        mask_area = int(np.sum(mask))
        
        result = SegmentationResult(
            item_id=detection.item_id,
            segmented_image_path=output_path,
            mask_area=mask_area,
            success=True
        )

        return result

    def _segment_bbox_fallback(
        self,
        image_path: str,
        detection: DetectionResult,
        output_path: str,
    ) -> SegmentationResult:
        """Fallback segmentation when SAM2 is unavailable."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]
        bbox = detection.bounding_box
        x_min = max(0, int(bbox["x_min"]))
        y_min = max(0, int(bbox["y_min"]))
        x_max = min(w, int(bbox["x_max"]))
        y_max = min(h, int(bbox["y_max"]))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 1
        segmented_image = self._apply_mask_with_transparency(image_rgb, mask)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA))

        mask_area = int(np.sum(mask))
        return SegmentationResult(
            item_id=detection.item_id,
            segmented_image_path=output_path,
            mask_area=mask_area,
            success=True,
        )
    
    def _apply_mask_with_transparency(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply mask to image and create transparent background.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
            
        Returns:
            RGBA image with transparent background (H, W, 4)
        """
        # Create RGBA image
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Copy RGB channels
        rgba_image[:, :, :3] = image
        
        # Set alpha channel based on mask
        rgba_image[:, :, 3] = (mask * 255).astype(np.uint8)
        
        return rgba_image
    
    def segment_with_refinement(
        self,
        image_path: str,
        detection: DetectionResult,
        output_path: str,
        use_points: bool = True
    ) -> SegmentationResult:
        """
        Segment with additional point prompts for better accuracy.
        
        Args:
            image_path: Path to original image
            detection: DetectionResult with bounding box
            output_path: Where to save segmented image
            use_points: Whether to add center point prompt
            
        Returns:
            SegmentationResult object
        """
        if self.predictor is None:
            self.initialize()
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_rgb)
        
        # Get bounding box
        bbox = detection.bounding_box
        input_box = np.array([
            bbox["x_min"],
            bbox["y_min"],
            bbox["x_max"],
            bbox["y_max"]
        ])
        
        # Optionally add center point as positive prompt
        point_coords = None
        point_labels = None
        
        if use_points:
            center_x = (bbox["x_min"] + bbox["x_max"]) / 2
            center_y = (bbox["y_min"] + bbox["y_max"]) / 2
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])  # 1 = foreground point
        
        # Predict with box and optional points
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=True  # Get multiple options
        )
        
        # Choose the mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Create transparent image
        segmented_image = self._apply_mask_with_transparency(image_rgb, mask)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA))
        
        mask_area = int(np.sum(mask))
        
        return SegmentationResult(
            item_id=detection.item_id,
            segmented_image_path=output_path,
            mask_area=mask_area,
            success=True
        )
    
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
                # Create failed result
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
