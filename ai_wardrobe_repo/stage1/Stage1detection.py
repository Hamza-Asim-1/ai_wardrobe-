"""
Stage 1: Object Detection using YOLO-World
Zero-shot detection of fashion items in images
"""

import os
import uuid
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from pathlib import Path

from config import YOLOConfig
from Schemas import DetectionResult


class YOLODetector:
    """
    YOLO-World detector for fashion items.
    Identifies and localizes multiple items in a single image.
    """
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
    def initialize(self):
        """Load the YOLO-World model"""
        try:
            # Import YOLO-World dependencies
            from ultralytics import YOLOWorld
            
            print(f"Loading YOLO-World model on {self.device}...")
            local_checkpoint = Path(__file__).resolve().parent / "yolov8x-worldv2.pt"
            weights = str(local_checkpoint) if local_checkpoint.exists() else "yolov8x-worldv2.pt"
            self.model = YOLOWorld(weights)  # Using YOLOv8-World
            
            # Set custom classes for fashion detection
            self.model.set_classes(self.config.target_categories)
            
            print(f"✓ YOLO-World initialized with {len(self.config.target_categories)} fashion categories")
            
        except ImportError:
            print("✗ Ultralytics unavailable. Please install it.")
            raise
        except Exception as e:
            print(f"✗ YOLO initialization failed ({e}).")
            raise
    
    def detect_items(
        self, 
        image_path: str,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Detect fashion items in an image.
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            self.initialize()


        
        conf = confidence_threshold or self.config.confidence_threshold
        iou = iou_threshold or self.config.iou_threshold
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        detections = []
        
        # Parse results
        if len(results) > 0:
            result = results[0]  # Single image
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                category = self.config.target_categories[class_id]
                
                # Create unique item ID
                item_id = str(uuid.uuid4())
                
                detection = DetectionResult(
                    item_id=item_id,
                    category=category,
                    confidence=confidence,
                    bounding_box={
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max)
                    }
                )
                
                detections.append(detection)
        
        return detections


    
    def visualize_detections(
        self,
        image_path: str,
        detections: List[DetectionResult],
        output_path: str
    ):
        """
        Draw bounding boxes on the image for visualization.
        
        Args:
            image_path: Original image path
            detections: List of DetectionResult objects
            output_path: Where to save the annotated image
        """
        from PIL import ImageDraw, ImageFont
        
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
            '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
        ]
        
        for idx, detection in enumerate(detections):
            bbox = detection.bounding_box
            color = colors[idx % len(colors)]
            
            # Draw bounding box
            draw.rectangle(
                [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]],
                outline=color,
                width=3
            )
            
            # Draw label
            label = f"{detection.category} ({detection.confidence:.2f})"
            
            # Background for text
            text_bbox = draw.textbbox((bbox["x_min"], bbox["y_min"] - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text(
                (bbox["x_min"], bbox["y_min"] - 25),
                label,
                fill='white',
                font=font
            )
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"✓ Detection visualization saved to {output_path}")
    
    def get_cropped_image(
        self,
        image_path: str,
        detection: DetectionResult,
        padding: int = 10
    ) -> Image.Image:
        """
        Extract a cropped image based on bounding box.
        
        Args:
            image_path: Original image path
            detection: DetectionResult with bounding box
            padding: Extra pixels to include around the box
            
        Returns:
            Cropped PIL Image
        """
        image = Image.open(image_path).convert("RGB")
        bbox = detection.bounding_box
        
        # Add padding
        x_min = max(0, int(bbox["x_min"]) - padding)
        y_min = max(0, int(bbox["y_min"]) - padding)
        x_max = min(image.width, int(bbox["x_max"]) + padding)
        y_max = min(image.height, int(bbox["y_max"]) + padding)
        
        cropped = image.crop((x_min, y_min, x_max, y_max))
        return cropped


# Standalone function for easy usage
def detect_fashion_items(
    image_path: str,
    config: Optional[YOLOConfig] = None
) -> List[DetectionResult]:
    """
    Convenience function to detect fashion items in an image.
    
    Args:
        image_path: Path to image
        config: Optional YOLOConfig, uses defaults if not provided
        
    Returns:
        List of DetectionResult objects
    """
    if config is None:
        config = YOLOConfig()
    
    detector = YOLODetector(config)
    return detector.detect_items(image_path)
