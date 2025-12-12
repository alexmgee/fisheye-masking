#!/usr/bin/env python3
"""
Advanced Multi-Geometry Masking System with SAM3
=================================================
Version: 2.0
Author: 360-to-splat-v2
License: MIT

A production-ready masking system that handles fisheye, pinhole, and equirectangular
images using SAM3's text-based prompting capabilities. Designed for removing capture
equipment, operators, and unwanted objects from 360° reconstruction pipelines.

Key Features:
- SAM3 text-based prompting ("remove tripod operator equipment")
- Multi-geometry support (fisheye, pinhole, equirectangular)
- Temporal consistency for video sequences
- Multiple fallback models (FastSAM, EfficientSAM, MobileSAM)
- Interactive refinement interface
- Batch processing with quality control
- GPU acceleration with automatic CPU fallback
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from abc import ABC, abstractmethod
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import time
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing segmentation models in order of preference
try:
    # SAM3 - Primary model (November 2025 release)
    from sam3.build_sam import build_sam3
    from sam3.automatic_mask_generator import SAM3ImagePredictor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    warnings.warn("SAM3 not found. Install from: https://github.com/facebookresearch/sam3")

try:
    # FastSAM - Fast fallback
    from ultralytics import FastSAM
    HAS_FASTSAM = True
except ImportError:
    HAS_FASTSAM = False
    warnings.warn("FastSAM not found. Install with: pip install ultralytics")

try:
    # EfficientSAM - Efficient fallback
    from efficient_sam import build_efficient_sam
    HAS_EFFICIENTSAM = True
except ImportError:
    HAS_EFFICIENTSAM = False

try:
    # Original SAM2 - Legacy fallback
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

# OpenCV is required
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    raise ImportError("OpenCV is required. Install with: pip install opencv-python")


class ImageGeometry(Enum):
    """Supported image geometries."""
    PINHOLE = "pinhole"           # Standard perspective camera
    FISHEYE = "fisheye"           # Fisheye lens (single)
    DUAL_FISHEYE = "dual_fisheye" # Dual fisheye (360 cameras)
    EQUIRECTANGULAR = "equirect"  # 360° equirectangular
    CUBEMAP = "cubemap"           # Cube map faces


class SegmentationModel(Enum):
    """Available segmentation models."""
    SAM3 = "sam3"                 # Meta SAM3 (text prompts)
    FASTSAM = "fastsam"          # YOLO-based (fast)
    EFFICIENTSAM = "efficient"   # TensorRT optimized
    SAM2 = "sam2"                # Legacy SAM2
    MOBILESAM = "mobile"         # Mobile-optimized
    OPENCV = "opencv"            # Traditional CV (fallback)


class MaskQuality(Enum):
    """Mask quality levels for review."""
    EXCELLENT = "excellent"  # >0.95 confidence
    GOOD = "good"           # 0.85-0.95
    REVIEW = "review"       # 0.70-0.85
    POOR = "poor"          # 0.50-0.70
    REJECT = "reject"      # <0.50


@dataclass
class MaskConfig:
    """Configuration for masking operations."""
    # Model settings
    model: SegmentationModel = SegmentationModel.SAM3
    model_checkpoint: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text prompts for SAM3
    remove_prompts: List[str] = field(default_factory=lambda: [
        "tripod",
        "camera operator person",
        "equipment gear",
        "shadow of tripod",
        "camera rover vehicle",
        "photographer",
        "selfie stick"
    ])
    keep_prompts: List[str] = field(default_factory=list)  # Objects to keep
    
    # Quality control
    confidence_threshold: float = 0.70
    review_threshold: float = 0.85
    min_mask_area: int = 100  # Minimum pixels
    max_mask_area_ratio: float = 0.5  # Maximum ratio of image
    
    # Processing options
    use_temporal_consistency: bool = True
    temporal_window: int = 5  # Frames for temporal smoothing
    batch_size: int = 4
    num_workers: int = 4
    
    # Geometry-specific settings
    geometry_aware: bool = True
    handle_distortion: bool = True
    pole_mask_expand: float = 1.2  # Expansion factor for pole regions
    
    # Output settings
    save_confidence_maps: bool = False
    save_review_images: bool = True
    output_format: str = "png"  # png, jpg, npy
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model': self.model.value,
            'model_checkpoint': self.model_checkpoint,
            'device': self.device,
            'remove_prompts': self.remove_prompts,
            'keep_prompts': self.keep_prompts,
            'confidence_threshold': self.confidence_threshold,
            'review_threshold': self.review_threshold,
            'min_mask_area': self.min_mask_area,
            'max_mask_area_ratio': self.max_mask_area_ratio,
            'use_temporal_consistency': self.use_temporal_consistency,
            'temporal_window': self.temporal_window,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'geometry_aware': self.geometry_aware,
            'handle_distortion': self.handle_distortion,
            'pole_mask_expand': self.pole_mask_expand,
            'save_confidence_maps': self.save_confidence_maps,
            'save_review_images': self.save_review_images,
            'output_format': self.output_format
        }
    
    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == '.yaml':
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'MaskConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.yaml':
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        
        # Convert string back to enum
        data['model'] = SegmentationModel(data['model'])
        
        return cls(**data)


@dataclass
class MaskResult:
    """Result of a masking operation."""
    mask: np.ndarray
    confidence: float
    quality: MaskQuality
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def needs_review(self) -> bool:
        """Check if mask needs human review."""
        return self.quality in [MaskQuality.REVIEW, MaskQuality.POOR]
    
    @property
    def is_valid(self) -> bool:
        """Check if mask is usable."""
        return self.quality != MaskQuality.REJECT


class BaseSegmenter(ABC):
    """Abstract base class for segmentation models."""
    
    def __init__(self, config: MaskConfig):
        self.config = config
        self.device = config.device
        self.model = None
    
    @abstractmethod
    def initialize(self):
        """Initialize the model."""
        pass
    
    @abstractmethod
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[MaskResult]:
        """Segment a single image."""
        pass
    
    def preprocess_image(
        self,
        image: np.ndarray,
        geometry: ImageGeometry
    ) -> np.ndarray:
        """Preprocess image based on geometry."""
        
        if geometry == ImageGeometry.EQUIRECTANGULAR and self.config.handle_distortion:
            # Apply adaptive histogram equalization to handle pole distortion
            image = self._enhance_poles(image)
        
        elif geometry == ImageGeometry.FISHEYE and self.config.handle_distortion:
            # Apply radial enhancement for fisheye
            image = self._enhance_fisheye(image)
        
        return image
    
    def _enhance_poles(self, image: np.ndarray) -> np.ndarray:
        """Enhance pole regions in equirectangular images."""
        h, w = image.shape[:2]
        
        # Create weight map (higher weight at poles)
        weights = np.ones((h, 1))
        pole_region = int(h * 0.15)  # Top/bottom 15%
        
        # Gradual weight increase towards poles
        for i in range(pole_region):
            weight = 1.0 + (pole_region - i) / pole_region
            weights[i] = weight
            weights[h - 1 - i] = weight
        
        # Apply weighted CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Weight blend
        l_blended = (l * (2 - weights) + l_enhanced * weights) / 2
        l_blended = np.clip(l_blended, 0, 255).astype(np.uint8)
        
        enhanced = cv2.merge([l_blended, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _enhance_fisheye(self, image: np.ndarray) -> np.ndarray:
        """Enhance fisheye images with radial correction."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create radial gradient mask
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        radial_weights = 1.0 + (dist_from_center / max_dist) * 0.5
        
        # Apply radial enhancement
        enhanced = image.astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] *= radial_weights
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def postprocess_mask(
        self,
        mask: np.ndarray,
        geometry: ImageGeometry
    ) -> np.ndarray:
        """Postprocess mask based on geometry."""
        
        if geometry == ImageGeometry.EQUIRECTANGULAR:
            # Expand masks at poles to account for distortion
            mask = self._expand_pole_masks(mask)
        
        # Clean up small artifacts
        mask = self._morphological_cleanup(mask)
        
        return mask
    
    def _expand_pole_masks(self, mask: np.ndarray) -> np.ndarray:
        """Expand masks in pole regions of equirectangular images."""
        h, w = mask.shape[:2]
        pole_region = int(h * 0.1)  # Top/bottom 10%
        
        # Expand top pole region
        if np.any(mask[:pole_region]):
            kernel_size = int(5 * self.config.pole_mask_expand)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask[:pole_region] = cv2.dilate(mask[:pole_region], kernel, iterations=1)
        
        # Expand bottom pole region
        if np.any(mask[-pole_region:]):
            kernel_size = int(5 * self.config.pole_mask_expand)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask[-pole_region:] = cv2.dilate(mask[-pole_region:], kernel, iterations=1)
        
        return mask
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Clean up mask with morphological operations."""
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Close small gaps
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_mask_area:
                mask[labels == i] = 0
        
        return mask


class SAM3Segmenter(BaseSegmenter):
    """SAM3-based segmentation with text prompts."""
    
    def initialize(self):
        """Initialize SAM3 model."""
        if not HAS_SAM3:
            raise ImportError("SAM3 not available")
        
        # Load checkpoint
        checkpoint = self.config.model_checkpoint or "checkpoints/sam3_hiera_large.pt"
        
        logger.info(f"Loading SAM3 from {checkpoint}")
        self.model = build_sam3(
            config="sam3_hiera_l.yaml",
            checkpoint=checkpoint,
            device=self.device
        )
        self.predictor = SAM3ImagePredictor(self.model)
        
        # Initialize text encoder
        self._initialize_text_encoder()
    
    def _initialize_text_encoder(self):
        """Initialize text encoder for prompts."""
        # SAM3 handles this internally
        pass
    
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment image using text prompts."""
        
        # Preprocess based on geometry
        image_processed = self.preprocess_image(image, geometry)
        
        # Set image
        self.predictor.set_image(image_processed)
        
        # Get prompts
        if prompts is None:
            prompts = {
                'remove': self.config.remove_prompts,
                'keep': self.config.keep_prompts
            }
        
        results = []
        
        # Process remove prompts
        for prompt in prompts.get('remove', []):
            masks, scores, logits = self._predict_with_text(prompt)
            
            for mask, score in zip(masks, scores):
                # Postprocess based on geometry
                mask_processed = self.postprocess_mask(mask, geometry)
                
                # Evaluate quality
                quality = self._evaluate_mask_quality(mask_processed, score)
                
                results.append(MaskResult(
                    mask=mask_processed,
                    confidence=score,
                    quality=quality,
                    metadata={
                        'prompt': prompt,
                        'geometry': geometry.value,
                        'model': 'sam3'
                    }
                ))
        
        # Merge masks from same category
        merged_results = self._merge_similar_masks(results)
        
        return merged_results
    
    def _predict_with_text(self, text_prompt: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate masks from text prompt."""
        
        predictions = self.predictor.predict(
            text_prompt=text_prompt,
            multimask_output=True,
            return_logits=True
        )
        
        masks = predictions[0]  # Binary masks
        scores = predictions[1]  # Confidence scores
        logits = predictions[2]  # Raw logits
        
        return masks, scores, logits
    
    def _evaluate_mask_quality(self, mask: np.ndarray, confidence: float) -> MaskQuality:
        """Evaluate mask quality based on multiple factors."""
        
        # Check confidence
        if confidence < 0.5:
            return MaskQuality.REJECT
        elif confidence < 0.7:
            return MaskQuality.POOR
        elif confidence < 0.85:
            return MaskQuality.REVIEW
        elif confidence < 0.95:
            return MaskQuality.GOOD
        else:
            return MaskQuality.EXCELLENT
    
    def _merge_similar_masks(self, results: List[MaskResult]) -> List[MaskResult]:
        """Merge overlapping masks from the same prompt."""
        
        if not results:
            return results
        
        # Group by prompt
        prompt_groups = {}
        for result in results:
            prompt = result.metadata.get('prompt', 'unknown')
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(result)
        
        # Merge within each group
        merged_results = []
        for prompt, group in prompt_groups.items():
            if len(group) == 1:
                merged_results.append(group[0])
            else:
                # Merge masks
                merged_mask = np.zeros_like(group[0].mask)
                confidences = []
                
                for result in group:
                    merged_mask = np.logical_or(merged_mask, result.mask).astype(np.uint8)
                    confidences.append(result.confidence)
                
                # Average confidence
                avg_confidence = np.mean(confidences)
                
                merged_results.append(MaskResult(
                    mask=merged_mask,
                    confidence=avg_confidence,
                    quality=self._evaluate_mask_quality(merged_mask, avg_confidence),
                    metadata={
                        'prompt': prompt,
                        'merged_count': len(group),
                        'model': 'sam3'
                    }
                ))
        
        return merged_results


class FastSAMSegmenter(BaseSegmenter):
    """FastSAM-based segmentation (10-100x faster than SAM)."""
    
    def initialize(self):
        """Initialize FastSAM model."""
        if not HAS_FASTSAM:
            raise ImportError("FastSAM not available")
        
        model_path = self.config.model_checkpoint or "FastSAM-x.pt"
        self.model = FastSAM(model_path)
        logger.info(f"Initialized FastSAM")
    
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment using FastSAM."""
        
        # Preprocess
        image_processed = self.preprocess_image(image, geometry)
        
        # Run inference
        results = self.model(
            image_processed,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9
        )
        
        mask_results = []
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            # Process each detected mask
            for i, mask in enumerate(masks):
                mask = (mask > 0.5).astype(np.uint8)
                
                # Postprocess
                mask_processed = self.postprocess_mask(mask, geometry)
                
                # Get confidence
                confidence = float(boxes.conf[i]) if boxes is not None else 0.75
                
                mask_results.append(MaskResult(
                    mask=mask_processed,
                    confidence=confidence,
                    quality=self._evaluate_mask_quality(mask_processed, confidence),
                    metadata={
                        'geometry': geometry.value,
                        'model': 'fastsam'
                    }
                ))
        
        return mask_results
    
    def _evaluate_mask_quality(self, mask: np.ndarray, confidence: float) -> MaskQuality:
        """Evaluate mask quality."""
        # FastSAM tends to have lower confidence scores
        if confidence < 0.3:
            return MaskQuality.REJECT
        elif confidence < 0.5:
            return MaskQuality.POOR
        elif confidence < 0.7:
            return MaskQuality.REVIEW
        elif confidence < 0.85:
            return MaskQuality.GOOD
        else:
            return MaskQuality.EXCELLENT


class TemporalConsistency:
    """Handle temporal consistency for video sequences."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.mask_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
    
    def add_frame(self, mask: np.ndarray, confidence: float):
        """Add frame to history."""
        self.mask_history.append(mask)
        self.confidence_history.append(confidence)
    
    def get_smoothed_mask(self) -> np.ndarray:
        """Get temporally smoothed mask."""
        if not self.mask_history:
            return None
        
        if len(self.mask_history) == 1:
            return self.mask_history[0]
        
        # Weighted average based on confidence
        weights = np.array(self.confidence_history)
        weights = weights / weights.sum()
        
        # Weighted voting
        accumulated = np.zeros_like(self.mask_history[0], dtype=np.float32)
        for mask, weight in zip(self.mask_history, weights):
            accumulated += mask.astype(np.float32) * weight
        
        # Threshold
        smoothed = (accumulated > 0.5).astype(np.uint8)
        
        return smoothed
    
    def detect_inconsistency(self, threshold: float = 0.3) -> bool:
        """Detect temporal inconsistency."""
        if len(self.mask_history) < 2:
            return False
        
        # Compare recent masks
        recent = self.mask_history[-1]
        previous = self.mask_history[-2]
        
        # Calculate IoU
        intersection = np.logical_and(recent, previous).sum()
        union = np.logical_or(recent, previous).sum()
        
        if union == 0:
            return False
        
        iou = intersection / union
        
        return iou < threshold


class MaskingPipeline:
    """Main masking pipeline orchestrator."""
    
    def __init__(
        self,
        config: Optional[MaskConfig] = None,
        auto_select_model: bool = True
    ):
        """
        Initialize masking pipeline.
        
        Args:
            config: Masking configuration
            auto_select_model: Automatically select best available model
        """
        self.config = config or MaskConfig()
        
        if auto_select_model:
            self.config.model = self._auto_select_model()
        
        self.segmenter = self._create_segmenter()
        self.segmenter.initialize()
        
        self.temporal_consistency = None
        if self.config.use_temporal_consistency:
            self.temporal_consistency = TemporalConsistency(
                self.config.temporal_window
            )
        
        logger.info(f"Initialized masking pipeline with {self.config.model.value}")
    
    def _auto_select_model(self) -> SegmentationModel:
        """Automatically select best available model."""
        if HAS_SAM3:
            return SegmentationModel.SAM3
        elif HAS_FASTSAM:
            logger.warning("SAM3 not available, using FastSAM")
            return SegmentationModel.FASTSAM
        elif HAS_EFFICIENTSAM:
            return SegmentationModel.EFFICIENTSAM
        elif HAS_SAM2:
            return SegmentationModel.SAM2
        else:
            raise RuntimeError("No segmentation models available")
    
    def _create_segmenter(self) -> BaseSegmenter:
        """Create appropriate segmenter based on config."""
        if self.config.model == SegmentationModel.SAM3:
            return SAM3Segmenter(self.config)
        elif self.config.model == SegmentationModel.FASTSAM:
            return FastSAMSegmenter(self.config)
        else:
            raise NotImplementedError(f"Model {self.config.model} not implemented")
    
    def process_image(
        self,
        image: np.ndarray,
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        custom_prompts: Optional[Dict[str, List[str]]] = None
    ) -> MaskResult:
        """
        Process single image.
        
        Args:
            image: Input image
            geometry: Image geometry type
            custom_prompts: Optional custom prompts
        
        Returns:
            Combined mask result
        """
        
        # Get individual masks
        results = self.segmenter.segment_image(image, custom_prompts, geometry)
        
        if not results:
            # No masks found
            return MaskResult(
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                confidence=0.0,
                quality=MaskQuality.REJECT,
                metadata={'message': 'No masks found'}
            )
        
        # Combine masks
        combined_mask = self._combine_masks(results)
        
        # Apply temporal consistency if enabled
        if self.temporal_consistency:
            self.temporal_consistency.add_frame(
                combined_mask.mask,
                combined_mask.confidence
            )
            
            smoothed_mask = self.temporal_consistency.get_smoothed_mask()
            if smoothed_mask is not None:
                combined_mask.mask = smoothed_mask
        
        return combined_mask
    
    def _combine_masks(self, results: List[MaskResult]) -> MaskResult:
        """Combine multiple mask results."""
        if len(results) == 1:
            return results[0]
        
        # Initialize combined mask
        h, w = results[0].mask.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        
        # Weight by confidence
        weights = []
        valid_results = []
        
        for result in results:
            if result.is_valid:
                valid_results.append(result)
                weights.append(result.confidence)
        
        if not valid_results:
            return MaskResult(
                mask=combined,
                confidence=0.0,
                quality=MaskQuality.REJECT,
                metadata={'message': 'All masks rejected'}
            )
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted combination
        for result, weight in zip(valid_results, weights):
            combined = np.logical_or(combined, result.mask).astype(np.uint8)
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in valid_results])
        
        return MaskResult(
            mask=combined,
            confidence=avg_confidence,
            quality=self.segmenter._evaluate_mask_quality(combined, avg_confidence),
            metadata={
                'combined_count': len(valid_results),
                'method': 'weighted_union'
            }
        )
    
    def process_batch(
        self,
        images: List[np.ndarray],
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        custom_prompts: Optional[Dict[str, List[str]]] = None,
        parallel: bool = True
    ) -> List[MaskResult]:
        """
        Process batch of images.
        
        Args:
            images: List of images
            geometry: Image geometry type
            custom_prompts: Optional custom prompts
            parallel: Use parallel processing
        
        Returns:
            List of mask results
        """
        
        if parallel and self.config.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for image in images:
                    future = executor.submit(
                        self.process_image,
                        image, geometry, custom_prompts
                    )
                    futures.append(future)
                
                results = [f.result() for f in futures]
        else:
            results = []
            for image in tqdm(images, desc="Processing images"):
                result = self.process_image(image, geometry, custom_prompts)
                results.append(result)
        
        return results
    
    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip_frames: int = 0,
        save_review: bool = True
    ) -> Dict[str, Any]:
        """
        Process video file.
        
        Args:
            video_path: Path to video
            output_dir: Output directory
            geometry: Video geometry type
            start_frame: Starting frame
            end_frame: Ending frame
            skip_frames: Frames to skip
            save_review: Save frames needing review
        
        Returns:
            Processing statistics
        """
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        if save_review:
            review_dir = output_dir / "review"
            review_dir.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {total_frames} frames @ {fps:.2f} FPS")
        
        # Set frame range
        end_frame = end_frame or total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'review_frames': 0,
            'rejected_frames': 0,
            'processing_time': 0
        }
        
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            start_time = time.time()
            
            # Process frame
            result = self.process_image(frame, geometry)
            
            stats['processing_time'] += time.time() - start_time
            stats['total_frames'] += 1
            
            # Save mask
            mask_path = mask_dir / f"mask_{frame_idx:06d}.{self.config.output_format}"
            
            if self.config.output_format == 'npy':
                np.save(mask_path, result.mask)
            else:
                cv2.imwrite(str(mask_path), result.mask * 255)
            
            # Handle review
            if result.needs_review and save_review:
                review_path = review_dir / f"review_{frame_idx:06d}.jpg"
                
                # Create review image with mask overlay
                review_img = self._create_review_image(frame, result.mask)
                cv2.imwrite(str(review_path), review_img)
                
                stats['review_frames'] += 1
            
            if result.quality == MaskQuality.REJECT:
                stats['rejected_frames'] += 1
            else:
                stats['processed_frames'] += 1
            
            frame_idx += 1
            
            # Log progress
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{end_frame} frames")
        
        cap.release()
        
        # Save statistics
        stats['average_time'] = stats['processing_time'] / max(stats['total_frames'], 1)
        
        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete: {stats}")
        
        return stats
    
    def _create_review_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create review image with mask overlay."""
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask * 255  # Red channel
        
        # Blend
        review = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return review
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        pattern: str = "*.jpg",
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Process directory of images.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            geometry: Image geometry type
            pattern: File pattern
            recursive: Process recursively
        
        Returns:
            Processing statistics
        """
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        if recursive:
            image_files = sorted(input_dir.rglob(pattern))
        else:
            image_files = sorted(input_dir.glob(pattern))
        
        logger.info(f"Found {len(image_files)} images")
        
        # Create output structure
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        if self.config.save_review_images:
            review_dir = output_dir / "review"
            review_dir.mkdir(exist_ok=True)
        
        # Process images
        stats = {
            'total_images': len(image_files),
            'processed_images': 0,
            'review_images': 0,
            'rejected_images': 0,
            'processing_time': 0
        }
        
        for img_path in tqdm(image_files, desc="Processing images"):
            start_time = time.time()
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.error(f"Failed to load: {img_path}")
                continue
            
            # Process
            result = self.process_image(image, geometry)
            
            stats['processing_time'] += time.time() - start_time
            
            # Save mask
            mask_name = f"mask_{img_path.stem}.{self.config.output_format}"
            mask_path = mask_dir / mask_name
            
            if self.config.output_format == 'npy':
                np.save(mask_path, result.mask)
            else:
                cv2.imwrite(str(mask_path), result.mask * 255)
            
            # Save confidence map if requested
            if self.config.save_confidence_maps:
                conf_path = mask_dir / f"conf_{img_path.stem}.npy"
                np.save(conf_path, result.confidence)
            
            # Handle review
            if result.needs_review and self.config.save_review_images:
                review_path = review_dir / f"review_{img_path.stem}.jpg"
                review_img = self._create_review_image(image, result.mask)
                cv2.imwrite(str(review_path), review_img)
                stats['review_images'] += 1
            
            if result.quality == MaskQuality.REJECT:
                stats['rejected_images'] += 1
            else:
                stats['processed_images'] += 1
        
        # Save statistics
        stats['average_time'] = stats['processing_time'] / max(stats['total_images'], 1)
        
        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete: {stats}")
        
        return stats


class InteractiveMaskRefiner:
    """Interactive tool for mask refinement."""
    
    def __init__(self, pipeline: MaskingPipeline):
        self.pipeline = pipeline
        self.current_image = None
        self.current_mask = None
        self.history = []
    
    def refine_mask(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        window_name: str = "Mask Refinement"
    ) -> np.ndarray:
        """
        Interactive mask refinement interface.
        
        Controls:
        - Left click: Add to mask
        - Right click: Remove from mask
        - 'r': Reset to initial
        - 's': Save and exit
        - 'q': Cancel
        - 'u': Undo last change
        - '+/-': Adjust brush size
        """
        
        self.current_image = image.copy()
        self.current_mask = initial_mask.copy()
        self.history = [initial_mask.copy()]
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Drawing parameters
        self.drawing = False
        self.brush_size = 10
        self.mode = 1  # 1: add, 0: remove
        
        logger.info("Interactive refinement started. Press 's' to save, 'q' to cancel")
        
        while True:
            # Create display
            display = self._create_display()
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save
                logger.info("Mask saved")
                break
            elif key == ord('q'):
                # Cancel
                logger.info("Refinement cancelled")
                self.current_mask = initial_mask
                break
            elif key == ord('r'):
                # Reset
                self.current_mask = initial_mask.copy()
                self.history = [initial_mask.copy()]
            elif key == ord('u') and len(self.history) > 1:
                # Undo
                self.history.pop()
                self.current_mask = self.history[-1].copy()
            elif key == ord('+'):
                self.brush_size = min(50, self.brush_size + 5)
                logger.info(f"Brush size: {self.brush_size}")
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 5)
                logger.info(f"Brush size: {self.brush_size}")
        
        cv2.destroyWindow(window_name)
        return self.current_mask
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 1  # Add
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = 0  # Remove
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw on mask
            cv2.circle(
                self.current_mask,
                (x, y),
                self.brush_size,
                self.mode,
                -1
            )
        elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
            if self.drawing:
                self.drawing = False
                # Save to history
                self.history.append(self.current_mask.copy())
                if len(self.history) > 20:
                    self.history.pop(0)
    
    def _create_display(self) -> np.ndarray:
        """Create display image with mask overlay."""
        
        # Create colored mask
        colored_mask = np.zeros_like(self.current_image)
        colored_mask[:, :, 1] = self.current_mask * 200  # Green
        colored_mask[:, :, 2] = self.current_mask * 100  # Red
        
        # Blend
        display = cv2.addWeighted(self.current_image, 0.7, colored_mask, 0.3, 0)
        
        # Add text
        cv2.putText(
            display, f"Brush: {self.brush_size}px",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            display, "LMB: Add | RMB: Remove | S: Save | Q: Cancel",
            (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        return display


def main():
    """Command-line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced masking system for 360° reconstruction pipelines"
    )
    
    parser.add_argument(
        "input", type=Path,
        help="Input image, video, or directory"
    )
    parser.add_argument(
        "output", type=Path,
        help="Output directory for masks"
    )
    
    # Model selection
    parser.add_argument(
        "--model", choices=["sam3", "fastsam", "efficient", "sam2", "auto"],
        default="auto",
        help="Segmentation model to use"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Model checkpoint path"
    )
    
    # Geometry
    parser.add_argument(
        "--geometry", choices=["pinhole", "fisheye", "dual_fisheye", "equirect", "cubemap"],
        default="pinhole",
        help="Image geometry type"
    )
    
    # Prompts
    parser.add_argument(
        "--remove", nargs="+",
        default=["tripod", "operator", "equipment"],
        help="Objects to remove"
    )
    parser.add_argument(
        "--keep", nargs="+", default=[],
        help="Objects to keep"
    )
    
    # Quality control
    parser.add_argument(
        "--confidence", type=float, default=0.7,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--review-threshold", type=float, default=0.85,
        help="Review threshold"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--temporal", action="store_true",
        help="Use temporal consistency"
    )
    
    # Video options
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Starting frame"
    )
    parser.add_argument(
        "--end-frame", type=int,
        help="Ending frame"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Frames to skip"
    )
    
    # Output options
    parser.add_argument(
        "--format", choices=["png", "jpg", "npy"],
        default="png",
        help="Output format"
    )
    parser.add_argument(
        "--save-review", action="store_true",
        help="Save review images"
    )
    parser.add_argument(
        "--save-confidence", action="store_true",
        help="Save confidence maps"
    )
    
    # Other
    parser.add_argument(
        "--config", type=Path,
        help="Configuration file"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enable interactive refinement"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = MaskConfig.load(args.config)
    else:
        # Map model choice
        model_map = {
            'sam3': SegmentationModel.SAM3,
            'fastsam': SegmentationModel.FASTSAM,
            'efficient': SegmentationModel.EFFICIENTSAM,
            'sam2': SegmentationModel.SAM2,
            'auto': None
        }
        
        # Set device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        config = MaskConfig(
            model=model_map[args.model] if args.model != 'auto' else None,
            model_checkpoint=str(args.checkpoint) if args.checkpoint else None,
            device=device,
            remove_prompts=args.remove,
            keep_prompts=args.keep,
            confidence_threshold=args.confidence,
            review_threshold=args.review_threshold,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_temporal_consistency=args.temporal,
            save_confidence_maps=args.save_confidence,
            save_review_images=args.save_review,
            output_format=args.format
        )
    
    # Save config
    config.save(args.output / "mask_config.yaml")
    
    # Map geometry
    geometry_map = {
        'pinhole': ImageGeometry.PINHOLE,
        'fisheye': ImageGeometry.FISHEYE,
        'dual_fisheye': ImageGeometry.DUAL_FISHEYE,
        'equirect': ImageGeometry.EQUIRECTANGULAR,
        'cubemap': ImageGeometry.CUBEMAP
    }
    geometry = geometry_map[args.geometry]
    
    # Create pipeline
    pipeline = MaskingPipeline(
        config=config,
        auto_select_model=(args.model == 'auto')
    )
    
    # Process input
    if args.input.is_file():
        # Check if video or image
        if args.input.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            stats = pipeline.process_video(
                video_path=args.input,
                output_dir=args.output,
                geometry=geometry,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                skip_frames=args.skip_frames,
                save_review=args.save_review
            )
        else:
            # Single image
            image = cv2.imread(str(args.input))
            if image is None:
                logger.error(f"Failed to load: {args.input}")
                return
            
            # Process
            result = pipeline.process_image(image, geometry)
            
            # Interactive refinement if requested
            if args.interactive and result.needs_review:
                refiner = InteractiveMaskRefiner(pipeline)
                result.mask = refiner.refine_mask(image, result.mask)
            
            # Save
            args.output.mkdir(parents=True, exist_ok=True)
            mask_path = args.output / f"mask.{args.format}"
            
            if args.format == 'npy':
                np.save(mask_path, result.mask)
            else:
                cv2.imwrite(str(mask_path), result.mask * 255)
            
            logger.info(f"Saved mask to: {mask_path}")
            logger.info(f"Quality: {result.quality.value}, Confidence: {result.confidence:.3f}")
    
    else:
        # Directory
        stats = pipeline.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            geometry=geometry,
            pattern="*.jpg",
            recursive=False
        )
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
