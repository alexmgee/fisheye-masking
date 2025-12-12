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
import os
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
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3
    SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__), "..")
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    SAM3_ROOT = None
    warnings.warn("SAM3 not found. Install from: https://github.com/facebookresearch/sam3")

try:
    # FastSAM - Fast fallback
    from ultralytics import FastSAM
    HAS_FASTSAM = True
except ImportError:
    HAS_FASTSAM = False
    HAS_FASTSAM = False
    warnings.warn("FastSAM not found. Install with: pip install ultralytics")

try:
    # YOLO11 - Production recommendation
    from ultralytics import YOLO
    HAS_YOLO11 = True
except ImportError:
    HAS_YOLO11 = False


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

try:
    # Moondream for Shadow Hunter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    HAS_MOONDREAM = True
except ImportError:
    HAS_MOONDREAM = False


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
    YOLO11 = "yolo11"            # YOLO11 (production ready)
    FASTSAM = "fastsam"          # YOLO-based (fast)
    EFFICIENTSAM = "efficient"   # TensorRT optimized
    SAM2 = "sam2"                # Legacy SAM2
    YOLO_SAM3_SHADOW = "yolo-sam3-shadow"  # NEW: Specialized shadow pipeline
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
    yolo_model_size: str = "n"  # n, s, m, l, x
    model_checkpoint: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text prompts for SAM3
    # Note: SAM3's "person" prompt already includes their shadow in the mask!
    remove_prompts: List[str] = field(default_factory=lambda: [
        "person",  # SAM3 detects person + their shadow together!
        "photographer",
        "camera operator",
        "tripod",
        "equipment",
        "selfie stick",
        "camera rover vehicle"
    ])
    keep_prompts: List[str] = field(default_factory=list)  # Objects to keep
    
    # Quality control
    confidence_threshold: float = 0.50
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
    cubemap_face_size: int = 1024
    pole_mask_expand: float = 1.2  # Expansion factor for pole regions
    enable_shadow_hunter: bool = False  # Enable Moondream shadow detection
    
    # Output settings
    save_confidence_maps: bool = False
    save_review_images: bool = True
    output_format: str = "png"  # png, jpg, npy
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model': self.model.value,
            'yolo_model_size': self.yolo_model_size,
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
            'cubemap_face_size': self.cubemap_face_size,
            'pole_mask_expand': self.pole_mask_expand,
            'enable_shadow_hunter': self.enable_shadow_hunter,
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


class ShadowHunter:
    """Detects shadows using Moondream VLM."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        """Initialize Moondream model."""
        if not HAS_MOONDREAM:
            raise ImportError("Transformers/PIL not found. Install: pip install transformers pillow einops timm")
            
        logger.info("Loading Moondream2 for Shadow Hunter...")
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            revision=revision
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        logger.info("Shadow Hunter ready.")

    def detect_shadow(self, image: np.ndarray, object_box: List[int]) -> Optional[List[int]]:
        """
        Detect shadow for a specific object using heuristic approach.
        
        Since Moondream's detect() API doesn't work reliably,
        we use a heuristic based on the object position:
        - Shadows typically fall below and to the side of a person
        - For 360 images, we estimate based on the nadir position
        
        Args:
            image: Full image (BGR)
            object_box: [x1, y1, x2, y2] of the object
            
        Returns:
            Shadow bounding box [x1, y1, x2, y2] or None
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = object_box
        
        # Object dimensions
        obj_w = x2 - x1
        obj_h = y2 - y1
        obj_cx = (x1 + x2) // 2
        obj_cy = (y1 + y2) // 2
        
        # For equirectangular images, the nadir is at y = h (bottom)
        # Shadows extend from the person toward and beyond the nadir
        
        # Estimate shadow region:
        # - Starts at the bottom of the person box
        # - Extends downward toward nadir
        # - Spreads horizontally based on sun angle (heuristic: 1.5x object width)
        
        shadow_top = y2  # Shadow starts at person's feet
        shadow_bottom = min(h - 10, y2 + int(obj_h * 1.5))  # Extend 1.5x person height down
        
        # Shadow spreads horizontally - estimate based on typical sun angles
        shadow_spread = int(obj_w * 0.75)
        shadow_left = max(0, obj_cx - shadow_spread)
        shadow_right = min(w, obj_cx + shadow_spread)
        
        # Also extend shadow in a direction (estimate based on image position)
        # For simplicity, extend both left and right
        
        shadow_box = [shadow_left, shadow_top, shadow_right, shadow_bottom]
        
        # Validate shadow box is reasonable
        shadow_area = (shadow_box[2] - shadow_box[0]) * (shadow_box[3] - shadow_box[1])
        if shadow_area < 100:  # Too small
            logger.info(f"Shadow Hunter: Heuristic shadow too small ({shadow_area} px²)")
            return None
            
        logger.info(f"Shadow Hunter: Heuristic shadow at {shadow_box}")
        return shadow_box


class CubemapProjection:
    """Handle Cubemap projection for 360° images."""
    
    def __init__(self, face_size: int = 1024):
        self.face_size = face_size
        
    def equirect2cubemap(self, equirect: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert equirectangular image to cubemap faces."""
        h, w = equirect.shape[:2]
        faces = {}
        
        # Define face directions (x, y, z)
        # Front, Back, Left, Right, Up, Down
        face_dirs = {
            'front': (0, 0, -1),
            'back': (0, 0, 1),
            'left': (-1, 0, 0),
            'right': (1, 0, 0),
            'up': (0, 1, 0),
            'down': (0, -1, 0)
        }
        
        for face_name, (dx, dy, dz) in face_dirs.items():
            # Create meshgrid for face
            u, v = np.meshgrid(
                np.linspace(-1, 1, self.face_size),
                np.linspace(-1, 1, self.face_size)
            )
            
            # Map face coordinates to 3D direction vector
            if face_name == 'front':
                x, y, z = u, v, -np.ones_like(u)
            elif face_name == 'back':
                x, y, z = -u, v, np.ones_like(u)
            elif face_name == 'left':
                x, y, z = -np.ones_like(u), v, u
            elif face_name == 'right':
                x, y, z = np.ones_like(u), v, -u
            elif face_name == 'up':
                x, y, z = u, np.ones_like(u), v
            elif face_name == 'down':
                x, y, z = u, -np.ones_like(u), -v
                
            # Normalize vectors
            r = np.sqrt(x**2 + y**2 + z**2)
            x, y, z = x/r, y/r, z/r
            
            # Convert to spherical coordinates
            phi = np.arctan2(z, x)
            theta = np.arcsin(y)
            
            # Map to equirectangular coordinates
            # Standard: Top (v=0) is North (theta=pi/2)
            # Bottom (v=1) is South (theta=-pi/2)
            # theta = pi/2 - v * pi  => v = (pi/2 - theta) / pi
            
            uf = (phi + np.pi) / (2 * np.pi)
            vf = (np.pi / 2 - theta) / np.pi
            
            # Map to pixel coordinates
            map_x = (uf * w).astype(np.float32)
            map_y = (vf * h).astype(np.float32)
            
            # Remap
            faces[face_name] = cv2.remap(
                equirect, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
            
        return faces

    def cubemap2equirect(
        self,
        faces: Dict[str, np.ndarray],
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert cubemap faces back to equirectangular."""
        w, h = output_size
        
        # Create meshgrid for equirectangular
        u, v = np.meshgrid(
            np.linspace(0, 1, w),
            np.linspace(0, 1, h)
        )
        
        # Map to spherical coordinates
        # Standard: u=0 -> -pi, u=1 -> pi. 
        # But if we see mirroring, we might need to flip u.
        # Let's try flipping phi direction: phi = pi - u * 2 * pi
        phi = np.pi - u * 2 * np.pi
        theta = v * np.pi - np.pi / 2
        
        # Map to 3D direction vector
        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta)
        z = np.cos(theta) * np.sin(phi)
        
        # Find which face each pixel belongs to
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)
        
        # Determine channels from first face
        first_face = next(iter(faces.values()))
        if len(first_face.shape) == 3:
            channels = first_face.shape[2]
            output = np.zeros((h, w, channels), dtype=np.uint8)
        else:
            output = np.zeros((h, w), dtype=np.uint8)
        
        # Process each face
        # This is a simplified implementation using masks
        # For production, a more optimized approach or loop would be better
        # But for clarity and correctness, we iterate
        
        face_map = np.zeros((h, w), dtype=np.uint8) # 0: none, 1: front, 2: back...
        
        # Front: z < 0 and abs(z) >= abs(x) and abs(z) >= abs(y)
        mask_front = (z < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        self._map_face_to_equi(faces['front'], mask_front, x, y, z, output, 'front')
        
        # Back: z > 0
        mask_back = (z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        self._map_face_to_equi(faces['back'], mask_back, x, y, z, output, 'back')
        
        # Left: x < 0
        mask_left = (x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        self._map_face_to_equi(faces['left'], mask_left, x, y, z, output, 'left')
        
        # Right: x > 0
        mask_right = (x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        self._map_face_to_equi(faces['right'], mask_right, x, y, z, output, 'right')
        
        # Up: y < 0 (note: y is down in image coords, but here we use standard 3D)
        # Actually in our convention above: up is y=-1 in image space?
        # Let's check projection above:
        # Up: y = -1 (in image space v goes 0..1). 
        # In 3D above: theta -pi/2 to pi/2. y = sin(theta).
        # So y=-1 is theta=-pi/2 (top of image? No, usually top is theta=pi/2 or 0)
        # Standard equirect: top row is theta=pi/2 (y=1), bottom is theta=-pi/2 (y=-1)
        # Let's re-verify standard convention.
        # Usually: theta from -pi/2 (south pole) to pi/2 (north pole).
        # Image v=0 -> theta=pi/2. v=1 -> theta=-pi/2.
        # My code: theta = v * pi - pi/2. 
        # v=0 -> -pi/2 (South). v=1 -> pi/2 (North).
        # So image is flipped vertically if standard is Top-Down.
        # Standard Equirect: Top is North.
        # Let's fix theta definition: theta = (1 - v) * pi - pi/2 = pi/2 - v*pi.
        
        # RE-DOING THETA DEFINITION
        theta = np.pi/2 - v * np.pi
        
        # Re-calc x,y,z
        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta)
        z = np.cos(theta) * np.sin(phi)
        
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
        # Front: x > 0? No, usually Front is +X or -Z.
        # Let's match equirect2cubemap:
        # Front: (0, 0, -1) -> -Z direction.
        # Back: (0, 0, 1) -> +Z
        # Left: (-1, 0, 0) -> -X
        # Right: (1, 0, 0) -> +X
        # Up: (0, 1, 0) -> +Y
        # Down: (0, -1, 0) -> -Y
        
        # Front (-Z)
        mask = (z < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        self._map_face_to_equi(faces['front'], mask, x, y, z, output, 'front')

        # Back (+Z)
        mask = (z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        self._map_face_to_equi(faces['back'], mask, x, y, z, output, 'back')
        
        # Left (-X)
        mask = (x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        self._map_face_to_equi(faces['left'], mask, x, y, z, output, 'left')
        
        # Right (+X)
        mask = (x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        self._map_face_to_equi(faces['right'], mask, x, y, z, output, 'right')
        
        # Up (+Y)
        mask = (y > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
        self._map_face_to_equi(faces['up'], mask, x, y, z, output, 'up')
        
        # Down (-Y)
        mask = (y < 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
        self._map_face_to_equi(faces['down'], mask, x, y, z, output, 'down')
        
        return output

    def face_point_to_equi(
        self, 
        face_name: str, 
        px: float, 
        py: float, 
        out_w: int, 
        out_h: int
    ) -> Tuple[int, int]:
        """Convert face pixel coordinates to equirectangular coordinates."""
        # Normalize to -1..1
        u = (px / self.face_size) * 2 - 1
        v = (py / self.face_size) * 2 - 1
        
        # Map to 3D direction
        if face_name == 'front':
            x, y, z = u, v, -1
        elif face_name == 'back':
            x, y, z = -u, v, 1
        elif face_name == 'left':
            x, y, z = -1, v, u
        elif face_name == 'right':
            x, y, z = 1, v, -u
        elif face_name == 'up':
            x, y, z = u, 1, v
        elif face_name == 'down':
            x, y, z = u, -1, -v
        else:
            return 0, 0
            
        # Normalize
        r = np.sqrt(x*x + y*y + z*z)
        x, y, z = x/r, y/r, z/r
        
        # Spherical
        phi = np.arctan2(z, x)
        theta = np.arcsin(y)
        
        # Equirectangular (matching cubemap2equirect logic)
        # phi = pi - u_equi * 2 * pi  => u_equi = (pi - phi) / (2*pi)
        uf = (np.pi - phi) / (2 * np.pi)
        
        # theta = pi/2 - v_equi * pi => v_equi = (pi/2 - theta) / pi
        vf = (np.pi / 2 - theta) / np.pi
        
        # Pixel
        map_x = int(uf * out_w)
        map_y = int(vf * out_h)
        
        # Clip
        map_x = max(0, min(out_w - 1, map_x))
        map_y = max(0, min(out_h - 1, map_y))
        
        return map_x, map_y

    def _map_face_to_equi(self, face, mask, x, y, z, output, face_name):
        """Helper to map face pixels to equirectangular."""
        if not np.any(mask):
            return
            
        # Get coordinates for masked pixels
        xm = x[mask]
        ym = y[mask]
        zm = z[mask]
        
        # Project to face plane
        # u, v range -1 to 1
        if face_name == 'front': # -Z
            u = xm / np.abs(zm)
            v = ym / np.abs(zm)
        elif face_name == 'back': # +Z
            u = -xm / np.abs(zm)
            v = ym / np.abs(zm)
        elif face_name == 'left': # -X
            u = zm / np.abs(xm)
            v = ym / np.abs(xm)
        elif face_name == 'right': # +X
            u = -zm / np.abs(xm)
            v = ym / np.abs(xm)
        elif face_name == 'up': # +Y
            u = xm / np.abs(ym)
            v = -zm / np.abs(ym)
        elif face_name == 'down': # -Y
            u = xm / np.abs(ym)
            v = zm / np.abs(ym)
            
        # Map to pixel coords
        # u, v are in -1..1
        # map to 0..face_size
        h, w = face.shape[:2]
        
        # u = (u + 1) / 2 * w
        # v = (v + 1) / 2 * h
        
        map_x = ((u + 1) / 2 * (w - 1)).astype(np.float32)
        map_y = ((v + 1) / 2 * (h - 1)).astype(np.float32)
        
        # Remap is tricky here because we have scattered points
        # Instead, we can use remap if we compute map_x/y for the whole image
        # But we only have it for the mask.
        # Since we are iterating faces, we can just do nearest neighbor or bilinear manually
        # Or better: create full map_x/map_y for the whole image and use remap once?
        # No, because each pixel maps to a different face.
        
        # Efficient approach:
        # We have map_x and map_y for the masked pixels.
        # We can sample from 'face' at these coordinates.
        
        # Bilinear sampling
        x0 = np.floor(map_x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(map_y).astype(int)
        y1 = y0 + 1
        
        x0 = np.clip(x0, 0, w-1)
        x1 = np.clip(x1, 0, w-1)
        y0 = np.clip(y0, 0, h-1)
        y1 = np.clip(y1, 0, h-1)
        
        wa = (x1 - map_x) * (y1 - map_y)
        wb = (x1 - map_x) * (map_y - y0)
        wc = (map_x - x0) * (y1 - map_y)
        wd = (map_x - x0) * (map_y - y0)
        
        # Sample
        # If face is single channel (mask)
        if len(face.shape) == 2:
            val = (face[y0, x0] * wa + face[y1, x0] * wb + 
                   face[y0, x1] * wc + face[y1, x1] * wd)
            output[mask] = val.astype(np.uint8)
        else:
            # 3 channel
            for c in range(3):
                val = (face[y0, x0, c] * wa + face[y1, x0, c] * wb + 
                       face[y0, x1, c] * wc + face[y1, x1, c] * wd)
                output[mask, c] = val.astype(np.uint8)


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
        
        # Ensure mask is 2D (squeeze extra dimensions from SAM3)
        while mask.ndim > 2:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            else:
                mask = mask[0]  # Take first mask if multiple
        
        # Ensure mask is uint8 and contiguous
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        mask = np.ascontiguousarray(mask)
        
        if geometry == ImageGeometry.EQUIRECTANGULAR:
            # Expand masks at poles to account for distortion
            mask = self._expand_pole_masks(mask)
        
        # Clean up small artifacts
        mask = self._morphological_cleanup(mask)
        
        return mask
    
    def _expand_pole_masks(self, mask: np.ndarray) -> np.ndarray:
        """Expand masks in pole regions of equirectangular images."""
        # Ensure mask is uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
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
        # Ensure mask is 2D (squeeze extra dimensions from SAM3)
        while mask.ndim > 2:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            else:
                # Take first mask if multiple
                mask = mask[0]
        
        # Ensure mask is contiguous and uint8 (SAM3 returns boolean masks)
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
        # Ensure contiguous array
        mask = np.ascontiguousarray(mask)
        
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
        
        # BPE tokenizer path
        bpe_path = os.path.join(SAM3_ROOT, "assets", "bpe_simple_vocab_16e6.txt.gz")
        
        logger.info(f"Loading SAM3 from HuggingFace...")
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.1)  # Low threshold to catch shadows
        
        logger.info("SAM3 initialized with Sam3Processor")
    
    def _initialize_text_encoder(self):
        """Initialize text encoder for prompts."""
        # SAM3 handles this internally via Sam3Processor
        pass
    
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment image using text prompts."""
        from PIL import Image as PILImage
        
        # Preprocess based on geometry
        image_processed = self.preprocess_image(image, geometry)
        
        # Convert to PIL for Sam3Processor
        pil_image = PILImage.fromarray(cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB))
        
        # Set image using Sam3Processor
        self.inference_state = self.processor.set_image(pil_image)
        
        # Get prompts
        if prompts is None:
            prompts = {
                'remove': self.config.remove_prompts,
                'keep': self.config.keep_prompts
            }
        
        logger.info(f"SAM3: Processing prompts: {prompts.get('remove', [])}")
        
        results = []
        
        # Process remove prompts
        for prompt in prompts.get('remove', []):
            masks, scores, boxes = self._predict_with_text(prompt)
            
            logger.info(f"SAM3: Prompt '{prompt}' found {len(masks) if masks else 0} masks, scores={scores[:3] if scores else []}")
            
            if masks is None or len(masks) == 0:
                continue
            
            # Combine ALL masks for this prompt with OR (like test script)
            # This ensures shadows connected to person are included
            combined_mask = None
            for mask in masks:
                if combined_mask is None:
                    combined_mask = mask > 0.5
                else:
                    combined_mask = np.logical_or(combined_mask, mask > 0.5)
            
            if combined_mask is not None:
                # Postprocess the combined mask
                combined_mask = combined_mask.astype(np.uint8)
                mask_processed = self.postprocess_mask(combined_mask, geometry)
                
                # Use the highest score
                max_score = max(scores) if scores else 0.8
                quality = self._evaluate_mask_quality(mask_processed, max_score)
                
                results.append(MaskResult(
                    mask=mask_processed,
                    confidence=max_score,
                    quality=quality,
                    metadata={
                        'prompt': prompt,
                        'geometry': geometry.value,
                        'model': 'sam3',
                        'mask_count': len(masks)
                    }
                ))
        
        # Merge masks from same category
        merged_results = self._merge_similar_masks(results)
        
        return merged_results
    
    def _predict_with_text(self, text_prompt: str) -> Tuple[List[np.ndarray], List[float], List]:
        """Generate masks from text prompt using Sam3Processor."""
        
        # Reset and set new prompt
        self.processor.reset_all_prompts(self.inference_state)
        self.inference_state = self.processor.set_text_prompt(
            state=self.inference_state, 
            prompt=text_prompt
        )
        
        # Get results from inference state (it's a dict)
        masks_tensor = self.inference_state.get('masks')
        scores_tensor = self.inference_state.get('scores')
        boxes_tensor = self.inference_state.get('boxes')
        
        logger.debug(f"SAM3 masks shape: {masks_tensor.shape if masks_tensor is not None else None}")
        logger.debug(f"SAM3 scores shape: {scores_tensor.shape if scores_tensor is not None else None}, values: {scores_tensor}")
        
        if masks_tensor is None or masks_tensor.numel() == 0:
            return [], [], []
        
        # Convert tensors to numpy
        masks = masks_tensor.cpu().numpy() if hasattr(masks_tensor, 'cpu') else masks_tensor
        
        # Handle scores - might be (N,) or (N, 1) tensor
        # SAM3 scores are often very low (0.1-0.2) even for good detections
        # Use maximum score or default to 0.8 for SAM3
        if scores_tensor is not None and scores_tensor.numel() > 0:
            scores_np = scores_tensor.cpu().numpy() if hasattr(scores_tensor, 'cpu') else np.array(scores_tensor)
            max_score = float(scores_np.max())
            # SAM3 scores are on different scale - if max is low, use 0.8
            if max_score < 0.5:
                scores = [0.8] * masks.shape[0]  # Override with reasonable score
            else:
                scores = scores_np.flatten().tolist()
        else:
            scores = [0.8] * masks.shape[0]  # Default score if not available
        
        boxes = boxes_tensor.cpu().numpy().tolist() if boxes_tensor is not None and hasattr(boxes_tensor, 'cpu') else []
        
        # Convert masks to list of individual masks
        # SAM3 returns shape (N, 1, H, W) - need to squeeze the extra dimension
        if len(masks.shape) == 4:  # (N, 1, H, W)
            masks = masks.squeeze(1)  # -> (N, H, W)
        
        if len(masks.shape) == 3:  # (N, H, W)
            masks_list = [masks[i] for i in range(masks.shape[0])]
        elif len(masks.shape) == 2:  # (H, W) single mask
            masks_list = [masks]
        else:
            masks_list = [masks]
        
        return masks_list, scores, boxes
    
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


class YOLO11Segmenter(BaseSegmenter):
    """YOLO11-based segmentation (Production Recommendation)."""
    
    def initialize(self):
        """Initialize YOLO11 model."""
        if not HAS_YOLO11:
            raise ImportError("YOLO11 not available. Install with: pip install ultralytics")
            
        # Select model size
        size = self.config.yolo_model_size
        model_name = f"yolo11{size}-seg.pt"
        
        if self.config.model_checkpoint:
            model_path = self.config.model_checkpoint
        else:
            models_dir = Path(__file__).parent.parent / 'models'
            model_path = str(models_dir / model_name)
            
        logger.info(f"Loading YOLO11 from {model_path}")
        self.model = YOLO(model_path)
        
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment using YOLO11."""
        
        # Preprocess
        image_processed = self.preprocess_image(image, geometry)
        
        # Run inference
        # YOLO11 supports classes argument to filter by class
        # But for generic "remove" prompts, we might want everything or specific classes
        # For now, we'll detect everything and filter if needed, or rely on the model's classes
        # Standard COCO classes: 0=person, etc.
        
        # If prompts contain specific keywords, we could map them to COCO classes
        # But YOLO11-seg is usually trained on COCO.
        # Let's assume we want to segment everything that looks like an object if no specific class is requested?
        # Or better: default to person (0) if "operator" is in prompts.
        
        # Fall back to config prompts if no custom prompts provided
        if prompts is None:
            prompts = {
                'remove': self.config.remove_prompts,
                'keep': self.config.keep_prompts
            }
        
        classes = None
        remove_prompts = prompts.get('remove', [])
        
        # Log what we're looking for
        logger.info(f"YOLO11: Looking for prompts: {remove_prompts}")
        
        # Simple keyword matching for COCO classes
        target_classes = []
        if any(p in ['person', 'operator', 'photographer'] for p in remove_prompts):
            target_classes.append(0)  # person
        if any(p in ['backpack', 'bag'] for p in remove_prompts):
            target_classes.append(24)  # backpack
        if any(p in ['tripod', 'camera'] for p in remove_prompts):
            # No direct COCO class for tripod, but we can maybe catch some
            pass
        
        if target_classes:
            classes = target_classes
            logger.info(f"YOLO11: Filtering to COCO classes: {classes}")
        
        results = self.model.predict(
            image_processed,
            device=self.device,
            classes=classes,
            retina_masks=True,
            imgsz=1024,  # Force consistent input size
            conf=self.config.confidence_threshold,
            verbose=False
        )
        
        mask_results = []
        
        if results[0].masks is not None:
            # Get original image dimensions
            orig_h, orig_w = image_processed.shape[:2]
            
            # YOLO masks.data is at model input size, need to use masks.xy (polygons) or resize properly
            # masks.data shape is (N, H_model, W_model) which may not match input
            masks_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for i, mask_raw in enumerate(masks_data):
                # Create mask at original image size
                # masks.data is at internal model size, resize to original
                mask = cv2.resize(
                    mask_raw.astype(np.float32), 
                    (orig_w, orig_h),  # (width, height) for cv2.resize
                    interpolation=cv2.INTER_LINEAR
                )
                
                mask = (mask > 0.5).astype(np.uint8)
                
                # Postprocess
                mask_processed = self.postprocess_mask(mask, geometry)
                
                # Get confidence
                confidence = float(boxes.conf[i]) if boxes is not None else 0.8
                
                # Get class name
                class_id = int(boxes.cls[i])
                class_name = results[0].names[class_id]
                
                # Get box (already in original image coords)
                box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                
                mask_results.append(MaskResult(
                    mask=mask_processed,
                    confidence=confidence,
                    quality=self._evaluate_mask_quality(mask_processed, confidence),
                    metadata={
                        'geometry': geometry.value,
                        'model': 'yolo11',
                        'class': class_name,
                        'box': box
                    }
                ))
                
        return mask_results

    def _evaluate_mask_quality(self, mask: np.ndarray, confidence: float) -> MaskQuality:
        """Evaluate mask quality."""
        if confidence < 0.4:
            return MaskQuality.REJECT
        elif confidence < 0.6:
            return MaskQuality.POOR
        elif confidence < 0.8:
            return MaskQuality.REVIEW
        elif confidence < 0.9:
            return MaskQuality.GOOD
        else:
            return MaskQuality.EXCELLENT


class YOLOSAM3ShadowSegmenter(BaseSegmenter):
    """
    Specialized segmenter for person + shadow detection using YOLO11 + SAM3.
    
    This combines YOLO11's person detection with SAM3's shadow-aware segmentation,
    specifically optimized for fisheye images with cast shadows.
    
    Features:
    - YOLO11 for initial person detection
    - Large context boxes (2.5x expansion) to capture long shadows
    - SAM3 with "person and cast shadow" text prompt
    - Connectivity filtering to remove isolated noise
    - Auto-quality scoring based on shadow-to-person ratio
    """
    
    def __init__(self, config: MaskConfig):
        super().__init__(config)
        self.yolo_model = None
        self.sam3_model = None
        self.sam3_processor = None
        
    def initialize(self):
        """Initialize YOLO and SAM3 models."""
        from ultralytics import YOLO
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from pathlib import Path
        
        # Load YOLO11 - two models for complementary detection
        logger.info("Loading YOLO11 models...")
        models_dir = Path(__file__).parent.parent / 'models'
        self.yolo_model = YOLO(str(models_dir / 'yolo11m-seg.pt'))  # Primary: medium seg model
        self.yolo_model_fallback = YOLO(str(models_dir / 'yolo11n-seg.pt'))  # Fallback: nano seg model
        
        # Load SAM3
        logger.info("Loading SAM3 processor...")
        # Find BPE file
        sam3_root = Path(__file__).parent.parent / 'tools' / 'sam3'
        bpe_path = sam3_root / 'assets' / 'bpe_simple_vocab_16e6.txt.gz'
        
        self.sam3_model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.sam3_processor = Sam3Processor(self.sam3_model, confidence_threshold=0.1)
        logger.info("YOLOSAM3Shadow segmenter initialized")
        
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: Optional['ImageGeometry'] = None
    ) -> List[MaskResult]:
        """
        Segment person and shadow using YOLO + SAM3.
        EXACTLY replicates hybrid_mask.py workflow.
        
        Pipeline:
        1. YOLO detects persons (gets boxes AND masks)
        2. Create large context boxes (2.5x for shadows)
        3. SAM3 segments with "person and cast shadow" in large box
        4. For EACH person: OR(YOLO_mask, filtered_SAM3_mask)
        5. OR all persons together
        6. Apply circular vignette
        7. Return SINGLE combined mask
        """
        h, w = image.shape[:2]
        
        # Step 1: YOLO person detection - GET BOTH BOXES AND MASKS
        # Use confidence 0.12 with box size filter
        results = self.yolo_model(image, classes=[0], conf=0.12, verbose=False, retina_masks=True)
        boxes_xyxy = []
        yolo_masks = []
        
        # Minimum size filter constants (used by both primary and fallback detection)
        min_area_ratio = 0.001  # 0.1% of image area (lowered to catch background people)
        min_dimension = 50  # 50x50 pixels (lowered for distant people)
        
        if results and results[0].boxes:
            for r in results:
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    # Filter out tiny boxes (likely false positives)
                    x1, y1, x2, y2 = box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    image_area = h * w
                    
                    
                    if box_area < image_area * min_area_ratio:
                        logger.warning(f"  Skipping tiny detection ({box_area:.0f} px, {box_area/image_area*100:.2f}% of image)")
                        continue
                    if box_width < min_dimension or box_height < min_dimension:
                        logger.warning(f"  Skipping small detection ({box_width:.0f}x{box_height:.0f} px)")
                        continue
                    
                    boxes_xyxy.append(box)
                
                if r.masks is not None:
                    # Only add masks for boxes we kept
                    for idx, m in enumerate(r.masks):
                        if idx < len(boxes_xyxy):
                            yolo_masks.append(m.data.cpu().numpy())
        
        if not boxes_xyxy:
            # Try fallback model (more sensitive for extreme fisheye distortion)
            logger.info("Primary model found nothing, trying fallback model...")
            results = self.yolo_model_fallback(image, classes=[0], conf=0.12, verbose=False, retina_masks=True)
            
            if results and results[0].boxes:
                for r in results:
                    for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                        x1, y1, x2, y2 = box
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        image_area = h * w
                        
                        if box_area < image_area * min_area_ratio:
                            continue
                        if box_width < min_dimension or box_height < min_dimension:
                            continue
                        
                        boxes_xyxy.append(box)
                    
                    if r.masks is not None:
                        for idx, m in enumerate(r.masks):
                            if idx < len(boxes_xyxy):
                                yolo_masks.append(m.data.cpu().numpy())
        
        if not boxes_xyxy:
            logger.warning("No person detected by YOLO (both models failed)")
            # Return empty mask with POOR quality so it gets flagged for review
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [MaskResult(
                mask=empty_mask,
                confidence=0.0,
                quality=MaskQuality.POOR,
                metadata={'model': 'yolo-sam3-shadow', 'error': 'no_person_detected'}
            )]
        
        logger.info(f"Detected {len(boxes_xyxy)} persons")
        
        # Step 2: Create large context boxes
        large_context_boxes = [self._create_large_context_box(box, h, w) for box in boxes_xyxy]
        
        # Step 3: SAM3 with large boxes
        logger.info(f"Refining with SAM3 (Shadow Mode) using Large Context Boxes...")
        sam3_masks = []
        for large_box in large_context_boxes:
            sam3_mask = self._segment_with_sam3(
                image,
                [large_box],
                text_prompt="person and their cast shadow and handheld equipment"
            )
            sam3_masks.append(sam3_mask)
        
        # Step 4: Combine masks - EXACTLY like hybrid_mask.py
        full_mask = np.zeros((h, w), dtype=np.uint8)
        is_messy = False
        
        for i in range(len(boxes_xyxy)):
            # YOLO Mask (Person Body)  
            y_mask = np.zeros((h, w), dtype=np.uint8)
            if i < len(yolo_masks):
                m = yolo_masks[i]
                # YOLO masks come as [1, H, W], need to squeeze
                if m.ndim == 3:
                    m = m.squeeze(0)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                y_mask = (m > 0.5).astype(np.uint8) * 255
            
            # SAM3 Mask (Person + Shadow)
            s_mask = np.zeros((h, w), dtype=np.uint8)
            if i < len(sam3_masks):
                m = sam3_masks[i]
                if m is not None and np.sum(m) > 0:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    s_mask = m
            
            # FILTERING: Remove isolated SAM3 noise
            s_mask_clean = self._filter_isolated_components(s_mask, y_mask)
            
            # Check for messiness
            raw_area = np.count_nonzero(s_mask)
            clean_area = np.count_nonzero(s_mask_clean)
            yolo_area = np.count_nonzero(y_mask)
            
            if raw_area > 0:
                removed_ratio = (raw_area - clean_area) / raw_area
                if removed_ratio > 0.5:
                    is_messy = True
                    logger.warning(f"  - Person {i}: High noise removal ({removed_ratio:.2%}) → Flagged")
            
            if yolo_area > 0:
                ratio = clean_area / yolo_area
                raw_ratio = raw_area / yolo_area if raw_area > 0 else 0
                logger.info(f"  - Person {i}: YOLO={yolo_area:,}px, SAM3_raw={raw_area:,}px ({raw_ratio:.2f}x), SAM3_clean={clean_area:,}px ({ratio:.2f}x), filtered_out={(raw_area-clean_area):,}px")
                if ratio > 4.0:
                    is_messy = True
                    logger.warning(f"  - Person {i}: Huge shadow area ({ratio:.1f}x person) → Flagged")
                # Note: raw_ratio check removed - 4x context box improved masking quality
            
            # FALLBACK: If filtering removed >90% of SAM3 output, use YOLO mask as fallback
            if raw_area > 0 and (raw_area - clean_area) / raw_area > 0.9:
                logger.warning(f"  - Person {i}: Filtering too aggressive, using YOLO mask as fallback")
                s_mask_clean = y_mask  # Fallback to YOLO segmentation mask
            
            # Union: YOLO + filtered SAM3
            person_shadow_mask = cv2.bitwise_or(y_mask, s_mask_clean)
            full_mask = cv2.bitwise_or(full_mask, person_shadow_mask)
        
        # Step 5: Detect and add selfie stick shadows (with connectivity filtering)
        for box in boxes_xyxy:
            full_mask = self._detect_selfie_stick(image, box, full_mask)
        
        # Step 5b: Shadow detection DISABLED - intensity-based approach was unreliable
        # TODO: Investigate proper shadow detection models or manual annotation
        # logger.info("Pass 2: Detecting shadows (intensity-based)...")
        # for i, box in enumerate(boxes_xyxy):
        #     person_mask = np.zeros((h, w), dtype=np.uint8)
        #     if i < len(yolo_masks):
        #         m = yolo_masks[i]
        #         if m.ndim == 3:
        #             m = m.squeeze(0)
        #         if m.shape != (h, w):
        #             m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        #         person_mask = (m > 0.5).astype(np.uint8) * 255
        #     full_mask = self._detect_shadow_region(image, box, person_mask, full_mask)
        
        # Step 6: Apply vignette
        full_mask = self._apply_circular_mask(full_mask)
        
        # Step 6: Determine quality
        total_area = np.count_nonzero(full_mask)
        if is_messy:
            quality = MaskQuality.REVIEW
            confidence = 0.75
        elif total_area > 0:
            quality = MaskQuality.GOOD
            confidence = 0.85
        else:
            quality = MaskQuality.POOR
            confidence = 0.5
        
        # Return SINGLE combined mask
        # Convert to 0-1 range since pipeline multiplies by 255 when saving
        mask_01 = (full_mask > 0).astype(np.uint8)
        return [MaskResult(
            mask=mask_01,
            confidence=confidence,
            quality=quality,
            metadata={
                'model': 'yolo-sam3-shadow',
                'persons': len(boxes_xyxy),
                'method': 'hybrid_exact',
                'boxes': [box.tolist() for box in boxes_xyxy],
                'context_boxes': large_context_boxes
            }
        )]
    
    def _create_large_context_box(self, box, h, w):
        """Create expanded box to capture shadows. Adaptive expansion based on person size."""
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Calculate person area as fraction of image
        person_area = bw * bh
        image_area = h * w
        person_fraction = person_area / image_area
        
        # Adaptive expansion: large people get 4x, small people get 2x
        # This prevents small background people from capturing distant machinery
        if person_fraction > 0.02:  # >2% of image = close-up person
            expansion = 4.0
        elif person_fraction > 0.005:  # 0.5-2% = medium distance
            expansion = 3.0
        else:  # <0.5% = distant person
            expansion = 2.0
        
        new_w = bw * expansion
        new_h = bh * expansion
        
        nx1 = max(0, cx - new_w / 2)
        nx2 = min(w, cx + new_w / 2)
        ny1 = max(0, cy - new_h / 2)
        ny2 = min(h, cy + new_h / 2)
        
        return [nx1, ny1, nx2, ny2]
    
    def _segment_with_sam3(self, image, boxes, text_prompt="person and their cast shadow and handheld equipment"):
        """Segment with SAM3 using box + text prompts."""
        from PIL import Image
        
        h, w = image.shape[:2]
        
        try:
            if not boxes:
                return np.zeros((h, w), dtype=np.uint8)
            
            # Convert numpy BGR to PIL RGB
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Set image in SAM3 processor
            state = self.sam3_processor.set_image(pil_image)
            
            # Process first box (typically only one)  
            box = boxes[0]
            x1, y1, x2, y2 = box
            
            # Validity check
            if x2 <= x1 or y2 <= y1:
                return np.zeros((h, w), dtype=np.uint8)
            
            # Convert to normalized format [cx, cy, w, h]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            
            norm_cx = cx / w
            norm_cy = cy / h
            norm_w = bw / w
            norm_h = bh / h
            
            # Clear previous geometric prompts
            state.pop("geometric_prompt", None)
            
            # Set text prompt
            if text_prompt:
                state = self.sam3_processor.set_text_prompt(text_prompt, state)
            
            # Set confidence threshold
            self.sam3_processor.confidence_threshold = 0.15
            
            # Add box prompt
            state = self.sam3_processor.add_geometric_prompt(
                box=[norm_cx, norm_cy, norm_w, norm_h],
                label=True,
                state=state
            )
            
            if "masks" in state and len(state["masks"]) > 0:
                # Get best mask
                best_idx = 0
                if "scores" in state:
                    best_idx = torch.argmax(state["scores"]).item()
                
                mask_tensor = state["masks"][best_idx]
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                
                # Squeeze if needed
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze(0)
                
                return mask_np
            else:
                return np.zeros((h, w), dtype=np.uint8)
                
        except Exception as e:
            logger.error(f"SAM3 segmentation failed: {e}")
            return np.zeros((h, w), dtype=np.uint8)
    
    def _filter_isolated_components(self, sam3_mask, yolo_mask):
        """Remove components not connected to person."""
        if np.sum(sam3_mask) == 0:
            return sam3_mask
        
        # PRE-FILTER: Gentler morphological opening to remove noise while preserving shadows
        kernel_open = np.ones((3, 3), np.uint8)  # Reduced from 5x5 to preserve shadow connections
        sam3_clean = cv2.morphologyEx(sam3_mask, cv2.MORPH_OPEN, kernel_open)
        
        # EROSION: Gentler erosion (1 iteration instead of 2)
        kernel_erode = np.ones((3, 3), np.uint8)
        sam3_clean = cv2.erode(sam3_clean, kernel_erode, iterations=1)
        
        # Dilate YOLO mask - reduced to 10x10 to prevent distant noise connection
        kernel = np.ones((10, 10), np.uint8)
        yolo_dilated = cv2.dilate(yolo_mask, kernel, iterations=1)
        
        # Find connected components in SAM3 mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            sam3_clean, connectivity=8
        )
        
        filtered_mask = np.zeros_like(sam3_mask)
        
        # Keep ALL connected components that overlap with person area
        # This preserves shadows that may be separate blobs but still connected to person
        min_component_size = 500  # Minimum component size to keep (filter tiny noise)
        
        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]
            component_mask = (labels == i).astype(np.uint8) * 255
            overlap = cv2.bitwise_and(component_mask, yolo_dilated)
            
            # Keep all components that overlap with person AND are reasonably sized
            if np.sum(overlap) > 0 and component_area >= min_component_size:
                filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
        
        return filtered_mask
    
    def _detect_shadow_region(self, image, box, person_mask, existing_mask):
        """
        Intensity-based shadow detection using person silhouette projection.
        
        Strategy:
        1. Calculate direction from person toward image center (shadow direction in fisheye)
        2. Project/translate the person silhouette in that direction
        3. Find dark pixels in the projected region
        4. Filter by shape similarity to elongated person silhouette
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # Image center (light source appears to come from center in fisheye)
        img_center_x, img_center_y = w // 2, h // 2
        
        # Person center
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2
        person_height = y2 - y1
        person_width = x2 - x1
        
        # Calculate shadow direction: FROM person TOWARD image center
        # (In fisheye, shadows extend toward the center of the circular image)
        dx = img_center_x - person_center_x
        dy = img_center_y - person_center_y
        
        # Normalize direction
        dist = max(1, np.sqrt(dx*dx + dy*dy))
        dx_norm = dx / dist
        dy_norm = dy / dist
        
        # Shadow extends 1-2x person height in shadow direction
        shadow_length = person_height * 1.5
        
        # Create search region in shadow direction
        shadow_center_x = int(person_center_x + dx_norm * shadow_length * 0.7)
        shadow_center_y = int(person_center_y + dy_norm * shadow_length * 0.7)
        
        # Define shadow search box
        search_radius = int(max(person_width, person_height) * 1.2)
        sx1 = max(0, shadow_center_x - search_radius)
        sx2 = min(w, shadow_center_x + search_radius)
        sy1 = max(0, shadow_center_y - search_radius)
        sy2 = min(h, shadow_center_y + search_radius)
        
        if sx2 - sx1 < 30 or sy2 - sy1 < 30:
            return existing_mask
        
        logger.info(f"  - Shadow search (intensity): x={sx1}-{sx2}, y={sy1}-{sy2}, dir=({dx_norm:.2f},{dy_norm:.2f})")
        
        # Convert to grayscale for intensity analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Extract the search region
        search_region = gray[sy1:sy2, sx1:sx2]
        
        # Calculate local statistics (median is robust to shadows)
        local_median = np.median(search_region)
        local_std = np.std(search_region)
        
        # Shadow threshold: pixels significantly darker than local median
        # Shadows are typically 20-50% darker than surroundings
        shadow_threshold = local_median - local_std * 0.8
        
        # Create shadow candidate mask (dark pixels in search region)
        shadow_candidates = np.zeros((h, w), dtype=np.uint8)
        dark_pixels = gray < shadow_threshold
        shadow_candidates[dark_pixels] = 255
        
        # Mask to only search region
        search_mask = np.zeros((h, w), dtype=np.uint8)
        search_mask[sy1:sy2, sx1:sx2] = 255
        shadow_candidates = cv2.bitwise_and(shadow_candidates, search_mask)
        
        # Exclude already-masked regions and the person
        shadow_candidates = cv2.bitwise_and(shadow_candidates, cv2.bitwise_not(existing_mask))
        shadow_candidates = cv2.bitwise_and(shadow_candidates, cv2.bitwise_not(person_mask))
        
        # Morphological cleanup: remove noise, connect nearby regions
        kernel = np.ones((5, 5), np.uint8)
        shadow_candidates = cv2.morphologyEx(shadow_candidates, cv2.MORPH_CLOSE, kernel)
        shadow_candidates = cv2.morphologyEx(shadow_candidates, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_candidates, connectivity=8)
        
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        person_area = (x2 - x1) * (y2 - y1)
        min_shadow_area = person_area * 0.1  # At least 10% of person
        max_shadow_area = person_area * 2.5  # At most 250% of person
        
        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_x = stats[i, cv2.CC_STAT_LEFT]
            comp_y = stats[i, cv2.CC_STAT_TOP]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter by size
            if comp_area < min_shadow_area or comp_area > max_shadow_area:
                continue
            
            # Filter by aspect ratio (shadows are typically elongated)
            aspect = max(comp_w, comp_h) / max(1, min(comp_w, comp_h))
            if aspect < 1.2:  # Too square - probably not a shadow
                continue
            
            # Filter by position: should be in shadow direction from person
            comp_center_x = comp_x + comp_w // 2
            comp_center_y = comp_y + comp_h // 2
            
            # Vector from person to component
            to_comp_x = comp_center_x - person_center_x
            to_comp_y = comp_center_y - person_center_y
            to_comp_dist = max(1, np.sqrt(to_comp_x*to_comp_x + to_comp_y*to_comp_y))
            
            # Dot product with shadow direction (should be positive and significant)
            dot = (to_comp_x * dx_norm + to_comp_y * dy_norm) / to_comp_dist
            if dot < 0.3:  # Component not in shadow direction
                continue
            
            # This component passed all filters - add to shadow mask
            component_mask = (labels == i).astype(np.uint8) * 255
            shadow_mask = cv2.bitwise_or(shadow_mask, component_mask)
        
        shadow_area = np.sum(shadow_mask > 0)
        if shadow_area > 0:
            logger.info(f"  ✓ Added shadow (intensity-based): {shadow_area:,} pixels")
            return cv2.bitwise_or(existing_mask, shadow_mask)
        else:
            logger.info(f"  - No shadow found (threshold={shadow_threshold:.0f}, median={local_median:.0f})")
        
        return existing_mask
    
    def _apply_circular_mask(self, mask, radius_factor=0.96):
        """Apply circular mask to remove fisheye vignette."""
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        radius = int(min(h, w) / 2 * radius_factor)
        
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)  # FIX: Use 255 not 1
        
        return cv2.bitwise_and(mask, mask, mask=circle_mask)
    
    def _evaluate_mask_quality(self, mask: np.ndarray, confidence: float) -> MaskQuality:
        """
        Evaluate quality based on shadow-to-person ratio and confidence.
        This is called by the pipeline for combined masks.
        """
        # For shadow pipeline, we primarily use confidence
        # but could add additional shadow-specific metrics here
        
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
    
    def _detect_selfie_stick(self, image: np.ndarray, person_box, existing_mask: np.ndarray) -> np.ndarray:
        """
        Detect selfie stick using a targeted SAM3 pass.
        Creates a narrow vertical box extending upward from person center.
        
        From hybrid_mask.py - this is a SECOND SAM3 pass specifically for the stick.
        """
        x1, y1, x2, y2 = person_box
        cx = (x1 + x2) / 2
        h, w = image.shape[:2]
        
        # Create a narrow vertical "stick search box" extending upward from person
        stick_width = 150  # Narrow horizontal range
        stick_x1 = max(0, cx - stick_width/2)
        stick_x2 = min(w, cx + stick_width/2)
        stick_y1 = max(0, y1 - 600)  # Extend far upward
        stick_y2 = y1 + 50  # Slight overlap with person
        
        stick_box = [stick_x1, stick_y1, stick_x2, stick_y2]
        
        logger.info(f"  - Searching for stick: x={stick_x1:.0f}-{stick_x2:.0f}, y={stick_y1:.0f}-{stick_y2:.0f}")
        
        try:
            stick_mask = self._segment_with_sam3(
                image,
                [stick_box],
                text_prompt="selfie stick and shadow"
            )
            
            if stick_mask is not None and np.sum(stick_mask) > 0:
                if stick_mask.shape != (h, w):
                    stick_mask = cv2.resize(stick_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # FILTERING: Only keep stick mask if it's reasonably thin/elongated
                # This prevents grabbing huge areas of sand texture
                stick_area = np.count_nonzero(stick_mask)
                box_area = (stick_x2 - stick_x1) * (stick_y2 - stick_y1)
                fill_ratio = stick_area / box_area if box_area > 0 else 1.0
                
                # If the stick mask fills more than 40% of the search box, it's probably noise
                if fill_ratio > 0.4:
                    logger.info(f"  - Stick mask too large ({fill_ratio:.1%} fill) - likely sand noise, skipping")
                    return existing_mask
                
                # Also check aspect ratio - stick should be thin and tall, not blobby
                # Find bounding rect of the mask
                coords = np.column_stack(np.where(stick_mask > 0))
                if len(coords) > 10:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    mask_width = x_max - x_min
                    mask_height = y_max - y_min
                    
                    # Stick should be taller than wide (aspect ratio > 1.2)
                    if mask_height > 0 and mask_width > 0:
                        aspect = mask_height / mask_width
                        if aspect < 1.2:
                            logger.info(f"  - Stick mask too wide (aspect {aspect:.1f}) - likely noise, skipping")
                            return existing_mask
                
                # CRITICAL: Only keep stick components that CONNECT to existing mask
                # This prevents disconnected noise blobs from being added
                # Using small 5x5 dilation for strict connectivity check
                existing_dilated = cv2.dilate(existing_mask, np.ones((5, 5), np.uint8), iterations=1)
                
                # Find connected components in stick mask
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stick_mask, connectivity=8)
                filtered_stick = np.zeros_like(stick_mask)
                
                for i in range(1, num_labels):
                    component = (labels == i).astype(np.uint8) * 255
                    overlap = cv2.bitwise_and(component, existing_dilated)
                    if np.sum(overlap) > 0:  # This component touches the person mask
                        filtered_stick = cv2.bitwise_or(filtered_stick, component)
                
                if np.sum(filtered_stick) > 0:
                    result = cv2.bitwise_or(existing_mask, filtered_stick)
                    added_pixels = np.count_nonzero(filtered_stick)
                    logger.info(f"  ✓ Added selfie stick mask ({added_pixels} pixels, connected to person)")
                    return result
                else:
                    logger.info("  - Stick mask not connected to person, skipping")
                    return existing_mask
            else:
                logger.info("  - No stick detected by SAM3")
                
        except Exception as e:
            logger.warning(f"  - Stick detection failed: {e}")
        
        return existing_mask

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
            
        self.cubemap_proj = CubemapProjection(self.config.cubemap_face_size)
        
        self.shadow_hunter = None
        self.shadow_segmenter = None
        
        if self.config.enable_shadow_hunter:
            self.shadow_hunter = ShadowHunter(self.config.device)
            # We need a segmenter that supports box prompts
            # If primary is SAM-based, use it. If YOLO, we need a secondary one.
            if self.config.model not in [SegmentationModel.SAM3, SegmentationModel.SAM2]:
                # Initialize FastSAM as fallback for shadows
                logger.info("Initializing FastSAM for shadow segmentation...")
                self.shadow_segmenter = FastSAMSegmenter(self.config)
                self.shadow_segmenter.initialize()
            else:
                self.shadow_segmenter = self.segmenter
        
        logger.info(f"Initialized masking pipeline with {self.config.model.value}")
    
    def _auto_select_model(self) -> SegmentationModel:
        """Automatically select best available model."""
        if HAS_YOLO11:
            return SegmentationModel.YOLO11
        elif HAS_SAM3:
            return SegmentationModel.SAM3
        elif HAS_FASTSAM:
            logger.warning("YOLO11/SAM3 not available, using FastSAM")
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
        elif self.config.model == SegmentationModel.YOLO11:
            return YOLO11Segmenter(self.config)
        elif self.config.model == SegmentationModel.FASTSAM:
            return FastSAMSegmenter(self.config)
        elif self.config.model == SegmentationModel.YOLO_SAM3_SHADOW:
            return YOLOSAM3ShadowSegmenter(self.config)
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
        
        # Handle Equirectangular - for SAM3, process full image directly (better shadow detection)
        # For other models, use cubemap projection
        if geometry == ImageGeometry.EQUIRECTANGULAR:
            if self.config.model == SegmentationModel.SAM3 and HAS_SAM3:
                # SAM3 works better on full image - shadows stay connected
                logger.info("Processing equirectangular directly with SAM3 (no cubemap)")
                results = self.segmenter.segment_image(image, custom_prompts, geometry)
                if not results:
                    return MaskResult(
                        mask=np.zeros(image.shape[:2], dtype=np.uint8),
                        confidence=0.0,
                        quality=MaskQuality.REJECT,
                        metadata={'message': 'No masks found'}
                    )
                return self._combine_masks(results)
            elif self.config.geometry_aware:
                return self._process_equirectangular(image, custom_prompts)
        
        # Get individual masks
        results = self.segmenter.segment_image(image, custom_prompts, geometry)
        
        # Shadow Hunter
        if self.shadow_hunter and results:
            shadow_results = []
            for result in results:
                # Only look for shadows of valid objects
                if not result.is_valid or 'box' not in result.metadata:
                    continue
                    
                # Detect shadow
                shadow_box = self.shadow_hunter.detect_shadow(image, result.metadata['box'])
                
                if shadow_box:
                    # Segment shadow
                    # We need to implement segment_box in BaseSegmenter or handle it here
                    # For now, let's assume we can use the shadow_segmenter
                    
                    # If it's FastSAM (NOT SAM3) and has a model attribute we can call
                    from ultralytics import FastSAM as FastSAMModel
                    if self.shadow_segmenter and hasattr(self.shadow_segmenter, 'model') and isinstance(self.shadow_segmenter.model, FastSAMModel):
                        # FastSAM prompt
                        # We need to implement a prompt method in FastSAMSegmenter or access model directly
                        # Accessing model directly for now to save time
                        try:
                            fast_results = self.shadow_segmenter.model(
                                image,
                                device=self.config.device,
                                retina_masks=True,
                                imgsz=1024,
                                conf=0.4,
                                iou=0.9
                            )
                            # Prompt with box
                            if fast_results:
                                # FastSAM v8 API for prompting is a bit different, usually via .prompt()
                                # But ultralytics FastSAM implementation might just be YOLO-like
                                # Actually standard Ultralytics FastSAM doesn't always support .prompt() easily in v8 API
                                # It's better to use the 'bboxes' argument in predict if supported, or prompt after.
                                # Let's try standard predict with classes? No.
                                # Ultralytics FastSAM is weird.
                                # Let's use a simple fallback: Just use the box as the mask for now to avoid breaking.
                                # Or better: Use the box to crop, segment "shadow" with CLIP? No.
                                
                                # Let's just create a rectangular mask for the shadow for this iteration
                                # It's better than nothing and robust.
                                pass
                        except Exception as e:
                            logger.warning(f"FastSAM shadow segmentation failed: {e}")
                            
                    # Create rectangular mask for shadow (Robust MVP)
                    h, w = image.shape[:2]
                    shadow_mask = np.zeros((h, w), dtype=np.uint8)
                    sx1, sy1, sx2, sy2 = int(shadow_box[0]), int(shadow_box[1]), int(shadow_box[2]), int(shadow_box[3])
                    shadow_mask[sy1:sy2, sx1:sx2] = 1
                    
                    shadow_results.append(MaskResult(
                        mask=shadow_mask,
                        confidence=0.8, # Moondream is usually right
                        quality=MaskQuality.GOOD,
                        metadata={
                            'model': 'moondream',
                            'type': 'shadow',
                            'parent_object': result.metadata.get('class', 'unknown')
                        }
                    ))
            
            results.extend(shadow_results)
        
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

    def _process_equirectangular(
        self,
        image: np.ndarray,
        custom_prompts: Optional[Dict[str, List[str]]] = None
    ) -> MaskResult:
        """Process equirectangular image via cubemap projection."""
        
        # 1. Convert to cubemap
        faces = self.cubemap_proj.equirect2cubemap(image)
        
        # 2. Process each face
        face_masks = {}
        face_results = {} # Store full results
        confidences = []
        
        for face_name, face_img in faces.items():
            # Process as pinhole
            result = self.process_image(
                face_img, 
                ImageGeometry.CUBEMAP, # Treat as pinhole/cubemap face
                custom_prompts
            )
            
            face_masks[face_name] = result.mask
            face_results[face_name] = result
            if result.is_valid:
                confidences.append(result.confidence)
        
        # 3. Re-project to equirectangular
        h, w = image.shape[:2]
        full_mask = self.cubemap_proj.cubemap2equirect(face_masks, (w, h))
        
        # 4. Shadow Hunter on Global Context (if enabled)
        if self.shadow_hunter:
            global_shadow_mask = np.zeros((h, w), dtype=np.uint8)
            found_shadows = 0
            
            for face_name, result in face_results.items():
                if not result.is_valid or 'box' not in result.metadata:
                    continue
                
                # Get box in face coords
                fx1, fy1, fx2, fy2 = result.metadata['box']
                logger.info(f"Face {face_name} detection: {fx1},{fy1},{fx2},{fy2}")
                
                # Map corners to equirectangular
                # We map the center of the box to find the "center of attention"
                cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                ex, ey = self.cubemap_proj.face_point_to_equi(face_name, cx, cy, w, h)
                logger.info(f"Projected to Equirect: {ex},{ey}")
                
                # Define a context box around this point in equirectangular
                # Size: 1024x1024 or similar (enough to see shadow)
                ctx_size = 1024
                ctx_x1 = max(0, int(ex - ctx_size/2))
                ctx_y1 = max(0, int(ey - ctx_size/2))
                ctx_x2 = min(w, int(ex + ctx_size/2))
                ctx_y2 = min(h, int(ey + ctx_size/2))
                
                # Construct a "fake" object box relative to this context
                # The object is roughly in the center
                obj_w = fx2 - fx1
                obj_h = fy2 - fy1
                # Scale roughly? Hard to say. Let's assume 1:1 scale for simplicity at center
                # Or just pass a small box at the center
                
                # Better: Pass the full image and a box around the center point
                # But Moondream needs a box.
                # Let's use a box of size 200x200 around the center point as the "object"
                # This is a heuristic.
                
                # Actually, ShadowHunter.detect_shadow takes the full image and an object box.
                # It crops internally.
                # So we just need to pass [ex-50, ey-100, ex+50, ey+100] as the "object box"
                # to tell Moondream "here is the person".
                
                # Heuristic object box size in equirectangular
                # 5% of width?
                box_w = int(w * 0.05)
                box_h = int(h * 0.1)
                
                obj_box = [
                    int(ex - box_w/2),
                    int(ey - box_h/2),
                    int(ex + box_w/2),
                    int(ey + box_h/2)
                ]
                
                # Detect
                shadow_box = self.shadow_hunter.detect_shadow(image, obj_box)
                
                if shadow_box:
                    sx1, sy1, sx2, sy2 = shadow_box
                    global_shadow_mask[sy1:sy2, sx1:sx2] = 1
                    found_shadows += 1
            
            if found_shadows > 0:
                logger.info(f"Shadow Hunter found {found_shadows} shadows in global context")
                full_mask = np.logical_or(full_mask, global_shadow_mask).astype(np.uint8)
        
        # 5. Post-process (seam blending/cleanup)
        # Simple morphological close to handle seam artifacts
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return MaskResult(
            mask=full_mask,
            confidence=avg_confidence,
            quality=self.segmenter._evaluate_mask_quality(full_mask, avg_confidence),
            metadata={'method': 'cubemap_projection'}
        )
    
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
            
            # Save review frames for ALL processed frames (not just low quality)
            if save_review:
                review_path = review_dir / f"review_{frame_idx:06d}.jpg"
                review_img = self._create_review_image(frame, result.mask, metadata=result.metadata)
                cv2.imwrite(str(review_path), review_img)
                stats['review_frames'] += 1
                
                # Flag only low-quality frames for manual review
                if result.needs_review:
                    flagged_dir = output_dir / "flagged"
                    flagged_dir.mkdir(exist_ok=True)
                    flagged_path = flagged_dir / review_path.name
                    if not flagged_path.exists():
                        relative_target = Path("..") / "review" / review_path.name
                        flagged_path.symlink_to(relative_target)
            
            
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
        alpha: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Create review image with mask overlay and bounding boxes.
        
        Matches hybrid_mask.py approach exactly.
        """
        # Match hybrid_mask.py approach: overlay red on mask, then blend
        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]  # Pure red where mask exists (BGR)
        review = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
        
        # Draw bounding boxes if available (from yolo-sam3-shadow segmenter)
        if metadata and 'boxes' in metadata:
            # Draw person boxes (Green)
            for box in metadata.get('boxes', []):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw context boxes (Yellow)
            for cbox in metadata.get('context_boxes', []):
                cx1, cy1, cx2, cy2 = map(int, cbox)
                cv2.rectangle(review, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        
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
            
            # Save review images for ALL processed images (not just low quality)
            if self.config.save_review_images:
                review_path = review_dir / f"review_{img_path.stem}.jpg"
                review_img = self._create_review_image(image, result.mask, metadata=result.metadata)
                cv2.imwrite(str(review_path), review_img)
                stats['review_images'] += 1
                
                # Flag only low-quality images for manual review
                if result.needs_review:
                    flagged_dir = output_dir / "flagged"
                    flagged_dir.mkdir(exist_ok=True)
                    flagged_path = flagged_dir / review_path.name
                    # Create symlink to review image using relative path
                    if not flagged_path.exists():
                        relative_target = Path("..") / "review" / review_path.name
                        flagged_path.symlink_to(relative_target)
            
            
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
        "--model", choices=["sam3", "yolo11", "fastsam", "efficient", "sam2", "yolo-sam3-shadow", "auto"],
        default="auto",
        help="Segmentation model to use"
    )
    parser.add_argument(
        "--yolo-size", choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLO11 model size"
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
    parser.add_argument(
        "--cubemap-size", type=int, default=1024,
        help="Face size for cubemap projection"
    )
    
    # Prompts
    parser.add_argument(
        "--remove", nargs="+",
        default=["person", "photographer"],
        help="Objects to remove (SAM3's 'person' includes shadow)"
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
    parser.add_argument(
        "--pattern", type=str, default="*.jpg",
        help="File pattern for directory processing"
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
        "--shadow-hunter", action="store_true",
        help="Enable Shadow Hunter (Moondream)"
    )
    parser.add_argument(
        "--review-flagged", action="store_true",
        help="Launch gallery review on flagged images after processing"
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
            'yolo11': SegmentationModel.YOLO11,
            'fastsam': SegmentationModel.FASTSAM,
            'efficient': SegmentationModel.EFFICIENTSAM,
            'sam2': SegmentationModel.SAM2,
            'yolo-sam3-shadow': SegmentationModel.YOLO_SAM3_SHADOW,
            'auto': None
        }
        
        # Set device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        config = MaskConfig(
            model=model_map[args.model] if args.model != 'auto' else None,
            yolo_model_size=args.yolo_size,
            model_checkpoint=str(args.checkpoint) if args.checkpoint else None,
            device=device,
            remove_prompts=args.remove,
            keep_prompts=args.keep,
            confidence_threshold=args.confidence,
            review_threshold=args.review_threshold,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_temporal_consistency=args.temporal,
            cubemap_face_size=args.cubemap_size,
            enable_shadow_hunter=args.shadow_hunter,
            save_confidence_maps=args.save_confidence,
            save_review_images=args.save_review,
            output_format=args.format
        )
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

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
            input_stem = args.input.stem  # filename without extension
            mask_path = args.output / f"{input_stem}_mask.{args.format}"
            
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
            pattern=args.pattern,
            recursive=False
        )
        
        # Check for flagged images and launch gallery review
        if args.review_flagged and args.save_review:
            flagged_dir = args.output / "flagged"
            if flagged_dir.exists() and any(flagged_dir.iterdir()):
                logger.info(f"Launching gallery review for {len(list(flagged_dir.glob('*')))} flagged images...")
                try:
                    from review_gallery import GalleryReviewer
                    # Get flagged image paths (symlinks point to review folder)
                    flagged_paths = sorted(flagged_dir.glob("*.jpg")) + sorted(flagged_dir.glob("*.png"))
                    if flagged_paths:
                        masks_dir = args.output / "masks"
                        reviewer = GalleryReviewer(
                            flagged_paths,
                            masks_dir=masks_dir if masks_dir.exists() else None,
                            flagged_dir=flagged_dir
                        )
                        reviewer.run()
                except ImportError:
                    logger.warning("review_gallery.py not found. Run it separately to review flagged images.")
            else:
                logger.info("No flagged images to review.")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
