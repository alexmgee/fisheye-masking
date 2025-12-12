#!/usr/bin/env python3
"""
Integration Example: Reframe + Masking Pipeline
===============================================
Shows how to combine reframe_v2.py and masking_v2.py
for a complete 360° preprocessing workflow.
"""

from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Optional
import json
import logging

# Import our modules
from reframe_v2 import (
    EquirectToRig, RigGenerator, RigConfig, 
    ProjectionBackend, ImageGeometry as ReframeGeometry
)
from masking_v2 import (
    MaskingPipeline, MaskConfig, SegmentationModel,
    ImageGeometry as MaskGeometry, MaskQuality
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedPreprocessor:
    """
    Integrated pipeline for 360° video preprocessing.
    Combines reframing and masking for COLMAP-ready output.
    """
    
    def __init__(
        self,
        rig_pattern: str = "ring12",
        masking_model: str = "sam3",
        use_gpu: bool = True
    ):
        """
        Initialize integrated preprocessor.
        
        Args:
            rig_pattern: Rig pattern (cube, ring8, ring12, geodesic)
            masking_model: Masking model (sam3, fastsam, auto)
            use_gpu: Use GPU acceleration
        """
        
        # Setup reframing
        if rig_pattern == "cube":
            rig_config = RigGenerator.create_cube_rig()
        elif rig_pattern == "ring8":
            rig_config = RigGenerator.create_ring_rig(8)
        elif rig_pattern == "ring12":
            rig_config = RigGenerator.create_ring_rig(12)
        elif rig_pattern == "geodesic":
            rig_config = RigGenerator.create_geodesic_rig()
        else:
            raise ValueError(f"Unknown rig pattern: {rig_pattern}")
        
        self.reframer = EquirectToRig(
            rig_config=rig_config,
            backend=ProjectionBackend.TORCH_GPU if use_gpu else ProjectionBackend.OPENCV,
            num_workers=8
        )
        
        # Setup masking
        mask_config = MaskConfig(
            model=SegmentationModel.SAM3 if masking_model == "sam3" else None,
            device="cuda" if use_gpu else "cpu",
            remove_prompts=[
                "tripod",
                "camera operator person",
                "equipment gear",
                "shadow of tripod",
                "selfie stick",
                "photographer"
            ],
            confidence_threshold=0.70,
            review_threshold=0.85,
            use_temporal_consistency=True,
            geometry_aware=True,
            save_review_images=True
        )
        
        self.masker = MaskingPipeline(
            config=mask_config,
            auto_select_model=(masking_model == "auto")
        )
        
        logger.info(f"Initialized with {rig_pattern} rig and {masking_model} masking")
    
    def process_360_image(
        self,
        equirect_path: Path,
        output_dir: Path
    ) -> Dict:
        """
        Process single 360° image.
        
        Args:
            equirect_path: Path to equirectangular image
            output_dir: Output directory
        
        Returns:
            Processing statistics
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        equirect = cv2.imread(str(equirect_path))
        if equirect is None:
            raise ValueError(f"Failed to load: {equirect_path}")
        
        logger.info(f"Processing: {equirect_path.name}")
        
        # Step 1: Reframe to pinhole views
        logger.info("Step 1: Reframing to pinhole views...")
        views = self.reframer.process_image(equirect)
        
        # Save pinhole views and masks
        pinhole_dir = output_dir / "pinhole_views"
        mask_dir = output_dir / "masks"
        pinhole_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        
        stats = {
            'total_views': len(views),
            'masked_views': 0,
            'review_needed': 0,
            'rejected': 0
        }
        
        # Step 2: Mask each pinhole view
        logger.info("Step 2: Masking pinhole views...")
        
        for camera_name, view_image in views.items():
            # Save pinhole view
            view_path = pinhole_dir / f"{camera_name}.jpg"
            cv2.imwrite(str(view_path), view_image)
            
            # Generate mask
            result = self.masker.process_image(
                view_image,
                geometry=MaskGeometry.PINHOLE
            )
            
            # Save mask
            mask_path = mask_dir / f"{camera_name}_mask.png"
            cv2.imwrite(str(mask_path), result.mask * 255)
            
            # Update statistics
            if result.quality != MaskQuality.REJECT:
                stats['masked_views'] += 1
            if result.needs_review:
                stats['review_needed'] += 1
            if result.quality == MaskQuality.REJECT:
                stats['rejected'] += 1
            
            logger.info(f"  {camera_name}: Quality={result.quality.value}, "
                       f"Confidence={result.confidence:.2f}")
        
        # Step 3: Save metadata
        metadata = {
            'source': str(equirect_path),
            'rig_config': self.reframer.rig_config.to_dict(),
            'statistics': stats
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Complete: {stats['masked_views']}/{stats['total_views']} "
                   f"views masked, {stats['review_needed']} need review")
        
        return stats
    
    def process_360_video(
        self,
        video_path: Path,
        output_dir: Path,
        skip_frames: int = 0,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Process 360° video with full pipeline.
        
        Args:
            video_path: Path to video
            output_dir: Output directory
            skip_frames: Frames to skip (0 for all)
            max_frames: Maximum frames to process
        
        Returns:
            Processing statistics
        """
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        frames_dir = output_dir / "frames"
        pinhole_dir = output_dir / "pinhole_views"
        masks_dir = output_dir / "masks"
        review_dir = output_dir / "review"
        
        for d in [frames_dir, pinhole_dir, masks_dir, review_dir]:
            d.mkdir(exist_ok=True)
        
        # Create camera subdirectories
        for camera in self.reframer.rig_config.cameras:
            (pinhole_dir / camera.name).mkdir(exist_ok=True)
            (masks_dir / camera.name).mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {total_frames} frames @ {fps:.2f} FPS")
        
        # Statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_masks': 0,
            'review_masks': 0,
            'rejected_masks': 0
        }
        
        frame_idx = 0
        processed = 0
        
        while True:
            ret, equirect = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            # Check max frames
            if max_frames and processed >= max_frames:
                break
            
            logger.info(f"Processing frame {frame_idx}/{total_frames}")
            
            # Step 1: Reframe
            views = self.reframer.process_image(equirect)
            
            # Step 2: Process each view
            frame_name = f"frame_{processed:06d}"
            
            for camera_name, view_image in views.items():
                # Save pinhole view
                view_path = pinhole_dir / camera_name / f"{frame_name}.jpg"
                cv2.imwrite(str(view_path), view_image)
                
                # Generate mask
                result = self.masker.process_image(
                    view_image,
                    geometry=MaskGeometry.PINHOLE
                )
                
                # Save mask
                mask_path = masks_dir / camera_name / f"{frame_name}_mask.png"
                cv2.imwrite(str(mask_path), result.mask * 255)
                
                # Update statistics
                stats['total_masks'] += 1
                if result.needs_review:
                    stats['review_masks'] += 1
                    
                    # Save review image
                    review_img = self._create_review_image(view_image, result.mask)
                    review_path = review_dir / f"{camera_name}_{frame_name}.jpg"
                    cv2.imwrite(str(review_path), review_img)
                
                if result.quality == MaskQuality.REJECT:
                    stats['rejected_masks'] += 1
            
            processed += 1
            stats['processed_frames'] += 1
            frame_idx += 1
        
        cap.release()
        
        stats['total_frames'] = frame_idx
        
        # Save configuration and statistics
        config = {
            'video': str(video_path),
            'rig_config': self.reframer.rig_config.to_dict(),
            'mask_config': self.masker.config.to_dict(),
            'statistics': stats
        }
        
        with open(output_dir / "pipeline_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save rig configuration for COLMAP
        self.reframer.rig_config.save(output_dir / "rig_config.json")
        
        logger.info(f"Complete: Processed {stats['processed_frames']} frames")
        logger.info(f"Generated {stats['total_masks']} masks")
        logger.info(f"Review needed: {stats['review_masks']} masks")
        
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
    
    def prepare_for_colmap(
        self,
        processed_dir: Path,
        colmap_dir: Path
    ):
        """
        Prepare processed data for COLMAP reconstruction.
        
        Args:
            processed_dir: Directory with processed data
            colmap_dir: Output directory for COLMAP
        """
        
        processed_dir = Path(processed_dir)
        colmap_dir = Path(colmap_dir)
        colmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Create COLMAP directory structure
        images_dir = colmap_dir / "images"
        masks_dir = colmap_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Copy and rename images for COLMAP rig format
        pinhole_dir = processed_dir / "pinhole_views"
        mask_source_dir = processed_dir / "masks"
        
        logger.info("Preparing data for COLMAP...")
        
        # Copy images and masks with COLMAP naming convention
        for camera in self.reframer.rig_config.cameras:
            camera_images = pinhole_dir / camera.name
            camera_masks = mask_source_dir / camera.name
            
            if camera_images.exists():
                for img_path in sorted(camera_images.glob("*.jpg")):
                    # COLMAP expects: frame_XXXX_cam_YY.jpg
                    frame_num = img_path.stem.split('_')[-1]
                    new_name = f"frame_{frame_num}_cam_{camera.id:02d}.jpg"
                    
                    # Copy image
                    cv2.imwrite(
                        str(images_dir / new_name),
                        cv2.imread(str(img_path))
                    )
                    
                    # Copy corresponding mask
                    mask_path = camera_masks / f"{img_path.stem}_mask.png"
                    if mask_path.exists():
                        cv2.imwrite(
                            str(masks_dir / new_name.replace('.jpg', '.png')),
                            cv2.imread(str(mask_path), 0)
                        )
        
        # Copy rig configuration
        import shutil
        shutil.copy(
            processed_dir / "rig_config.json",
            colmap_dir / "rig_config.json"
        )
        
        # Create COLMAP script
        script = f"""#!/bin/bash
# COLMAP reconstruction script

DATASET_PATH={colmap_dir}

colmap feature_extractor \\
    --database_path $DATASET_PATH/database.db \\
    --image_path $DATASET_PATH/images \\
    --ImageReader.mask_path $DATASET_PATH/masks \\
    --ImageReader.camera_model PINHOLE

colmap exhaustive_matcher \\
    --database_path $DATASET_PATH/database.db

colmap rig_bundle_adjuster \\
    --input_path $DATASET_PATH/sparse/0 \\
    --output_path $DATASET_PATH/sparse/0 \\
    --rig_config_path $DATASET_PATH/rig_config.json \\
    --BundleAdjustment.refine_principal_point 1
"""
        
        script_path = colmap_dir / "run_colmap.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        import os
        os.chmod(script_path, 0o755)
        
        logger.info(f"COLMAP data prepared in: {colmap_dir}")
        logger.info(f"Run: bash {script_path}")


def main():
    """Example usage of integrated pipeline."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated 360° preprocessing pipeline"
    )
    
    parser.add_argument("input", type=Path, help="Input 360° image or video")
    parser.add_argument("output", type=Path, help="Output directory")
    
    parser.add_argument("--rig", default="ring12", 
                       choices=["cube", "ring8", "ring12", "geodesic"],
                       help="Rig pattern")
    parser.add_argument("--model", default="sam3",
                       choices=["sam3", "fastsam", "auto"],
                       help="Masking model")
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="Frames to skip for video")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process")
    parser.add_argument("--prepare-colmap", action="store_true",
                       help="Prepare output for COLMAP")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Create processor
    processor = IntegratedPreprocessor(
        rig_pattern=args.rig,
        masking_model=args.model,
        use_gpu=not args.no_gpu
    )
    
    # Process input
    if args.input.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video
        stats = processor.process_360_video(
            video_path=args.input,
            output_dir=args.output,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames
        )
    else:
        # Image
        stats = processor.process_360_image(
            equirect_path=args.input,
            output_dir=args.output
        )
    
    # Prepare for COLMAP if requested
    if args.prepare_colmap:
        colmap_dir = args.output / "colmap_ready"
        processor.prepare_for_colmap(args.output, colmap_dir)
    
    print(f"\nProcessing complete!")
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    if args.prepare_colmap:
        print(f"\nCOLMAP data ready in: {args.output / 'colmap_ready'}")
        print(f"Run: bash {args.output / 'colmap_ready' / 'run_colmap.sh'}")


if __name__ == "__main__":
    main()
