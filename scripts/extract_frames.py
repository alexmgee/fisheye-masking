#!/usr/bin/env python3
"""
Extract frames from fisheye video at configurable FPS with blur filtering.

Features:
- Laplacian variance blur detection (standard photogrammetry method)
- Configurable blur threshold
- Discards blurry frames automatically
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Calculate blur score using Laplacian variance.
    Higher score = sharper image.
    
    Args:
        image: BGR image
    
    Returns:
        Laplacian variance (blur score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def extract_frames_with_blur_filter(
    input_video: Path,
    output_dir: Path,
    fps: float = 5.0,
    blur_threshold: float = 100.0,
    quality: int = 1,
    output_format: str = 'jpg',
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    keep_blurry: bool = False
) -> Tuple[int, int]:
    """
    Extract frames from video with blur filtering.
    
    Args:
        input_video: Path to video file
        output_dir: Directory for extracted frames
        fps: Frames per second to extract (default: 5)
        blur_threshold: Minimum blur score to keep (default: 100)
        quality: JPEG quality 1-31, lower is better (default: 1)
        output_format: 'jpg' or 'png'
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        keep_blurry: If True, save blurry frames to separate folder
    
    Returns:
        Tuple of (frames_kept, frames_discarded)
    """
    input_video = Path(input_video)
    output_dir = Path(output_dir)
    
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if keep_blurry:
        blurry_dir = output_dir / 'blurry'
        blurry_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video: {input_video.name}")
    print(f"  Duration: {duration:.1f}s @ {video_fps:.1f} FPS")
    print(f"  Extracting at {fps} FPS with blur threshold {blur_threshold}")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    
    # Handle start/end times
    start_frame = int((start_time or 0) * video_fps)
    end_frame = int((end_time or duration) * video_fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_kept = 0
    frames_discarded = 0
    frame_num = start_frame
    
    # Calculate expected frames for progress bar
    expected_frames = (end_frame - start_frame) // frame_interval
    
    with tqdm(total=expected_frames, desc="Extracting") as pbar:
        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_num - start_frame) % frame_interval == 0:
                # Calculate blur score
                blur_score = calculate_blur_score(frame)
                
                # Determine filename
                filename = f"{input_video.stem}_{frame_num:06d}.{output_format}"
                
                if blur_score >= blur_threshold:
                    # Keep this frame
                    output_path = output_dir / filename
                    if output_format == 'jpg':
                        cv2.imwrite(str(output_path), frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        cv2.imwrite(str(output_path), frame)
                    frames_kept += 1
                else:
                    # Discard or save to blurry folder
                    if keep_blurry:
                        output_path = blurry_dir / filename
                        cv2.imwrite(str(output_path), frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frames_discarded += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'kept': frames_kept, 
                    'blur': f'{blur_score:.0f}'
                })
            
            frame_num += 1
    
    cap.release()
    
    print(f"\nResults:")
    print(f"  Kept: {frames_kept} frames")
    print(f"  Discarded: {frames_discarded} blurry frames")
    print(f"  Output: {output_dir}")
    
    return frames_kept, frames_discarded


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video with blur filtering"
    )
    parser.add_argument('input', type=Path, help='Input video file')
    parser.add_argument('output', type=Path, help='Output directory for frames')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='Frames per second to extract (default: 5)')
    parser.add_argument('--blur-threshold', type=float, default=100.0,
                        help='Minimum blur score to keep frame (default: 100)')
    parser.add_argument('--quality', type=int, default=1,
                        help='JPEG quality 1-31, lower is better (default: 1)')
    parser.add_argument('--format', choices=['jpg', 'png'], default='jpg',
                        help='Output format (default: jpg)')
    parser.add_argument('--start', type=float, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--keep-blurry', action='store_true',
                        help='Save blurry frames to separate folder')
    
    args = parser.parse_args()
    
    try:
        kept, discarded = extract_frames_with_blur_filter(
            args.input,
            args.output,
            fps=args.fps,
            blur_threshold=args.blur_threshold,
            quality=args.quality,
            output_format=args.format,
            start_time=args.start,
            end_time=args.end,
            keep_blurry=args.keep_blurry
        )
        print(f"\nSuccess! Kept {kept} sharp frames, discarded {discarded} blurry.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
