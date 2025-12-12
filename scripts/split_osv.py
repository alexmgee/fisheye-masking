#!/usr/bin/env python3
"""
Split OSV/INSV files into separate front and rear fisheye streams.

Handles both dual-stream (separate video tracks) and single-stream 
(side-by-side) layouts from DJI Osmo 360 and Insta360 cameras.
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Tuple, Optional
import argparse


def probe_video(input_file: Path) -> dict:
    """Probe video file to get stream information."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        str(input_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    return json.loads(result.stdout)


def get_video_layout(probe_data: dict) -> Tuple[str, int, int]:
    """
    Determine the video layout from probe data.
    
    Returns:
        layout: 'dual_stream' or 'side_by_side' or 'top_bottom'
        width: video width
        height: video height
    """
    video_streams = [s for s in probe_data['streams'] if s['codec_type'] == 'video']
    
    if len(video_streams) >= 2:
        # Dual stream - separate tracks for front/rear
        w = video_streams[0]['width']
        h = video_streams[0]['height']
        return 'dual_stream', w, h
    
    elif len(video_streams) == 1:
        w = video_streams[0]['width']
        h = video_streams[0]['height']
        
        # Check aspect ratio to determine layout
        # Side-by-side: width is roughly 2x height (e.g., 5760x2880)
        # Top-bottom: height is roughly 2x width
        if w > h * 1.5:
            return 'side_by_side', w, h
        elif h > w * 1.5:
            return 'top_bottom', w, h
        else:
            # Single fisheye or unknown - return as-is
            return 'single', w, h
    
    raise RuntimeError("No video streams found")


def split_dual_stream(input_file: Path, output_dir: Path, lossless: bool = True) -> Tuple[Path, Path]:
    """
    Split dual-stream video into separate files (lossless copy).
    """
    front_output = output_dir / f"{input_file.stem}_front.mp4"
    rear_output = output_dir / f"{input_file.stem}_rear.mp4"
    
    # Stream 0 -> front
    cmd_front = [
        'ffmpeg', '-y',
        '-i', str(input_file),
        '-map', '0:0',
        '-c', 'copy' if lossless else 'libx264 -crf 0',
        '-f', 'mp4',
        str(front_output)
    ]
    
    # Stream 1 -> rear
    cmd_rear = [
        'ffmpeg', '-y',
        '-i', str(input_file),
        '-map', '0:1',
        '-c', 'copy' if lossless else 'libx264 -crf 0',
        '-f', 'mp4',
        str(rear_output)
    ]
    
    print(f"Extracting front lens (stream 0)...")
    subprocess.run(cmd_front, check=True)
    
    print(f"Extracting rear lens (stream 1)...")
    subprocess.run(cmd_rear, check=True)
    
    return front_output, rear_output


def split_side_by_side(input_file: Path, output_dir: Path, width: int, height: int) -> Tuple[Path, Path]:
    """
    Split side-by-side video using crop filter.
    Uses high quality encoding (CRF 0) since we must re-encode.
    """
    front_output = output_dir / f"{input_file.stem}_front.mp4"
    rear_output = output_dir / f"{input_file.stem}_rear.mp4"
    
    half_width = width // 2
    
    # Use filter_complex to output both at once
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_file),
        '-filter_complex',
        f'[0:v]crop={half_width}:{height}:0:0[front];'
        f'[0:v]crop={half_width}:{height}:{half_width}:0[rear]',
        '-map', '[front]',
        '-c:v', 'libx264', '-crf', '0', '-preset', 'fast',
        str(front_output),
        '-map', '[rear]',
        '-c:v', 'libx264', '-crf', '0', '-preset', 'fast',
        str(rear_output)
    ]
    
    print(f"Splitting side-by-side video...")
    subprocess.run(cmd, check=True)
    
    return front_output, rear_output


def split_top_bottom(input_file: Path, output_dir: Path, width: int, height: int) -> Tuple[Path, Path]:
    """
    Split top-bottom video using crop filter.
    """
    front_output = output_dir / f"{input_file.stem}_front.mp4"
    rear_output = output_dir / f"{input_file.stem}_rear.mp4"
    
    half_height = height // 2
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_file),
        '-filter_complex',
        f'[0:v]crop={width}:{half_height}:0:0[front];'
        f'[0:v]crop={width}:{half_height}:0:{half_height}[rear]',
        '-map', '[front]',
        '-c:v', 'libx264', '-crf', '0', '-preset', 'fast',
        str(front_output),
        '-map', '[rear]',
        '-c:v', 'libx264', '-crf', '0', '-preset', 'fast',
        str(rear_output)
    ]
    
    print(f"Splitting top-bottom video...")
    subprocess.run(cmd, check=True)
    
    return front_output, rear_output


def split_osv(input_file: Path, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Main function to split OSV/INSV file into front and rear fisheye streams.
    
    Args:
        input_file: Path to OSV/INSV file
        output_dir: Output directory (default: same as input)
    
    Returns:
        Tuple of (front_path, rear_path)
    """
    input_file = Path(input_file)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_dir = output_dir or input_file.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Probe the file
    print(f"Analyzing: {input_file.name}")
    probe_data = probe_video(input_file)
    layout, width, height = get_video_layout(probe_data)
    
    print(f"  Layout: {layout}")
    print(f"  Resolution: {width}x{height}")
    
    if layout == 'dual_stream':
        return split_dual_stream(input_file, output_dir)
    elif layout == 'side_by_side':
        return split_side_by_side(input_file, output_dir, width, height)
    elif layout == 'top_bottom':
        return split_top_bottom(input_file, output_dir, width, height)
    elif layout == 'single':
        # Just copy the file as "front" - it's already a single fisheye
        output = output_dir / f"{input_file.stem}_fisheye.mp4"
        subprocess.run(['ffmpeg', '-y', '-i', str(input_file), '-c', 'copy', str(output)], check=True)
        return output, None
    
    raise RuntimeError(f"Unknown layout: {layout}")


def main():
    parser = argparse.ArgumentParser(
        description="Split OSV/INSV files into separate front and rear fisheye streams"
    )
    parser.add_argument('input', type=Path, help='Input OSV/INSV file')
    parser.add_argument('-o', '--output', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        front, rear = split_osv(args.input, args.output)
        print(f"\nSuccess!")
        print(f"  Front: {front}")
        if rear:
            print(f"  Rear:  {rear}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
