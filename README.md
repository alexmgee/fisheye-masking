# fisheye-masking


**Advanced preprocessing pipeline for 360°/Fisheye photogrammetry and Gaussian Splatting.**

This toolkit provides a production-ready workflow to prepare raw fisheye footage for 3D reconstruction. It handles the entire pipeline: splitting raw OSV/INSV files, extracting sharp frames, automatically masking out operators/tripods using state-of-the-art AI (SAM3), and providing tools for manual quality control.

---

## 🚀 Workflow Overview

The pipeline consists of four main stages:

1.  **Split** (`split_osv.py`): Separate raw dual-lens footage into front/rear streams.
2.  **Extract** (`extract_frames.py`): Extract frames with automatic blur detection to keep only sharp images.
3.  **Mask** (`masking_v2.py`): Automatically mask unwanted objects (tripod, operator, shadows) using text-prompted AI.
4.  **Review** (`review_gallery.py`): Efficiently review, refine, and correct masks with a custom GUI.

```mermaid
graph LR
    A[Raw OSV/INSV] -->|split_osv.py| B[Front/Rear MP4]
    B -->|extract_frames.py| C[Sharp Frames]
    C -->|masking_v2.py| D[Auto Masks]
    D -->|review_gallery.py| E[Final Dataset]
    E --> F[COLMAP / Splatting]
```

---

## 🛠️ Installation

### Prerequisites
*   **Operating System**: Linux (Recommended)
*   **GPU**: NVIDIA GPU (RTX 30-series or newer recommended for SAM3)
*   **Drivers**: CUDA 11.8+
*   **Tools**: `ffmpeg`, `ffprobe`

### Environment Setup

Create a dedicated Conda environment:

```bash
conda create -n fisheye-workflow python=3.10
conda activate fisheye-workflow

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install numpy opencv-python tqdm pyyaml scipy
```

### Install AI Models (Optional but Recommended)
For the best masking quality, install **SAM3**. The system will fallback to YOLO/FastSAM if SAM3 is missing.

```bash
# Install SAM3
git clone https://github.com/facebookresearch/sam3
cd sam3 && pip install -e . && cd ..

# Download Checkpoint
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt -P checkpoints/

# Install Ultralytics (for YOLO/FastSAM fallback)
pip install ultralytics
```

---

## 📚 Component Guide

### 1. Split OSV/INSV
Splits raw dual-lens files (e.g., from Insta360, Google Street View cameras) into separate video files for front and rear lenses. Handles dual-stream, side-by-side, and top-bottom layouts automatically.

```bash
python scripts/split_osv.py input_file.osv --output projects/my_scan/raw
```

### 2. Extract Frames
Extracts frames from video files while automatically filtering out blurry images using Laplacian variance. This ensures your photogrammetry dataset only contains sharp data.

```bash
python scripts/extract_frames.py input_front.mp4 projects/my_scan/frames --fps 2 --blur-threshold 120
```

*   `--blur-threshold`: Higher values = stricter sharpness requirement (default: 100).
*   `--fps`: Extraction rate (default: 5).

### 3. Auto Masking (Masking v2)
The core of the automated pipeline. Uses text-based prompting (e.g., "remove tripod") to segment unwanted objects. Supports fisheye and equirectangular geometry awareness.

**Key Features:**
*   **Text Prompts**: "tripod", "camera operator", "shadow of tripod"
*   **Geometry Aware**: Handles lens distortion correcty.
*   **Quality Control**: Assigns confidence scores to masks.

```bash
# Basic usage
python scripts/masking_v2.py projects/my_scan/frames projects/my_scan/masks --geometry fisheye

# With custom prompts and SAM3
python scripts/masking_v2.py projects/my_scan/frames projects/my_scan/masks \
    --geometry fisheye \
    --model sam3 \
    --remove "tripod" "operator" "shadow" "equipment"
```

*   See [Masking Reference](docs/reference/masking_v2_reference.md) for advanced configuration.

### 4. Review Gallery
A high-performance manual review tool designed for rapid QA.

*   **View**: Browse frames and mask overlays efficiently.
*   **Filter**: Automatically flags low-confidence masks for review.
*   **Edit**: Interactive painting/erasing of masks.
*   **Zoom/Pan**: Full resolution inspection with `Scroll` to zoom and `Middle-Drag` to pan.

```bash
# Review flagged/low-confidence masks
python scripts/review_gallery.py projects/my_scan/masks --flagged
```

**Controls:**
*   `Scroll`: Zoom in/out (centered on cursor)
*   `Middle-Click Drag`: Pan image
*   `Left Click`: Paint mask (Add)
*   `Right Click`: Erase mask (Remove)
*   `Z / X`: Decrease / Increase Brush Size
*   `S`: Save and Go to Next
*   `Space`: Skip to Next
*   `G`: Toggle Gallery View

---

## 📂 Project Structure

```
fisheye-workflow/
├── scripts/
│   ├── split_osv.py        # Raw file splitter
│   ├── extract_frames.py   # Blur-aware frame extractor
│   ├── masking_v2.py       # AI Auto-masking pipeline
│   └── review_gallery.py   # Manual review GUI
├── docs/
│   └── reference/          # Detailed tech references
├── models/                 # Model checkpoints
└── projects/               # Workspace for your datasets
```

## 🔧 Troubleshooting

*   **Zoom not working in Reviewer?** Ensure you are running the latest version of `review_gallery.py` which uses a custom rendering engine. The window is resizable; maximize it for the best experience.
*   **CUDA OOM (Out of Memory)?** If `masking_v2.py` crashes on high-res images, try reducing `--batch-size` to 1 or using a smaller model (e.g., `--model fastsam`).

---
**Author**: 360-to-splat-v2 Team
