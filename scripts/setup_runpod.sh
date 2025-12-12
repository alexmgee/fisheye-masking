#!/bin/bash
# setup_runpod.sh
# Quick setup script for standard RunPod PyTorch instances
# Usage: source scripts/setup_runpod.sh

echo "🚀 Starting RunPod Setup for Fisheye Workflow..."

# 1. System Dependencies
echo "📦 Installing system packages..."
apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0

# 2. Python Dependencies
echo "🐍 Installing Python dependencies..."
pip install ultralytics opencv-python tqdm pyyaml scipy

# 3. Install SAM3
if [ ! -d "sam3" ]; then
    echo "🤖 Installing SAM3..."
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    pip install -e .
    cd ..
else
    echo "✅ SAM3 already cloned."
fi

# 4. Download Checkpoints
echo "⬇️  Downloading Model Checkpoints..."
mkdir -p checkpoints
if [ ! -f "checkpoints/sam3_hiera_large.pt" ]; then
    wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt -P checkpoints/
else
    echo "✅ SAM3 checkpoint already exists."
fi

echo "✨ Setup Complete! You can now run processing:"
echo "   python scripts/masking_v2.py ..."
