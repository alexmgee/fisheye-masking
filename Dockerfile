# Base image with CUDA 11.8 support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 11.8
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    tqdm \
    pyyaml \
    scipy \
    ultralytics \
    matplotlib

# Install SAM3
WORKDIR /app
RUN git clone https://github.com/facebookresearch/sam3.git \
    && cd sam3 \
    && pip3 install -e .

# Download SAM3 Checkpoint
RUN mkdir -p /app/checkpoints \
    && wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt -P /app/checkpoints/

# Copy workflow scripts
COPY . /app/fisheye-workflow

# Set working directory
WORKDIR /app/fisheye-workflow

# Default command (interactive shell)
CMD ["/bin/bash"]
