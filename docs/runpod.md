# RunPod Deployment Guide

This guide explains how to deploy the `fisheye-workflow` on [RunPod](https://www.runpod.io/) GPU instances for high-performance processing.

## Option 1: Quick Start (Standard Pytorch Template)

The easiest way to get started if you don't want to build a custom Docker image.

1.  **Launch a Pod**:
    *   Go to RunPod Console -> **Deploy**.
    *   Select a GPU (e.g., RTX 3090, RTX 4090, or A6000).
    *   Choose Template: **RunPod PyTorch 2.1** (or similar with CUDA 11.8+).
    *   Set **Disk Size**: At least 50GB (to hold datasets + checkpoints).

2.  **Connect**:
    *   Use the **Web Terminal** or SSH into your pod.

3.  **Setup**:
    Clone your repo (or upload your code) and run the setup script:

    ```bash
    git clone https://github.com/your-repo/fisheye-workflow.git
    cd fisheye-workflow
    
    # Run the setup script to install SAM3 & dependencies
    bash scripts/setup_runpod.sh
    ```

4.  **Run Processing**:
    You are now ready to run scripts!
    ```bash
    python scripts/masking_v2.py ...
    ```

---

## Option 2: Custom Docker Image (Reproducible)

For a cleaner, pre-configured environment, you can build and use our custom Docker image.

### 1. Build & Push Image
On your local machine (with Docker installed):

```bash
# Build image
docker build -t your-username/fisheye-workflow:latest .

# Push to Docker Hub
docker push your-username/fisheye-workflow:latest
```

### 2. Deploy on RunPod
1.  Go to **Templates** -> **New Template**.
2.  **Image Name**: `your-username/fisheye-workflow:latest`
3.  **Container Disk**: 20GB
4.  **Volume Disk**: 50GB (for your data)
5.  **Volume Mount Path**: `/workspace`
6.  Save and **Deploy** this template on a GPU instance.

---

## 📂 Managing Data

RunPod instances are ephemeral (they disappear when terminated), but your **Volume** (`/workspace`) persists.

*   **Upload Data**: Use `scp`, `rsync`, or the Jupyter Lab "Upload" button (if available).
    ```bash
    # Example Upload via SCP
    scp -P [PORT] -r /local/path/to/project root@[IP]:/workspace/
    ```

*   **Download Results**:
    ```bash
    # Example Download via SCP
    scp -P [PORT] -r root@[IP]:/workspace/project/masks /local/download/path/
    ```

## ⚡ Recommended Specs

For efficient SAM3 processing:
*   **VRAM**: >8GB (RTX 3080/3090/4090 recommended)
*   **CPU**: 4+ vCPUs
*   **RAM**: >16GB
