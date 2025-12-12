# RunPod Master Class: From Google Drive to 3D Splats

This guide is designed for first-time RunPod users. It implements a **Cost-Efficient Workflow**: we will use a cheap "Network Volume" to store your data permanently, use a small instance to download everything from Google Drive, and only pay for a powerful GPU when we are actually processing.

---

## 🏗️ Phase 1: Create Persistent Storage (Network Volume)
*Why?* Standard pods lose all data when you turn them off. Network Volumes live forever (until you delete them) and can be attached to any pod.

1.  **Login** to [RunPod.io](https://www.runpod.io/).
2.  Go to **Storage** (on the left menu) -> **Network Volumes**.
3.  Click **Create**.
    *   **Name**: `fisheye-data`
    *   **Data Center**: Choose a region (e.g., `US - CA` or `EU - RO`). *Note: You MUST create your Pods in this same region later.*
    *   **Size**: Start with **50 GB** (adjust based on your footage size).
4.  Click **Create**.
    *   *Cost: Very cheap (approx $0.07 / GB / month).*

---

## 🚚 Phase 2: Data Transfer (The "Cheap" Pod)
*Goal: Download big files from Google Drive without paying for an expensive GPU.*

1.  **Deploy a Pod**:
    *   Go to **Pods** -> **Deploy**.
    *   **Select Region**: **MUST** be the same region as your Storage (e.g., `US - CA`).
    *   **GPU**: Scroll down to the "Community Cloud" or "Secure Cloud" and pick the **cheapest** thing you find (e.g., a CPU instance or a cheap GPU like a P4000 or generic). We only need internet access, not power.
    *   **Template**: Select **RunPod PyTorch 2.1**.
    *   **Customize Deployment** (Important!):
        *   Click **"Edit Template"** or "Customize".
        *   Scroll to **Network Volume Mounts**.
        *   Select your `fisheye-data` volume.
        *   **Mount Path**: Enter `/workspace` (This makes your volume the main folder).
    *   Click **Continue** -> **Deploy**.

2.  **Connect to Terminal**:
    *   Wait for the pod to start (Status: "Running").
    *   Click **Connect** -> **Start Web Terminal** -> **Connect**.

3.  **Download from Google Drive**:
    *   **Get your Google Drive Link**:
        *   Right-click your folder/file in Drive -> **Share** -> **"Anyone with the link"**.
        *   Copy the Link.
    *   **Install gdown** (tool for downloading drive files):
        ```bash
        pip install gdown
        ```
    *   **Download**:
        ```bash
        # Replace the URL below with your actual link
        gdown --folder "https://drive.google.com/drive/folders/YOUR_ID_HERE?usp=sharing"
        ```
        *   *Note: If it's a single file (zip/mp4), remove `--folder`.*

4.  **Organize**:
    *   Move your files so they are neat.
    *   ```bash
        mkdir -p projects/my_scan/raw
        mv YOUR_VIDEO_FILE.mp4 projects/my_scan/raw/
        ```
    *   *Verify*: Type `ls -R` to see your files are there.

5.  **Shutdown**:
    *   Since your data is on the **Network Volume** (`/workspace`), it is SAFE.
    *   Go back to RunPod Dashboard -> **My Pods**.
    *   **Stop** (and **Terminate**) this cheap pod. You stop paying for compute immediately.

---

## 🚀 Phase 3: The Powerhouse (The "Expensive" Pod)
*Goal: Run SAM3 AI masking with top-tier performance.*

1.  **Deploy GPU Pod**:
    *   **Region**: Same as before.
    *   **GPU**: Choose an **RTX 3090** or **RTX 4090**. (SAM3 loves VRAM).
    *   **Template**: **RunPod PyTorch 2.1**.
    *   **Network Volume**: Just like before, attach `fisheye-data` to `/workspace`.

2.  **Setup the Workflow**:
    *   Connect to **Web Terminal**.
    *   Your data is already there! Check with `ls`.
    *   **Install the Pipeline** (One-Liner):
        ```bash
        git clone https://github.com/alexmgee/fisheye-masking.git
        cd fisheye-masking
        bash scripts/setup_runpod.sh
        ```

3.  **Run the Magic**:
    
    a. **Split** (if needed):
    ```bash
    python scripts/split_osv.py /workspace/projects/my_scan/raw/VID_001.insv -o /workspace/projects/my_scan/split
    ```

    b. **Extract Frames**:
    ```bash
    python scripts/extract_frames.py /workspace/projects/my_scan/split/VID_001_front.mp4 /workspace/projects/my_scan/frames --fps 2
    ```

    c. **Auto Mask w/ SAM3**:
    ```bash
    # This uses the GPU!
    python scripts/masking_v2.py /workspace/projects/my_scan/frames /workspace/projects/my_scan/masks \
        --geometry fisheye \
        --model sam3 \
        --remove "operator" "tripod" "shadow"
    ```

4.  **Review (Optional)**:
    *   You can't see the GUI easily on RunPod Web Terminal.
    *   **Option A**: Trust the AI (it usually works well with SAM3).
    *   **Option B**: Compress the masks and download them to review locally.
        ```bash
        cd /workspace/projects/my_scan
        zip -r masks_review.zip masks/
        ```

---

## 🏁 Phase 4: Download & Cleanup

1.  **Download Results**:
    *   **From RunPod**:
        *   In the Dashboard, file browser (if using Jupyter) or use `gdown` simply to re-upload to Drive? No, easier to use transfer.sh or just scp.
    *   **Simplest Way (jupyter)**:
        *   On the Pod dashboard, click **Connect** -> **Connect via HTTP (Port 8888)** to open Jupyter Lab.
        *   Navigate files on the left.
        *   Right-click `masks_review.zip` -> **Download**.

2.  **Terminate**:
    *   Once you have your masks downloaded locally, **Terminate** the GPU pod.

3.  **Storage**:
    *   You can keep the **Network Volume** as long as you want (it's cheap) for the next project, or delete it if you are done forever.

---
**Summary Checklist**
- [ ] Create Network Storage (50GB+).
- [ ] Launch Cheap Pod + Attach Storage.
- [ ] `gdown` footage from Google Drive.
- [ ] Terminate Cheap Pod.
- [ ] Launch RTX 3090 Pod + Attach Storage.
- [ ] Run `fisheye-masking` pipeline.
- [ ] Download results.
- [ ] Terminate GPU Pod.
