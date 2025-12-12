# **Synthesis Report: Omnidirectional 3D Reconstruction Pathways**

## **1\. Executive Summary: The "Virtual" vs. "Physical" Divergence**

This report synthesizes two distinct methodologies for reconstructing high-fidelity 3D Gaussian Splats from omnidirectional capture devices (specifically DJI Osmo Action 360, Insta360, and Laowa 4mm fisheye optics). The choice between these pathways forces a trade-off between **software compatibility** and **optical data integrity**.

* **Path A: The Virtual Rig (Geometry-First)**  
  * **Concept:** Mathematically decomposes spherical data into 12 "virtual" rectilinear cameras arranged in a dodecahedron.  
  * **Primary Engines:** COLMAP 3.13 / GLOMAP 1.2.  
  * **Philosophy:** "Fix the data to fit the solver." This forces spherical data into the Pinhole model that standard Structure-from-Motion (SfM) and Gaussian Splatting (3DGS) algorithms prefer.  
* **Path B: The Raw Fisheye (Optics-First)**  
  * **Concept:** Treats the camera as a physical rig of two opposing \>180° lenses, preserving original sensor pixels and distortion curves.  
  * **Primary Engines:** Agisoft Metashape Pro / NVIDIA 3DGRUT.  
  * **Philosophy:** "Fix the solver to fit the data." This requires advanced lens modeling and specialized rendering (Ray Tracing/Unscented Transform) to handle the extreme non-linearity of the optics.

## ---

**2\. Technical Comparison: Workflow Architecture**

### **2.1 Data Engineering & Pre-processing**

The divergence begins immediately at the file handling level.

| Feature | Path A: Virtual Rig (12-View) | Path B: Raw Fisheye (Dual-View) |
| :---- | :---- | :---- |
| **Input Format** | Equirectangular Video (Stitched). | **Raw Dual-Stream** (Unstitched). |
| **Extraction** | Requires projecting lat/long map to 12 planar views (dodecahedron). | Requires **FFmpeg stream separation** (e.g., separating .osv streams). |
| **Image Count** | High (\~12x input frames). Explodes dataset size. | Low (2x input frames). Efficient storage. |
| **Pixel Integrity** | **Resampled.** Pixels are interpolated during equirectangular stitch *and* planar projection. | **Native.** Pixels are 1:1 from the sensor (excluding compression). |
| **Color Space** | Typically handled post-stitch. | **Critical:** Must transform Rec.2020 \-\> Rec.709 *before* masking to prevent artifacts. |

**Synthesis Note:** Path A relies on the camera's internal stitching (or DJI Studio), which introduces "ghosting" parallax errors at the seam. Path B eliminates stitching entirely, allowing the SfM software to solve the precise offset between the front and rear lenses as a rigid 3D transformation.1

### **2.2 The Reconstruction Engine (SfM)**

**Path A (COLMAP/GLOMAP):**

* **Rig Constraint:** We define a mathematical "hard constraint" where 12 cameras share an optical center.  
* **GLOMAP Advantage:** Since the 12 views are rigid, GLOMAP's global rotation averaging is extremely effective here. It treats the 12 views as a single "super-view," preventing the scene from bending over long trajectories.3  
* **Weakness:** It assumes a single center of projection. The physical Osmo 360 has *two* centers (offset by \~2cm). This mismatch creates noise in the point cloud at close range (\<1m).

**Path B (Metashape/Gradeeterna):**

* **Lens Model:** Uses high-order polynomial distortion models (OPENCV\_FISHEYE or Metashape's internal model) to map the curved rays.  
* **"Look-Behind" Capability:** The Laowa 4mm (210°) and Osmo lenses (\>180°) can see "behind" themselves. Path B utilizes this overlap for massive loop closure strength. A feature leaving the front lens is simultaneously visible in the rear lens, creating a continuous track that Path A breaks into disjointed observations.5  
* **Masking:** Requires the "Gradeeterna" workflow of aggressive circular masking to remove the sensor bezel and the photographer, which is more labor-intensive than Path A.6

### **2.3 The Neural Rendering Endpoint**

This is the most critical decision point. The choice of reconstruction path restricts your choice of rendering technology.

* **If you choose Path A (Virtual Rig):**  
  * **Compatible with:** Standard gsplat, Luma AI, Polycam, Splatfacto.  
  * **Why:** These engines use rasterization (sorting 2D splats). They require linear (pinhole) camera models. The 12-view split "tricks" them into working with spherical data.  
  * **Result:** High compatibility, but higher VRAM usage (rendering 12 views to see 360°) and potential seam artifacts.  
* **If you choose Path B (Raw Fisheye):**  
  * **Compatible with:** **NVIDIA 3DGRUT**, **Fisheye-GS**, or specialized branches of Nerfstudio.  
  * **Why:** Standard rasterization fails on curved rays. You need **Ray Tracing (3DGRT)** or the **Unscented Transform (3DGUT)** to correctly map a 3D Gaussian ellipsoid onto a curved fisheye sensor.7  
  * **Result:** Superior visual fidelity (no resampling blur), fewer artifacts at the poles, but significantly higher compute requirements for training.

## ---

**3\. Implementation Guides**

### **Path A: The Virtual Rig Implementation (COLMAP \+ GLOMAP)**

Step 1: Pinhole Extraction  
Convert equirectangular frames to 12 perspective images (Dodecahedral layout).

* *Note:* Ensure filenames match exactly across folders (e.g., cam01/frame001.jpg, cam02/frame001.jpg).9

**Step 2: Database Creation**

Bash

colmap feature\_extractor \\  
    \--database\_path database.db \\  
    \--image\_path./images \\  
    \--ImageReader.single\_camera\_per\_folder 1 \\  
    \--ImageReader.camera\_model PINHOLE

Step 3: Rig Configuration  
Create rig\_config.json defining the dodecahedron geometry.

Bash

colmap rig\_configurator \\  
    \--database\_path database.db \\  
    \--rig\_config\_path rig\_config.json

Step 4: Global Reconstruction  
Use GLOMAP to solve the scene globally using the rig constraints.

Bash

glomap mapper \\  
    \--database\_path database.db \\  
    \--image\_path./images \\  
    \--output\_path./sparse

### **Path B: The Raw Fisheye Implementation (Metashape \+ 3DGRUT)**

Step 1: Stream Separation  
Use FFmpeg to extract discrete streams from the .osv/.insv container without transcoding (if possible) or using high-bitrate intermediates.10

Bash

ffmpeg \-i input.osv \-map 0:0 \-c copy front.mp4  
ffmpeg \-i input.osv \-map 0:1 \-c copy rear.mp4

**Step 2: Metashape Alignment (Gradeeterna Method)**

1. **Import:** Load front/rear image sequences.  
2. **Masking:** Apply a circular mask to hide the sensor edges and the photographer.11  
3. **Calibration:** Set camera type to **Fisheye**.  
4. **Align:** Run alignment. Metashape handles the extreme distortion significantly better than COLMAP's default mapper.

Step 3: Export for 3DGRUT  
Use the Metashape export function (v2.1+) to generate a COLMAP-compatible folder structure.

* *Critical:* Ensure the exported camera model is OPENCV\_FISHEYE.  
* *Scripting:* If using older versions, use the "Gradeeterna" Python script to map internal Metashape calibration ($k\_1-k\_4$) to COLMAP format.12

Step 4: Training  
Train using NVIDIA 3DGRUT which natively supports the fisheye distortion.

Bash

python train.py \--config-name apps/colmap\_3dgut.yaml \\  
    dataset.source\_path=./dataset \\  
    camera\_model=FISHEYE

## ---

**4\. Recommended Workflow: The "Hybrid Superior"**

Based on the synthesis of both reports, the **Path B (Raw Fisheye)** approach yields scientifically superior data, but **Path A (Virtual Rig)** is safer for general software compatibility.  
The "Hybrid Superior" workflow combines the best of both:

1. **Ingest (Path B):** Split the .osv file using FFmpeg into two discrete fisheye streams. Do not stitch.  
2. **Color:** Apply Rec.2020 \-\> Rec.709 transform and circular crop in Davinci Resolve.  
3. **Alignment (Path B \- Metashape):** Import into **Agisoft Metashape**. Use "Fisheye" calibration.  
   * *Critical Step:* Use Metashape to refine the poses. It handles the extreme distortion better than COLMAP.  
   * *Refinement:* Create a "Multi-Camera System" in Metashape to enforce the fixed offset between Front/Rear lenses.  
4. **The Fork (Export Decision):**  
   * **For 3DGRUT (Max Quality):** Export directly as OPENCV\_FISHEYE. Train using Ray Tracing.  
   * **For Standard 3DGS (Compatibility):** Use Metashape's "Convert Images" tool to project the aligned fisheyes into the 12-view dodecahedron format (or a cube map) *after* alignment. This gives you the alignment precision of the raw fisheye track, but the file compatibility of the virtual rig.

## **5\. Summary Table**

| Feature | Virtual Rig (COLMAP) | Raw Fisheye (Metashape) |
| :---- | :---- | :---- |
| **Best For** | Standard 3DGS, Web Viewers, Speed. | Archival Quality, Research, 3DGRUT. |
| **Key Weakness** | Resampling blur; Parallax errors from stitching. | High complexity; Requires specialized renderer. |
| **Parallax Handling** | **Poor.** Ignores physical lens offset. | **Excellent.** Models physical lens offset. |
| **Geometric Drift** | Mitigated by GLOMAP global solver. | Mitigated by "Look-Behind" optical overlap. |
| **Masking** | Easy (rectangular). | Hard (Circular \+ Dynamic Photographer masking). |

Final Recommendation:  
If you have access to NVIDIA GPUs capable of running 3DGRUT, adopt the Raw Fisheye workflow immediately. It is the only path that respects the physical reality of your Laowa and Osmo optics. If you need to publish to web viewers (Luma/Polycam), use the Hybrid approach: Align as fisheye in Metashape, then rectify to pinhole for the final training set.

#### **Works cited**

1. Some INSV files combined while others are separate? : r/Insta360 \- Reddit, accessed December 6, 2025, [https://www.reddit.com/r/Insta360/comments/1clitei/some\_insv\_files\_combined\_while\_others\_are\_separate/](https://www.reddit.com/r/Insta360/comments/1clitei/some_insv_files_combined_while_others_are_separate/)  
2. Merging / Concatenating Insta 360 Videos : r/Insta360 \- Reddit, accessed December 6, 2025, [https://www.reddit.com/r/Insta360/comments/1465wrg/merging\_concatenating\_insta\_360\_videos/](https://www.reddit.com/r/Insta360/comments/1465wrg/merging_concatenating_insta_360_videos/)  
3. Activity · colmap/glomap \- GitHub, accessed December 6, 2025, [https://github.com/colmap/glomap/activity?ref=main](https://github.com/colmap/glomap/activity?ref=main)  
4. colmap/glomap: GLOMAP \- Global Structured-from-Motion Revisited \- GitHub, accessed December 6, 2025, [https://github.com/colmap/glomap](https://github.com/colmap/glomap)  
5. Laowa 4mm 2.8 Fisheye lens review \- YouTube, accessed December 6, 2025, [https://www.youtube.com/watch?v=EtLq4FhRtBU](https://www.youtube.com/watch?v=EtLq4FhRtBU)  
6. gradeeterna (u/gradeeterna) \- Reddit, accessed December 6, 2025, [https://www.reddit.com/user/gradeeterna/](https://www.reddit.com/user/gradeeterna/)  
7. I captured my kitchen with 3DGRUT using 180 degree fisheye images : r/GaussianSplatting, accessed December 6, 2025, [https://www.reddit.com/r/GaussianSplatting/comments/1k033rk/i\_captured\_my\_kitchen\_with\_3dgrut\_using\_180/](https://www.reddit.com/r/GaussianSplatting/comments/1k033rk/i_captured_my_kitchen_with_3dgrut_using_180/)  
8. nvpro-samples/vk\_gaussian\_splatting: Sample viewer implementing several rendering methods for 3D gaussians using Vulkan API \- GitHub, accessed December 6, 2025, [https://github.com/nvpro-samples/vk\_gaussian\_splatting](https://github.com/nvpro-samples/vk_gaussian_splatting)  
9. Rig Support — COLMAP 3.14.0.dev0 | 5b9a079a (2025-11-14) documentation, accessed December 6, 2025, [https://colmap.github.io/rigs.html](https://colmap.github.io/rigs.html)  
10. How to extract video+audio streams from MP4 file via ffmpeg? \- Super User, accessed December 6, 2025, [https://superuser.com/questions/1736640/how-to-extract-videoaudio-streams-from-mp4-file-via-ffmpeg](https://superuser.com/questions/1736640/how-to-extract-videoaudio-streams-from-mp4-file-via-ffmpeg)  
11. Masking \- Agisoft Metashape, accessed December 6, 2025, [https://www.agisoft.com/forum/index.php?topic=8012.0](https://www.agisoft.com/forum/index.php?topic=8012.0)  
12. Gaussian Splatting \- Agisoft Metashape, accessed December 6, 2025, [https://www.agisoft.com/forum/index.php?topic=15861.0](https://www.agisoft.com/forum/index.php?topic=15861.0)  
13. Support Colmap Export \- Feedback & Requests \- Epic Developer Community Forums, accessed December 6, 2025, [https://forums.unrealengine.com/t/support-colmap-export/1277698](https://forums.unrealengine.com/t/support-colmap-export/1277698)