# **Advanced Photogrammetric Reconstruction utilizing High-Distortion Fisheye Optics and Radiance Fields**

## **1\. Introduction: The Curvilinear Shift in 3D Reconstruction**

The field of three-dimensional reconstruction, specifically through the lens of Photogrammetry and Neural Radiance Fields (NeRF), is currently undergoing a significant paradigmatic shift. Traditionally, the reconstruction pipeline has been dominated by the rectilinear (pinhole) camera model. This preference was not born of optical superiority, but rather mathematical convenience; the linearity of light rays in a pinhole model simplifies the epipolar geometry required for Structure-from-Motion (SfM) and allows for straightforward rasterization in graphics pipelines. However, the pinhole model imposes severe limitations on field of view (FoV), necessitating large datasets of hundreds or thousands of images to ensure sufficient overlap and environmental coverage.

This report explores the rigorous implementation of high-distortion, ultra-wide-angle optics—specifically circular fisheye lenses with 180° to 210° fields of view—as a primary data source for high-fidelity 3D reconstruction. The utilization of such optics, exemplified by the Laowa 4mm f/2.8 lens and the dual-optical modules of 360° cameras like the DJI Osmo Action 360 and Insta360 series, offers a transformative advantage: the ability to capture comprehensive environmental geometry with a fraction of the frames required by rectilinear systems.

However, this optical advantage introduces complex challenges in data engineering and mathematical modeling. The widely adopted workflow of stitching dual-fisheye footage into equirectangular projections, while suitable for human consumption via VR headsets, is mathematically destructive for photogrammetric analysis. Stitching introduces parallax errors, ghosting artifacts, and non-uniform pixel density distributions that degrade the performance of gradient-based optimization algorithms used in 3D Gaussian Splatting (3DGS).

Consequently, this research establishes a comprehensive technical framework for the "Gradeeterna" workflow and its derivatives. This approach advocates for the decoupling of 360° footage into discrete, raw fisheye streams, the rigorous management of color spaces (Rec.2020 to Rec.709), and the utilization of advanced alignment algorithms in Agisoft Metashape that can accommodate the extreme non-linearities of the OPENCV\_FISHEYE distortion model. Furthermore, it examines the integration of these datasets into emerging neural rendering frameworks such as NVIDIA’s 3DGRUT (3D Gaussian Ray Tracing / Unscented Transform), which natively handles curvilinear ray casting, thereby bypassing the information loss associated with traditional undistortion processes.

---

## **2\. Optical Theory and Hardware Specifications**

To build a robust reconstruction set, one must first understand the fundamental differences between the optical acquisition hardware and the requisite projection models. The user’s hardware profile—comprising the Laowa 4mm circular fisheye and dual-lens 360° cameras—presents a unique set of geometric properties that deviate significantly from standard photogrammetric norms.

### **2.1 The Laowa 4mm f/2.8 Circular Fisheye**

The Laowa 4mm f/2.8 is a distinct optical instrument in the reconstruction toolkit due to its extreme field of view, which exceeds the standard 180° hemisphere, capturing up to 210°.1

#### **2.1.1 Optical Geometry and "Looking Behind"**

Unlike a standard 180° fisheye, which captures a complete hemisphere forward of the sensor plane, a 210° lens captures light rays originating from *behind* the sensor plane.

* **Geometric Implication:** In a photogrammetric context, this 30° of "back-facing" data (210° \- 180°) provides exceptional loop-closure capabilities. When a camera is rotated, features captured in the peripheral "look-behind" zone remain visible for a longer duration than in standard lenses. This increases the "track length" of tie points—the number of consecutive frames in which a feature is resolved—which directly correlates to the stability of the Bundle Adjustment solution in SfM software.2  
* **Sensor Coverage:** On a Micro Four Thirds (MFT) sensor, this lens produces a circular image inscribed within the frame. The corners of the sensor are completely black (no data). This necessitates strict masking protocols (discussed in Section 4\) to prevent SfM algorithms from detecting spurious features in the sensor noise of the unexposed corners.3

#### **2.1.2 Nodal Point and Parallax Management**

The entrance pupil (often conflated with the nodal point) of the Laowa 4mm is located extremely close to the front element. In panoramic photography, rotating around the entrance pupil minimizes parallax. However, in photogrammetry, *translation* is required to generate parallax for depth estimation.

* **The Challenge:** Due to the extreme FoV, objects very close to the lens (within 10-20cm) will exhibit massive parallax shifts relative to the background with even millimeter-level camera movements. While beneficial for depth resolution, this can confuse feature matchers if the movement is too erratic.  
* **Best Practice:** For reconstruction, a smooth, sliding translation is preferred over rotation-heavy movements, as the 210° FoV already ensures orientation coverage.

### **2.2 Dual-Lens 360° Camera Architectures (DJI & Insta360)**

The user's query highlights the use of the DJI Osmo 360 and Insta360 cameras. These devices are effectively "generalized stereo" rigs, but with opposing view vectors.

#### **2.2.1 The Fallacy of Equirectangular Stitching**

The standard consumer workflow for these cameras involves "stitching": taking the images from the front and rear lenses and warping them into a 2:1 lat-long (equirectangular) map.

* **Data Destruction:** This process involves blending pixels along the seam line (where the FoVs of the two lenses overlap). Since the optical centers of the two lenses are physically separated (by the thickness of the camera body, typically 1-2 cm), there is inevitable parallax. The stitching algorithm creates "ghosts" or blends features in this overlap zone.  
* **SfM Failure Mode:** Structure-from-Motion algorithms assume a single center of projection. An equirectangular image stitched from two offset lenses technically has *two* centers of projection. This violation of the central projection theorem introduces error into the camera pose estimation, leading to "drift" or misalignment in the final point cloud.5

#### **2.2.2 The Raw Data Approach**

To maximize reconstruction fidelity, one must treat the 360° camera not as a single omnidirectional sensor, but as **two separate monocular cameras** rigidly fixed back-to-back. This aligns with the user's intent to "explore paths that involve using the 360 footage as two 180 degree fisheye videos instead" \[User Query\]. By processing the streams independently, we maintain the geometric integrity of each lens's projection, allowing the SfM software to solve for the precise extrinsic offset between the two sensors.

---

## **3\. Data Engineering: Decoupling and Extraction**

The foundational step in this high-end workflow is the surgical extraction of raw optical data from the proprietary containers (.osv, .insv) used by DJI and Insta360. This requires a deep understanding of FFmpeg filter graphs and container specifications.

### **3.1 Analyzing Proprietary Containers**

The .osv (DJI) and .insv (Insta360) formats are wrappers around standard H.264 or H.265 (HEVC) video streams. However, the internal organization of these streams varies by camera generation and recording mode.

#### **3.1.1 Dual-Stream vs. Single-Stream Layouts**

* **Dual-Stream:** High-end modes often store the front and rear sensor data as two discrete video tracks within a single MP4 container. This is identified in FFmpeg as Stream \#0:0 and Stream \#0:1.  
* **Single-Stream (Side-by-Side):** Some modes, particularly on older units or specific framerates, commit the sensor data to a single video frame where the left half represents the front lens and the right half represents the rear lens (or a Top-Bottom arrangement).7

### **3.2 Advanced FFmpeg Extraction Workflows**

The user hypothesized that .osv files can be separated. This is correct, but the method depends on the internal layout identified above.

#### **Workflow A: Discrete Stream Extraction (Lossless)**

If ffmpeg \-i input.osv reveals two video streams, the extraction is trivial and mathematically lossless (no transcoding required). This is the ideal scenario for photogrammetry as it preserves the exact compression artifacts of the source without compounding them.

Bash  
\# Extract Front Lens (Stream 0\) to a separate file  
ffmpeg \-i input\_video.osv \-map 0:0 \-c copy \-f mp4 output\_front\_lens.mp4

\# Extract Rear Lens (Stream 1\) to a separate file  
ffmpeg \-i input\_video.osv \-map 0:1 \-c copy \-f mp4 output\_rear\_lens.mp4

* **Explanation:** The \-map 0:0 command isolates the first video stream. \-c copy instructs FFmpeg to copy the bitstream directly (video pass-through) without decoding and re-encoding. This guarantees 100% fidelity to the source.9

#### **Workflow B: Side-by-Side (SBS) Separation (Transcoding Required)**

If the file contains a single stream (e.g., resolution 5760x2880), the lenses are spatially multiplexed. We must use the crop filter to separate them. This necessitates re-encoding, so we must use a high-bitrate intermediate codec (ProRes or CRF 0\) to prevent compression artifacts that would confuse the SfM feature matchers.

**Filter Complex for Left/Right Separation:**

Bash  
ffmpeg \-i input\_sbs.mp4 \-filter\_complex \\  
"\[0:v\]crop=iw/2:ih:0:0\[front\]; \\  
 \[0:v\]crop=iw/2:ih:iw/2:0\[rear\]" \\  
\-map "\[front\]" \-c:v libx264 \-crf 0 \-preset veryslow output\_front\_lens.mp4 \\  
\-map "\[rear\]" \-c:v libx264 \-crf 0 \-preset veryslow output\_rear\_lens.mp4

* **Deep Dive:**  
  * crop=iw/2:ih:0:0: Crops a window with width \= InputWidth/2 and height \= InputHeight, starting at coordinates (0,0). This isolates the left hemisphere.  
  * crop=iw/2:ih:iw/2:0: Crops the same size window but starts at x \= InputWidth/2. This isolates the right hemisphere.  
  * \-crf 0: Constant Rate Factor 0 ensures lossless compression in the H.264 standard. This results in large files but preserves all edge details required for feature detection.11

### **3.3 Frame Extraction Strategy**

For the "Gradeeterna" workflow, working with image sequences is superior to video files. This allows for easier masking and color management in external tools like DaVinci Resolve.

**Optimal Extraction Command:**

Bash  
ffmpeg \-i output\_front\_lens.mp4 \-vf "fps=2" \-q:v 1 \-qmin 1 \-qmax 1 front\_%05d.jpg

* **FPS=2:** Capturing every frame (30fps or 60fps) provides too much overlap (small baseline), which degrades SfM accuracy due to lack of parallax. A rate of 2Hz to 5Hz is typically optimal for walking speeds.  
* **q:v 1:** Forces the highest JPEG quality. Alternatively, extracting to PNG (front\_%05d.png) is preferred to avoid compression artifacts entirely, though it increases storage requirements significantly.

---

## **4\. Color Science: The Rec.2020 to Rec.709 Pipeline**

The user explicitly highlighted Gradeeterna’s comment regarding color transformation: *"Transformed from Rec2020 to Rec709"*. This is a nuanced but critical step often overlooked in amateur reconstruction workflows.

### **4.1 The Problem with Log and Wide Gamut in NeRF**

Modern 360° cameras (like the Insta360 X3/X4 or DJI Osmo) typically capture in a "Flat" or "Log" profile, often encapsulated in a wide color gamut like Rec.2020. This is done to preserve dynamic range in high-contrast outdoor environments.

* **NeRF/Gaussian Splatting Implication:** These algorithms model scene radiance. If the input images are flat/logarithmic, the Spherical Harmonics (SH) coefficients learned by the Gaussian Splat will attempt to reproduce this flat, washed-out appearance. Furthermore, the lack of contrast in Log footage can hinder the SIFT (Scale-Invariant Feature Transform) algorithms in Metashape, which rely on local contrast gradients to identify keypoints.

### **4.2 Implementing the Gradeeterna Color Workflow**

Gradeeterna recommends cropping and transforming in **DaVinci Resolve**. This tool is the industry standard for color management.

**Step-by-Step Implementation:**

1. **Import:** Load the extracted fisheye image sequences (or video clips) into Resolve.  
2. **Circular Crop:** Apply a circular power window in the Color page to mask out the black sensor borders/vignetting. This ensures the histogram and auto-exposure calculations are based solely on the visual data, not the black void.  
3. **Color Space Transform (CST):**  
   * **Input Color Space:** Rec.2020 / DJI D-Gamut (check camera specs).  
   * **Input Gamma:** DJI D-Log / Insta360 Log.  
   * **Output Color Space:** Rec.709.  
   * **Output Gamma:** Gamma 2.4 (or sRGB).  
4. **Tone Mapping:** Apply a mild S-curve to expand contrast. This enhances the edges and textures, making them more "visible" to the feature matching algorithms in Metashape.13  
5. **Export:** Render the sequence as high-quality JPG or PNG/TIFF for Metashape.

---

## **5\. Agisoft Metashape: The Alignment Engine**

Agisoft Metashape (formerly PhotoScan) is the engine of choice for this workflow because of its robust support for the "Fisheye" camera model and its ability to handle "dirty" data (dynamic objects, lens flares) better than the stricter COLMAP pipeline.14

### **5.1 Masking: The "Gradeeterna" Essential**

The quote specifically mentions: "I cropped each fisheye circle in Resolve... and deal with the masking".

The Laowa 4mm and 360 camera modules project a circle. The corners are black.

* **The Problem:** If unmasked, Metashape will detect "features" in the sensor noise of the black corners. Since these noise patterns are static relative to the camera, the software will interpret them as points at infinity or lock the camera poses incorrectly.  
* **The Metashape Solution:**  
  1. Import the images.  
  2. Open one image. Use the **Magic Wand** or **Circle Selection** tool to select the black background.  
  3. Invert selection if necessary.  
  4. Right Click \> **Masks** \> **Export Masks**. Save this as a binary mask template.  
  5. Right Click \> **Masks** \> **Import Masks**. Apply this template to **all cameras** in the chunk.  
  6. **Dynamic Masking (Advanced):** For the "Gradeeterna" level of quality, you must also mask the photographer. Using automated segmentation tools like **SAM2 (Segment Anything Model 2\)** or **YOLO**, one can generate per-frame masks of the person holding the camera. These can be imported into Metashape (matching filenames image\_001.jpg \-\> image\_001\_mask.png) to ensure the operator is not reconstructed as a ghostly artifact in the final splat.5

### **5.2 Camera Model Configuration**

This is the most critical technical setting in the entire pipeline. Metashape uses a variation of the Brown-Conrady distortion model.

**Settings:**

* **Camera Type:** **Fisheye** (Do NOT use "Frame").  
* **Parameters:**  
  * **f (Focal Length):** Initialize with Auto.  
  * **cx, cy (Principal Point):** The center of the circle.  
  * **k1, k2, k3, k4 (Radial Distortion):** These parameters model the curvature. $k\_4$ is often necessary for 210° lenses to model the extreme edge compression.  
  * **p1, p2 (Tangential Distortion):** **CRITICAL ADVICE:** You should likely **lock these to 0**.  
    * *Reasoning:* COLMAP's standard OPENCV\_FISHEYE model (which 3DGS uses) usually does not account for tangential distortion in the same way Metashape does. If Metashape solves for non-zero $p\_1/p\_2$, you cannot easily export this to the format required for Gaussian Splatting without writing a complex custom lens shader. By locking them to 0, you force the solver to optimize the radial parameters ($k$) to account for all distortion, ensuring better compatibility with downstream tools.16

### **5.3 Alignment Strategy: "Without Markers"**

Gradeeterna notes aligning "without markers or multicam rig as they are only in Pro version".

This implies utilizing Metashape Standard’s image matching capabilities on the decoupled streams.

* **Workflow:** Import all front lens images and all rear lens images as a single "chunk". Metashape will treat them as independent cameras.  
* **Overlap:** Because the Laowa 4mm and 360 lenses have \>180° FoV, there is visual overlap between the front and rear views. Metashape will find features in this overlap zone (the rim of the fisheye circle) to lock the front and rear sequences together.  
* **Optimization:** Once aligned, select all cameras \-\> Optimize Cameras. Check "Fit k4" for the Laowa lens.18

---

## **6\. The Bridge to Radiance Fields: Export and Transformation**

Once alignment is complete, the data must be exported. The user quote mentions a "custom stuff" script to transform Metashape export to opencv\_fisheye. This addresses a gap in interoperability.

### **6.1 The Data Format Discrepancy**

* **Metashape Internal:** Uses a proprietary coordinate system and a specific fisheye polynomial.  
* **COLMAP / 3DGS:** Expects a cameras.txt defining the model as OPENCV\_FISHEYE with parameters f\_x, f\_y, c\_x, c\_y, k\_1, k\_2, k\_3, k\_4.  
* **The Conflict:** Standard Metashape export (prior to version 2.1) often exported cameras as PINHOLE or generic RADIAL, even if they were fisheye. This causes the 3DGS training to fail because it assumes straight rays.

### **6.2 Solution 1: Native Export (Metashape 2.1.3+)**

Recent updates to Metashape Standard have added native COLMAP support.

* **Command:** File \> Export \> Export Cameras \> Format: COLMAP (\*.txt).  
* **Validation:** Open the resulting cameras.txt. Verify the model ID is OPENCV\_FISHEYE. If it is, the "custom stuff" mentioned by Gradeeterna is no longer strictly necessary for basic compatibility, provided you locked $p\_1/p\_2$ to zero during alignment.19

### **6.3 Solution 2: The "Pinhole Transformation" (Alternative Path)**

Gradeeterna suggested: "I think you would get good results even just transforming to pinhole in Metashape if you have enough coverage."

This is a vital alternative for users who find the raw fisheye pipeline too complex or incompatible with specific NeRF implementations (like nerfacto).

**The Mechanism:**

1. Align images as **Fisheye** in Metashape (to get accurate poses).  
2. Use Tools \> Mesh \> Build Mesh (Low quality is fine).  
3. Use File \> Export \> Render Photos.  
   * This function allows you to re-project the aligned images.  
   * **Settings:** Select "System: Frame" (Pinhole).  
   * **FoV:** Set a reasonable FoV (e.g., 100°-120°).  
4. **Result:** Metashape will generate a new set of undistorted, rectilinear images.  
5. **Trade-off:**  
   * *Pros:* Perfectly compatible with all NeRF/3DGS software. No specialized fisheye shaders needed.  
   * *Cons:* **Massive Data Loss.** Rectifying a 210° fisheye to 100° pinhole discards the outer \~50% of the image circle. This destroys the "look behind" capability and reduces the loop closure strength. You lose the primary advantage of the fisheye lens.

### **6.4 Solution 3: The "Custom Stuff" (Python Scripting)**

For the highest fidelity, one uses a Python script in Metashape Pro to write a cameras.txt that maps the internal calibration directly to OPENCV\_FISHEYE.

**Key Mapping Logic for the Script:**

Python  
\# Conceptual Mapping  
colmap\_fx \= metashape\_calib.f  
colmap\_fy \= metashape\_calib.f  
colmap\_cx \= metashape\_calib.cx \+ image\_width / 2  
colmap\_cy \= metashape\_calib.cy \+ image\_height / 2  
colmap\_k1 \= metashape\_calib.k1  
colmap\_k2 \= metashape\_calib.k2  
colmap\_k3 \= metashape\_calib.k3  
colmap\_k4 \= metashape\_calib.k4  
\# Note: Metashape centers (0,0) at the image center. COLMAP centers (0,0) at top-left.

This script ensures that the exported dataset retains the full 210° data but labels it correctly for the training software.21

---

## **7\. Neural Rendering with Distortion: 3DGUT and 3DGRT**

The user requested research on "GUT". This refers to the **3D Gaussian Unscented Transform (3DGUT)**, often implemented alongside **3D Gaussian Ray Tracing (3DGRT)** in the **3DGRUT** framework developed by NVIDIA and Toronto AI Labs.23

### **7.1 Why Standard Splatting Fails with Fisheye**

Standard 3DGS uses **EWA (Elliptical Weighted Average) Splatting**. This relies on a local affine approximation (linearization) of the projection function to map the 3D Gaussian ellipsoid to a 2D Gaussian splat on the screen.

* **The Failure:** With a 210° fisheye, the projection is highly non-linear. A single Gaussian, if projected near the edge of the lens, should appear curved (like a banana shape). The affine approximation forces it to be a straight ellipse. This mismatch causes severe artifacts and blurring at the periphery of the image.

### **7.2 The 3DGUT Solution**

3DGUT replaces the EWA formulation with the **Unscented Transform (UT)**.

* **Mechanism:** The UT is a method for calculating the statistics of a random variable which undergoes a non-linear transformation. Instead of linearizing the function (like EWA), UT picks a set of sample points (sigma points) around the Gaussian, projects *each one* through the exact fisheye model, and then reconstructs the 2D Gaussian from these projected points.  
* **Result:** This allows for accurate splatting through highly distorted lenses without the computational cost of full ray tracing.

### **7.3 Implementation Workflow**

To use the Gradeeterna dataset with 3DGUT:

1. **Dataset Structure:** Organize the Metashape export into the COLMAP structure (/sparse/0/).  
2. **Camera Model:** Ensure cameras.txt defines OPENCV\_FISHEYE.  
3. **Training:** Use the NVIDIA 3DGRUT codebase.  
4. Bash

python train.py \--config-name apps/colmap\_3dgut.yaml \\  
dataset.source\_path=/path/to/fisheye\_data \\  
camera\_model=FISHEYE

5.   
   * 3DGRUT will natively read the $k\_1-k\_4$ parameters and apply the Unscented Transform during the forward pass.  
   * This allows the network to learn from the pixels at the very edge of the Laowa 4mm lens, utilizing the full 210° context.14

---

## **8\. Comparative Analysis of Reconstruction Pathways**

Based on the research, three distinct pathways emerge for the user's setup.

| Feature | Path A: The Gradeeterna Pure | Path B: The Pinhole Rectification | Path C: The Stitched Equirectangular |
| :---- | :---- | :---- | :---- |
| **Input Data** | Raw Fisheye Circles (Split Streams) | Raw Fisheye Circles (Split Streams) | Stitched Equirectangular Video |
| **Data Usage** | 100% of Sensor Data (210° FoV) | \~60% of Data (Cropped to \~110°) | 100% of Data (but distorted/ghosted) |
| **Alignment** | Metashape (Fisheye Model) | Metashape (Fisheye \-\> Render Pinhole) | COLMAP / Metashape (Spherical) |
| **Export Format** | OPENCV\_FISHEYE | PINHOLE | EQUIRECTANGULAR (rarely supported well) |
| **Rendering Engine** | **3DGUT / 3DGRT** (NVIDIA) | Standard gsplat / nerfstudio | Standard 3DGS (with artifacts) |
| **Pros** | Highest geometric fidelity; Loop closure from "look-behind"; No black box stitching. | Compatible with ALL software; Easier to process. | Easiest to capture/view. |
| **Cons** | Requires custom scripts or 3DGUT; Complex masking. | Massive loss of peripheral data; Less stable SfM. | Parallax ghosts; Polar stretching; Low quality. |

## **9\. Conclusion**

The transition to raw fisheye reconstruction represents the cutting edge of photogrammetry. For the user's specific hardware—the Laowa 4mm and separate 360° lens streams—the **Gradeeterna workflow coupled with NVIDIA's 3DGUT** is the scientifically superior path.

It requires abandoning the convenience of "stitching" in favor of treating the 360° camera as a rig of two independent, highly distorted sensors. By splitting the .osv containers using FFmpeg filter graphs, managing color via Rec.709 transformation, and utilizing Metashape's robust fisheye alignment (while locking tangential distortion), one prepares the data for the Unscented Transform in 3DGUT. This pipeline preserves the optical reality of the 210° lens, transforming the extreme distortion from a liability into a geometric asset for robust 3D reconstruction.

### **10\. References & Citations**

* **FFmpeg & Containers:**.7  
* **Optical Theory:**.1  
* **Metashape & Alignment:**.5  
* **Neural Rendering (GUT):**.14  
* **Color & Workflow:**.5

