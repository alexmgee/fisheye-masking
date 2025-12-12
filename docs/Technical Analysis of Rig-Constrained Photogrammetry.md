# **Technical Analysis of Rig-Constrained Photogrammetry: Optimizing 360° Video Reconstruction for Gaussian Splatting**

## **1\. Introduction: The Intersection of Spherical Imaging and Neural Rendering**

The rapid proliferation of consumer-grade 360° cameras, such as the DJI Osmo and Insta360 series, has democratized the capture of omnidirectional visual data. Simultaneously, the emergence of Neural Radiance Fields (NeRF) and, more recently, 3D Gaussian Splatting (3DGS), has revolutionized the synthesis of photo-realistic 3D scenes. However, bridging the gap between raw spherical video and high-fidelity neural rendering remains a non-trivial engineering challenge. Gaussian Splatting algorithms, while robust in synthesizing novel views, are fundamentally dependent on the accuracy of the input Structure-from-Motion (SfM) data. The quality of the sparse point cloud and the precision of the estimated camera intrinsics and extrinsics act as the deterministic ceiling for the final visual fidelity of the splat.  
This report provides an exhaustive technical analysis of a specialized workflow: transforming 360° video footage into a rig-constrained photogrammetric reconstruction using **COLMAP 3.13** and **GLOMAP 1.2**. The specific scenario involves a dataset of 412 equirectangular frames decomposed into \~4,957 pinhole images across 12 distinct perspective views. This configuration constitutes a "virtual camera rig," where 12 synthetic sensors are fixed in a static geometric relationship relative to a moving center.  
The analysis is structured to guide a domain expert through the theoretical underpinnings, architectural implementations, and practical execution of this pipeline. We perform a deep dive into the mathematical formulation of rig constraints within Bundle Adjustment (BA), the operational differences between Incremental (COLMAP) and Global (GLOMAP) mapping, and the precise configuration required to enforce dodecahedral geometry during reconstruction. Furthermore, we examine the downstream implications for Gaussian Splatting, ensuring that the rigorous geometric consistency achieved via rig constraints translates into temporally stable and artifact-free neural representations.

### **1.1 The Challenge of Spherical Data in Photogrammetry**

Traditional photogrammetric pipelines are predicated on the pinhole camera model, which maps 3D world points onto a 2D planar sensor via perspective projection. Omnidirectional cameras, however, capture light on a spherical manifold. To store this data, it is projected onto a 2D plane, typically using an equirectangular (latitude-longitude) projection.  
While convenient for storage, equirectangular projection introduces severe non-linear distortion, particularly near the poles. Standard local feature descriptors used in SfM—such as the Scale-Invariant Feature Transform (SIFT)—are designed to be invariant to uniform scaling and rotation, but not to the anisotropic stretching inherent in spherical projections. Consequently, feature matching performance degrades significantly when comparing regions with different distortion characteristics (e.g., matching a feature at the equator in frame $t$ to a feature near the pole in frame $t+1$).  
To circumvent this, the industry standard practice, and the method employed in this user scenario, is to tessellate the sphere into multiple planar perspective views. A decomposition into **12 views** suggests a geometric arrangement based on a Platonic solid, likely a dodecahedron. This creates 12 "virtual cameras" with overlapping fields of view (FOV) that cover the entire sphere. Unlike a physical multi-camera rig where sensors have physical baselines (parallax), this virtual rig has a baseline of zero; all 12 cameras share the exact same optical center. This unique property simplifies the translation component of the rig calibration to zero but places extreme importance on the precise estimation of rotational constraints.

### **1.2 The Paradigm Shift: Incremental vs. Global SfM**

The user has specifically requested the integration of **GLOMAP 1.2**. This choice reflects an awareness of the limitations of traditional incremental SfM for video data.  
**Incremental SfM (COLMAP Mapper):**

* **Mechanism:** Starts with an initial two-view reconstruction (a seed pair) and sequentially registers new images one by one.  
* **Vulnerability:** In video sequences, especially linear trajectories (e.g., a camera moving down a street), errors in pose estimation accumulate over time. This phenomenon, known as "drift," can cause a straight street to curve or a flat floor to bow.  
* **Cost:** The computational cost grows significantly as the model expands, requiring repeated Bundle Adjustment (BA) steps to redistribute error.

**Global SfM (GLOMAP):**

* **Mechanism:** Decouples the estimation of rotations and translations. It first estimates the global rotation of all cameras simultaneously by solving a rotation averaging problem on the view graph. Subsequently, it solves for camera translations and scene structure.  
* **Advantage:** By considering all pairwise constraints simultaneously, global SfM is inherently more robust to drift and significantly faster (often 1-2 orders of magnitude) for large datasets.  
* **Rig Integration:** The recent inclusion of rig constraints in GLOMAP (Pull Request \#201) is critical. Without it, the global solver would treat the 4,957 images as independent, potentially leading to a chaotic solution where the 12 views of a single frame explode outwards or fail to maintain their spherical arrangement. The rig constraint condenses the problem, effectively reducing the number of unknown poses from 4,957 to 412 (the trajectory of the rig), while enforcing the rigidity of the 12-view cluster.

## ---

**2\. Geometric Foundations: The Virtual Dodecahedral Rig**

To configure the software correctly, we must first establish the rigorous mathematical definition of the user's data structure. The dataset consists of 412 time-steps. At each time step $t$, we have 12 images, $I\_{t,1}$ through $I\_{t,12}$.

### **2.1 The Geometry of the Split**

A 12-view split of a sphere is optimal when arranged according to the faces of a regular dodecahedron. A dodecahedron provides a more uniform sampling of the spherical surface compared to a cube map (6 views), reducing the maximum angle of incidence and thus minimizing perspective distortion at the edges of the image planes.  
For a regular dodecahedron centered at the origin, the face centers (which correspond to the optical axes of our 12 virtual cameras) can be described using the golden ratio $\\phi \= \\frac{1+\\sqrt{5}}{2} \\approx 1.6180339887$.  
The 12 normal vectors $\\mathbf{n}$ corresponding to the face centers are cyclical permutations of:

1. $(0, \\pm \\phi, \\pm \\frac{1}{\\phi})$  
2. $(\\pm \\frac{1}{\\phi}, 0, \\pm \\phi)$  
3. $(\\pm \\phi, \\pm \\frac{1}{\\phi}, 0)$

There are exactly 12 such vectors. To create the rig configuration, we must define the rotation matrix $R\_{rig \\to sensor}$ that rotates the rig's canonical coordinate system (usually aligned with Camera 1\) to align with each of these 12 normal vectors.

### **2.2 Coordinate System Transformations**

In COLMAP, the transformation from the Rig frame to the Camera (Sensor) frame is defined as:

$$\\mathbf{x}\_{cam} \= R\_{rel} \\mathbf{x}\_{rig} \+ \\mathbf{t}\_{rel}$$

Where:

* $\\mathbf{x}\_{rig}$ is a point in the rig's coordinate system.  
* $\\mathbf{x}\_{cam}$ is the same point expressed in the specific sensor's coordinate system.  
* $R\_{rel}$ (Rotation) and $\\mathbf{t}\_{rel}$ (Translation) are the static rig parameters.

For a virtual split of a 360° image:

* **$\\mathbf{t}\_{rel}$ is strictly $$**. Since the images are generated from a single point of projection (the nodal point of the physical 360 camera), there is no physical baseline between the 12 virtual views.  
* **$R\_{rel}$** is the rotation required to look in the direction of the dodecahedron face.

If we assume Camera 1 points along the $+Z$ axis (Identity rotation), then for Camera $k$, the rotation $R\_k$ must rotate the vector $^T$ to align with the face normal $\\mathbf{n}\_k$.

### **2.3 Quaternion Representation for Configuration**

The rig\_config.json file used by COLMAP requires rotations to be specified as quaternions in the order $\[q\_w, q\_x, q\_y, q\_z\]$.  
The formula to calculate the rotation quaternion $q$ between two unit vectors $\\mathbf{u}$ (source, e.g., camera optical axis $$) and $\\mathbf{v}$ (destination, e.g., face normal) is derived from the axis-angle representation:

* **Rotation Axis:** $\\mathbf{w} \= \\mathbf{u} \\times \\mathbf{v}$  
* **Rotation Angle:** $\\theta \= \\arccos(\\mathbf{u} \\cdot \\mathbf{v})$

The quaternion components are:

$$q\_w \= \\cos(\\frac{\\theta}{2})$$

$$ \\begin{bmatrix} q\_x \\ q\_y \\ q\_z \\end{bmatrix} \= \\sin(\\frac{\\theta}{2}) \\frac{\\mathbf{w}}{|\\mathbf{w}|} $$  
It is imperative that the user verifies the exact orientation used by their splitting script. If the script simply used Euler angles (e.g., yaw/pitch steps), those must be converted to quaternions. A mismatch between the rig\_config.json and the actual image content will cause the reconstruction to fail catastrophically, as the feature matches will essentially contradict the geometric priors enforced by the rig.  
Table 1: Example Euler Angles for a 12-View Spherical Split (Hypothetical Grid Layout)  
Note: If the user used a grid layout instead of a dodecahedron, the angles might look like this:

| Camera ID | Pitch (ϕ) | Yaw (θ) |
| :---- | :---- | :---- |
| 1 | $+30^\\circ$ | $0^\\circ$ |
| 2 | $+30^\\circ$ | $60^\\circ$ |
| ... | ... | ... |
| 7 | $-30^\\circ$ | $0^\\circ$ |
| ... | ... | ... |

If the exact angles are unknown, the **"Unknown Rig"** workflow (learning the rig from SfM) must be employed, which will be detailed in Section 4\.

## ---

**3\. Architecture of COLMAP 3.13 and GLOMAP 1.2**

Understanding the software architecture is essential for troubleshooting and optimization. The user is operating with specific versions: COLMAP 3.13 (the latest stable release as of late 2024/early 2025\) and GLOMAP 1.2 (a very recent release incorporating rig support).

### **3.1 COLMAP 3.13 Database Internals**

COLMAP relies on a SQLite database (database.db) to store all session data.

* **Table cameras**: Stores intrinsic parameters. For this project, there should be exactly **12 rows** in this table if the user sets single\_camera\_per\_folder=1. Each row corresponds to one of the 12 virtual lenses. Since these are virtual pinholes, the distortion parameters ($k\_1, k\_2, p\_1, p\_2$) should be zero. The model should be SIMPLE\_PINHOLE or PINHOLE.  
* **Table images**: Stores the extrinsic parameters ($R, t$) and the link to the camera ID. This table will contain \~4,957 rows.  
* **Table matches**: Stores the raw feature correspondences between image pairs.

Rig Implementation in Database:  
It is crucial to note that rig constraints are not stored as a persistent table in the standard COLMAP schema in a way that the Mapper automatically respects without flags.

* The rig\_configurator tool modifies the database to potentially add specific identifiers or grouping logic, but primarily it is used to generate a loaded Reconstruction object in memory that has rig constraints applied.  
* When using the CLI, the rig configuration is often passed via a JSON file or implied by the database state after running rig\_configurator.

### **3.2 GLOMAP 1.2 and PR \#201**

The integration of rig support into GLOMAP (Pull Request \#201, mentioned in the user prompt 1) is a significant architectural update.  
How GLOMAP Handles Rigs:  
In a standard Global SfM pipeline, the View Graph consists of $N$ nodes (images) and $M$ edges (matches). The rotation averaging algorithm seeks to find global rotations $R\_i$ that minimize the relative rotation error $E\_{ij} \= \\| R\_{ij} \- R\_j R\_i^T \\|$.  
With rig support:

1. **Node Collapse/Constraint:** GLOMAP identifies that images $\\{I\_{t,1},..., I\_{t,12}\\}$ belong to Rig Frame $t$.  
2. **Parameter Reduction:** Instead of solving for 12 independent rotations $R\_{t,k}$, it solves for a single rig rotation $R\_{rig,t}$. The constraint $R\_{t,k} \= R\_{rel,k} R\_{rig,t}$ is hard-coded into the solver.  
3. **Graph Strength:** This dramatically strengthens the view graph. Feature matches between Frame $t$ and Frame $t+1$ are aggregated. A match between $I\_{t,1}$ and $I\_{t+1,1}$ provides a constraint on the rig motion, as does a match between $I\_{t,2}$ and $I\_{t+1,2}$. This redundancy makes the rotation estimation extremely robust to outliers and noise.

Constraint Rigidity:  
Snippet 2 mentions Mapper.ba\_refine\_sensor\_from\_rig 0\. This flag determines whether the rig is "hard" (perfectly rigid) or "soft" (allows for slight calibration refinement).

* For **Physical Rigs**: Soft constraints are often preferred to account for thermal expansion or mechanical flex.  
* For **Virtual Rigs (User's Case)**: **Hard constraints** are mandatory. The 12 views are mathematically generated from a single file; they cannot "flex." Any deviation is a calculation error, not a physical reality. Therefore, refinement must be disabled (0).

## ---

**4\. Comprehensive Workflow Implementation**

This section details the step-by-step execution of the pipeline, translating the theoretical requirements into actionable CLI commands.

### **4.1 Step 1: Data Structuring and Feature Extraction**

**Objective:** Organize the 4,957 images so COLMAP recognizes them as 12 distinct cameras, not 4,957 distinct cameras.  
Directory Structure:  
/dataset/  
/images/  
/cam01/  
frame\_0001.jpg  
frame\_0002.jpg  
...  
/cam02/  
frame\_0001.jpg  
...  
...  
/cam12/  
frame\_0001.jpg  
...  
Constraint: Filenames must be identical across folders for the same timestamp (e.g., frame\_0001.jpg in all 12 folders). This allows rig\_configurator to automatically group them by name.  
**Command: Feature Extraction**

Bash

colmap feature\_extractor \\  
    \--database\_path /dataset/database.db \\  
    \--image\_path /dataset/images \\  
    \--ImageReader.single\_camera\_per\_folder 1 \\  
    \--ImageReader.camera\_model PINHOLE \\  
    \--SiftExtraction.use\_gpu 1

**Rationale:**

* single\_camera\_per\_folder 1: This forces COLMAP to create only 12 unique camera intrinsics. This is critical. If set to 0, COLMAP creates 4,957 intrinsics, exploding the parameter space and likely causing "intrinsics drift" where the focal length of frame 100 varies wildly from frame 1\.  
* camera\_model PINHOLE: Since the images are rectified sections of a sphere, they follow a linear projection model.

### **4.2 Step 2: Defining the Rig Configuration**

We must populate the rig\_config.json. There are two approaches: Deterministic (if angles known) and Empirical (if angles unknown).

#### **Approach A: Deterministic (Manual JSON Construction)**

If the user knows the Euler angles used to split the sphere, they should calculate the quaternions and build the JSON.  
**JSON Structure:**

JSON

\[  
  {  
    "ref\_camera\_id": 1,  
    "cameras": \[  
      {  
        "camera\_id": 1,  
        "image\_prefix": "cam01/",  
        "ref\_sensor": true  
      },  
      {  
        "camera\_id": 2,  
        "image\_prefix": "cam02/",  
        "cam\_from\_rig\_rotation": \[0.9659, 0, 0.2588, 0\],   
        "cam\_from\_rig\_translation":   
      },  
     ...  
      {  
        "camera\_id": 12,  
        "image\_prefix": "cam12/",  
        "cam\_from\_rig\_rotation": \[qw, qx, qy, qz\],  
        "cam\_from\_rig\_translation":   
      }  
    \]  
  }  
\]

* ref\_camera\_id: Defines which of the 12 cameras acts as the anchor for the rig.  
* image\_prefix: Matches the folder names in /dataset/images/.

#### **Approach B: Empirical (Learning the Rig)**

If the precise rotations are unknown, we use COLMAP to reverse-engineer them.

1. **Subset Selection:** Select \~10 frames where the environment is rich in features (e.g., outdoors, high contrast).  
2. **Unconstrained Reconstruction:** Run automatic\_reconstructor or mapper on just these images *without* rig constraints.  
   Bash  
   colmap mapper \\  
       \--database\_path database\_calib.db \\  
       \--image\_path images\_subset \\  
       \--output\_path sparse\_calib

3. **Rig Extraction:** Use the rig\_configurator to analyze the relative poses in this sparse model.  
   Bash  
   colmap rig\_configurator \\  
       \--database\_path database\_calib.db \\  
       \--input\_path sparse\_calib/0 \\  
       \--rig\_config\_path rig\_config\_learned.json \\  
       \--output\_path sparse\_rigged

   This command calculates the average relative rotation between the cameras across the reconstructed frames and writes the rig\_config\_learned.json file. This file can then be used for the full dataset.

### **4.3 Step 3: Feature Matching**

Matching strategy is vital for video. Matching every image against every other (Exhaustive) is $O(N^2)$. For 5,000 images, this is $25,000,000$ pairs—computationally heavy but feasible with GLOMAP's efficiency, though the matching step itself (COLMAP) will be the bottleneck.  
Strategy 1: Sequential Matching (Recommended for Video)  
Matches frame $t$ against $t \\pm \\delta$.

Bash

colmap sequential\_matcher \\  
    \--database\_path /dataset/database.db \\  
    \--SequentialMatching.overlap 20 \\  
    \--SequentialMatching.loop\_detection 1 \\  
    \--SequentialMatching.vocab\_tree\_path /path/to/vocab\_tree.bin

**Critical Nuance:** sequential\_matcher typically assumes file names are ordered temporally.

* **Problem:** With folders (cam01/frame001, cam02/frame001), a standard alphanumeric sort might list all of cam01 then all of cam02. The matcher might try to match cam01/frame412 with cam02/frame001, which is wrong.  
* **Workaround:** Ensure the database ingest order or the matching logic respects the temporal overlap. The safest bet for this specific folder structure is often **Vocab Tree Matching** or **Transitive Matching**.

Strategy 2: Vocabulary Tree Matching (Robust)  
This ignores filenames and matches based on visual similarity using a pre-trained index.

Bash

colmap vocab\_tree\_matcher \\  
    \--database\_path /dataset/database.db \\  
    \--VocabTreeMatching.vocab\_tree\_path vocab\_tree\_flickr100k\_words.bin \\  
    \--VocabTreeMatching.match\_list\_path match\_list.txt

*Performance Note:* For 5,000 images, this is very efficient and handles loop closures (e.g., the camera returning to the start position) automatically.

### **4.4 Step 4: Configuring the Rig in the Database**

Before running GLOMAP, we must apply the rig configuration to the database.

Bash

colmap rig\_configurator \\  
    \--database\_path /dataset/database.db \\  
    \--rig\_config\_path /dataset/rig\_config.json

**Verification:** This step does not output a model. It modifies the database.db. You cannot easily "see" the result without inspecting the database tables, but it prepares the data for GLOMAP's loader.

### **4.5 Step 5: Global Reconstruction with GLOMAP**

This is the execution of the user's primary requirement: using GLOMAP 1.2 with Rig Support.

Bash

glomap mapper \\  
    \--database\_path /dataset/database.db \\  
    \--image\_path /dataset/images \\  
    \--output\_path /dataset/sparse/glomap

Interpretation of Output:  
GLOMAP will output a directory structure compatible with COLMAP:

* cameras.bin: The optimized intrinsics for the 12 cameras.  
* images.bin: The optimized poses for all 4,957 images.  
* points3D.bin: The sparse point cloud.

**Important:** Even though the solver used rig constraints, the output images.bin contains the **absolute pose** of every single image. This is exactly what Gaussian Splatting training requires. It does not output a "rig path" file; it bakes the rig transform into the individual image poses.

### **4.6 Step 6: Rigidity Enforcement (Bundle Adjustment)**

GLOMAP provides a global solution, but standard GLOMAP (as of v1.2) might optimize the rig parameters significantly or treat them as soft constraints during the translation averaging. To ensure the virtual rig is perfectly preserved (no "bending" of the virtual dodecahedron), it is advisable to run a final pass of **COLMAP Bundle Adjustment**.

Bash

colmap bundle\_adjuster \\  
    \--input\_path /dataset/sparse/glomap \\  
    \--output\_path /dataset/sparse/refined \\  
    \--BundleAdjustment.refine\_sensor\_from\_rig 0 \\  
    \--BundleAdjustment.refine\_focal\_length 0 \\  
    \--BundleAdjustment.refine\_principal\_point 0 \\  
    \--BundleAdjustment.refine\_extra\_params 0

**Flags Explained:**

* refine\_sensor\_from\_rig 0: **Hard Constraint.** Do not change the relative rotations/translations defined in the JSON.  
* refine\_focal\_length 0: Trust the pinhole extraction logic. Refining focal length for virtual cameras can sometimes lead to instability if the texture is low.

## ---

**5\. Integration with Gaussian Splatting**

The transition from SfM output to Gaussian Splatting (3DGS) training involves data conversion and handling specific artifacts inherent to spherical capture.

### **5.1 Converter Pipeline**

Standard 3DGS implementations (Inria, Nerfstudio) use a Python script (typically convert.py or ns-process-data) to ingest COLMAP data.  
The "Skip Matching" Flag:  
Since we have already performed a high-quality reconstruction using GLOMAP, we must instruct the converter to use our existing data rather than running COLMAP again (which would default to the inferior Incremental mapper without rig constraints).

Bash

python convert.py \\  
    \--source\_path /dataset/ \\  
    \--skip\_matching \\  
    \--model\_path /dataset/sparse/refined

* \--source\_path: Root folder containing the images folder.  
* \--model\_path: Point this to the output of our bundle\_adjuster step.

### **5.2 Handling Virtual Rig Artifacts in 3DGS**

360° rig data presents unique challenges for volumetric rendering.  
1\. The "Singularity" Floater:  
In a virtual rig, all 12 cameras have an optical center at $(0,0,0)$ relative to the rig. However, in the 3DGS coordinate space, this results in a trajectory of camera centers. If the near-plane clipping of the training camera is set too low, the renderer may attempt to reconstruct the "inside" of the camera rig, resulting in a dense cloud of floaters or a black sphere artifact at the camera path.

* **Mitigation:** Increase the near-plane clipping distance in the training configuration.

2\. Background Explosion:  
360° cameras capture the entire environment, including the sky and distant horizon. SfM will triangulate points at infinity (or very far distances). Gaussian Splatting attempts to model these distant points with Gaussians.

* **Issue:** Thousands of Gaussians may be allocated to the sky, wasting memory and VRAM.  
* **Mitigation:**  
  * **Sky Masking:** Use semantic segmentation to mask out the sky in the input images.  
  * **Background Model:** Use a 3DGS variant that supports a separate background model (e.g., a skybox NeRF combined with foreground Gaussians).

3\. The "Seam" Problem:  
Even with perfect rig constraints, there can be slight discontinuities in auto-exposure or white balance between the 12 split views if the original ISP (Image Signal Processor) of the Osmo/Insta360 did not harmonize them perfectly.

* **Effect:** The Gaussian Splat may learn "seams" or lines in the sky where the color shifts.  
* **Mitigation:** Ensure the input equirectangular video is globally color-corrected before splitting.

## ---

**6\. Optimization and Troubleshooting**

### **6.1 Common Failure Modes**

**1\. Initialization Failure:**

* *Symptom:* GLOMAP returns a model with very few images registered, or multiple disconnected components.  
* *Cause:* Insufficient overlap between the 12 views in the split, or lack of rotation in the video (pure translation).  
* *Fix:* Check the FOV of the pinhole extraction. Ensure there is at least 15-20% overlap between adjacent views in the dodecahedron.

**2\. Scale Drift:**

* *Symptom:* The reconstructed trajectory looks like a funnel (expanding or contracting).  
* *Cause:* Monocular scale ambiguity. Even with a rig, if the "stride" of the walker changes, scale can drift.  
* *Fix:* Use **Loop Closure**. Ensure the recording path loops back to the start. The vocab\_tree\_matcher is essential here to detect the loop and snap the scale back.

**3\. "Bent" Rig:**

* *Symptom:* The floor appears curved.  
* *Cause:* Radial distortion parameters were optimized despite being a virtual pinhole.  
* *Fix:* Ensure refine\_extra\_params is set to 0 in Bundle Adjustment. The virtual lens is perfect; do not let COLMAP try to "un-distort" it.

### **6.2 Comparison with Other Approaches**

Why use this complex 12-camera rig pipeline instead of alternatives?

* **vs. OpenMVG Spherical:** OpenMVG supports native spherical SfM. However, converting OpenMVG output to the specific format required by 3DGS training codes is often more cumbersome than using the COLMAP-native workflow, as most 3DGS parsers are hard-coded for COLMAP binaries.  
* **vs. 6-View Cube Map:** As established, the 12-view dodecahedron minimizes distortion, leading to higher-quality feature maps and denser point clouds, which directly benefits the initialization of Gaussians.

## **7\. Conclusion**

The transformation of Osmo 360 footage into a high-fidelity Gaussian Splat requires a disciplined adherence to geometric principles. By treating the decomposed video as a **Rigid Multi-Camera System**, we leverage the constraints of the physical world (the fixed nature of the lens assembly) to constrain the optimization problem.  
The combination of **COLMAP 3.13** (for rigorous database management and feature extraction) and **GLOMAP 1.2** (for scalable, global reconstruction) represents the current state-of-the-art in open-source photogrammetry. The "Global" nature of GLOMAP is particularly well-suited to the closed-loop trajectories common in 360 video recording, effectively mitigating drift.  
For the user, the critical path to success lies in the **rig\_config.json**. This file is the bridge between the pixel data and the solver. It must mathematically mirror the splitting script used to generate the pinhole images. With this constraint correctly applied, and the subsequent "Hard Constraint" bundle adjustment, the resulting sparse model provides the ideal foundation—accurate poses, consistent scale, and dense structure—for training photorealistic 3D Gaussian Splats.

#### **Works cited**

1. Activity · colmap/glomap \- GitHub, accessed December 6, 2025, [https://github.com/colmap/glomap/activity?ref=main](https://github.com/colmap/glomap/activity?ref=main)  
2. Rig Support — COLMAP 3.14.0.dev0 | 5b9a079a (2025-11-14) documentation, accessed December 6, 2025, [https://colmap.github.io/rigs.html](https://colmap.github.io/rigs.html)