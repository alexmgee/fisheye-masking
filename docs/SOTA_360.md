# State-of-the-art tools for 360° video masking and object removal

**The 2024-2025 landscape has fundamentally shifted toward unified foundation models.** SAM 2.1 and the newly released SAM 3 now provide video-native segmentation, YOLO11 delivers production-ready detection with segmentation, and ProPainter remains the gold standard for practical video inpainting—though diffusion-based DiffuEraser is emerging for complex scenes. However, 360°/equirectangular imagery still lacks native model support; the practical approach involves cubemap projection workflows using standard models.

---

## SAM family now dominates both image and video segmentation

The Segment Anything Model family has evolved dramatically, with SAM 2.1 becoming the go-to foundation for video segmentation and SAM 3 introducing revolutionary concept-based prompting.

**SAM 2.1** (released September 29, 2024) represents the current production recommendation. Built atop SAM 2's July 2024 release, it adds improved handling of visually similar objects, small objects, and occlusions. The December 2024 update introduced `torch.compile` support with `vos_optimized=True` for major speed gains—achieving **39-91 FPS** depending on model size. Four model variants range from 38.9M parameters (tiny) to 224.4M (large), available at https://github.com/facebookresearch/sam2.

**SAM 3** emerged November 2025 with a paradigm shift: **promptable concept segmentation**. Rather than point/box prompts, you can now specify "yellow school bus" or "people wearing red" and the 848M-parameter model detects, segments, and tracks all matching instances. It supports 270,000+ concepts and integrates with multimodal LLMs for complex queries. GitHub: https://github.com/facebookresearch/sam3.

The lightweight variants remain highly relevant for constrained deployments:

- **MobileSAM** (~10M params, 10ms/image) offers SAM-quality masks on mobile devices—5× faster than FastSAM while maintaining higher accuracy
- **EfficientSAM** (9.8M params, December 2023) achieved +4.1 AP over FastSAM through novel SAMI pretraining; its new **EfficientTAM** extension enables video segmentation
- **SAM-HQ** adds high-quality edge detail with minimal overhead (~0.5% more parameters); **HQ-SAM 2** beta released November 2024
- **Grounded-SAM 2** combines Grounding DINO 1.5/1.6 with SAM 2 for text-prompted video segmentation, supporting Florence-2 and DINO-X integrations

---

## YOLO11 is the recommended detector for production deployment

Ultralytics released **YOLO11** at YOLO Vision 2024 (September 30, 2024), establishing it as the new standard. Key improvements include C3k2 blocks and C2PSA spatial attention, delivering **22% fewer parameters** than YOLOv8m with higher mAP. Unlike YOLOv10 (May 2024, Tsinghua), YOLO11 supports the full task spectrum: detection, instance segmentation, pose estimation, oriented bounding boxes, and classification.

**YOLO11 segmentation benchmarks** on COCO show the yolo11x-seg achieving 54.7 box mAP and 43.4 mask mAP with just 62.1M parameters. Native tracking integration via BoT-SORT and ByteTrack enables frame-persistent object tracking:

```python
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")
results = model.track(source="video.mp4", tracker="bytetrack.yaml", persist=True)
```

**YOLOv12** (February 2025) introduced attention-centric architecture but runs ~9% slower than YOLOv10-N. **YOLO-World** (January 2024, Tencent) enables open-vocabulary detection via text prompts without retraining—achieving 35.4 AP on LVIS at 52 FPS. For fisheye-specific applications, **YOLOv9-FishEye** (https://github.com/tuansunday05/fe8kv9) adapts YOLOv9 with deformable convolutions trained on the FishEye8K dataset.

| Model | Release | Segmentation | Best Use Case |
|-------|---------|--------------|---------------|
| **YOLO11** | Sep 2024 | ✅ Excellent | Production recommendation |
| YOLOv10 | May 2024 | ❌ None | NMS-free edge deployment |
| YOLO-World | Jan 2024 | ✅ Good | Open-vocabulary needs |
| YOLOv9-FishEye | 2024 | ✅ | Fisheye/360 cameras |

---

## Video object segmentation has converged on SAM 2 with specialized alternatives

**SAM 2's video propagation** capability makes it the default starting point for mask tracking. The `SAM2VideoPredictor` propagates prompts through entire videos with streaming memory architecture, achieving 3× fewer required interactions than prior approaches. December 2024 updates added independent per-object inference for multi-object scenarios.

**Cutie** (CVPR 2024 Highlight, https://github.com/hkchengrex/Cutie) excels where SAM 2 struggles with distractors. Its object-level memory reading delivers +8.7 J&F improvement on MOSE over XMem while running 3× faster than DeAOT. From the same author as XMem, Cutie handles challenging occlusions and visually similar objects better than pixel-level-only approaches.

**DEVA** (https://github.com/hkchengrex/Tracking-Anything-with-DEVA) remains essential for open-world scenarios. Its decoupled architecture pairs task-specific image segmentation with universal temporal propagation, natively integrating Grounding DINO for text-prompted video segmentation of novel objects.

Newer entrants targeting specific limitations:

- **SAM2Long** (https://github.com/Mark12Ding/SAM2Long): Training-free memory tree achieving +3.0 J&F over SAM 2 for long videos with occlusion/reappearance
- **EdgeTAM** (CVPR 2025): 22× faster than SAM 2, running **16 FPS on iPhone 15 Pro Max** with CoreML export
- **EfficientTAM**: Lightweight ViT encoder enabling ~10 FPS on mobile devices

---

## Shadow detection requires combining specialized models with person masks

No models specifically target human shadows, but general shadow detection tools integrate well with person segmentation pipelines. The **Unveiling Deep Shadows benchmark** (https://github.com/xw-hu/Unveiling-Deep-Shadows, September 2024) provides unified evaluation and pretrained weights for 8+ methods.

**SDDNet** (ACM Multimedia 2023) leads accuracy benchmarks at 4.86 BER on SBU-Refine. For real-time needs, **FSDNet** (https://github.com/xw-hu/FSDNet) processes **134 images/second** with acceptable 6.80 BER. Instance-aware shadow detection via **SSISv2** (https://github.com/stevewongv/SSIS) explicitly associates shadows with their source objects—crucial for person-shadow pairing.

For shadow removal after detection, **HomoFormer** (CVPR 2024, https://github.com/jiexiaou/HomoFormer) achieves state-of-the-art 3.62 RMSE on SRD. The lightweight alternative **BMNet** requires only 0.58M parameters. **ShadowDiffusion** (CVPR 2023) offers diffusion-based removal but runs slower at ~3 images/second.

**Practical pipeline**: Detect persons with YOLO11-seg → Detect shadows with SDDNet → Use morphological operations to associate person masks with nearby shadow regions → Apply HomoFormer for removal if needed.

---

## ProPainter leads video inpainting but diffusion models are catching up

**ProPainter** (ICCV 2023, https://github.com/sczhou/ProPainter) remains the practical choice for video object removal with 6,400+ GitHub stars. Its dual-domain propagation merges flow-based pixel transfer with transformer generation, handling arbitrary-length videos through sub-video batching (default 80 frames). GPU requirements scale with resolution: 720×480 needs ~8GB with FP16, while 1280×720 requires ~25GB.

Memory optimization flags include `--fp16`, `--neighbor_length` (reduce local neighbors), and `--ref_stride` (reduce global references). Community forks offer streaming variants and WebUI interfaces.

**DiffuEraser** (January 2025, https://github.com/lixiaowen-xw/DiffuEraser) from Alibaba represents the cutting edge for large mask scenarios. Built on Stable Diffusion 1.5 with BrushNet architecture, it uses ProPainter as a prior model for initialization. While requiring 24GB+ VRAM and running slower than ProPainter, it handles complex backgrounds and large occlusions better—reportedly outperforming Runway's commercial inpainting.

**Text-guided video inpainting** is emerging via **CoCoCo** (AAAI 2024, https://github.com/zibojia/COCOCO) and **AVID** (CVPR 2024). These enable replacing objects with text-described content, useful for generative editing beyond simple removal.

| Tool | GPU Memory | Speed | Best For |
|------|-----------|-------|----------|
| **ProPainter** | 8-12GB (720p) | ~0.12s/frame | Object removal, production |
| DiffuEraser | 24GB+ | Slower | Large masks, complex scenes |
| CoCoCo | 24GB+ | Moderate | Text-guided generation |

---

## Matting tools enable precise edge refinement for clean compositing

**RobustVideoMatting** (https://github.com/PeterL1n/RobustVideoMatting, 8,800+ stars) remains the production standard for trimap-free video matting, delivering 4K@76FPS and HD@104FPS. Its recurrent architecture ensures temporal consistency across exports to PyTorch, TensorFlow, ONNX, and CoreML.

**MatAnyone** (CVPR 2025, https://github.com/pq-yang/MatAnyone) represents the newest advancement with target-assigned video matting. Given a first-frame mask (from SAM2), it propagates with consistent memory and region-adaptive fusion—outperforming RVM on challenging multi-person scenes with harmonized backgrounds.

For image matting and edge refinement of segmentation masks:

- **ViTMatte** (https://github.com/hustvl/ViTMatte) offers state-of-the-art trimap-based matting, integrated into Hugging Face Transformers as `VitMatteForImageMatting`
- **Matte Anything** (https://github.com/hustvl/Matte-Anything) chains SAM → GroundingDINO → auto-trimap → ViTMatte for zero-shot interactive matting
- **BRIA RMBG 2.0** (November 2024, https://github.com/Bria-AI/RMBG-2.0) provides commercially-licensed background removal at 1024×1024

**Edge refinement pipeline**: Generate coarse mask (SAM/YOLO) → Create trimap via dilation/erosion of edges → Apply ViTMatte → Result: pixel-perfect boundaries with hair/fur detail preserved.

---

## 360° and fisheye imagery require projection-based workarounds

**No production-ready segmentation or inpainting models exist for equirectangular projection.** The practical approach converts equirectangular to cubemap (6 perspective faces), processes each face with standard models, then carefully blends and reconverts.

For fisheye detection, **FishEye8K** (https://github.com/MoyoG/FishEye8K) provides 8,000 images with 157,000 bounding boxes and baseline YOLOv7-E6E models. **YOLOv9-FishEye** (https://github.com/tuansunday05/fe8kv9) adds RepNLSKBottleNeck and ELAN-DCNv2 modules specifically for fisheye distortion. The AI City Challenge 2024 Track 4 solutions demonstrate YOLOv9-E with InternImage ensembles achieving strong results.

**360° inpainting tools** remain research-stage. **PIINET** (https://github.com/swhan0329/panorama_image_inpainting) uses cubemap conversion with GAN-based inpainting. **Skybox AI** (https://skybox.blockadelabs.com/) offers production-quality 360° panorama generation with inpainting support.

**Recommended 360° pipeline architecture:**
1. Extract equirectangular frames → Convert to cubemap (py360convert, Facebook Transform360)
2. Process each cube face independently: SAM2 segmentation → ProPainter inpainting
3. Apply sphere-aware edge blending at seams
4. Reconvert to equirectangular

---

## Integration frameworks simplify deployment through visual and programmatic interfaces

**ComfyUI** has emerged as the leading visual workflow tool with comprehensive node packages:

- **ComfyUI-segment-anything-2** (Kijai): SAM 2.1 support with auto model download, video segmentation nodes
- **ComfyUI_ProPainter_Nodes**: Video inpainting integration for object removal workflows
- **ComfyUI-YOLO** (kadirnar): YOLOv5-v10 support with detection, segmentation, pose estimation
- **ComfyUI-Impact-Pack**: FaceDetailer with YOLO+SAM integration for production refinement

**Hugging Face Diffusers** provides straightforward Python inpainting:
```python
from diffusers import AutoPipelineForInpainting
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)
```

**Python deployment** for core tools:
```bash
pip install sam2 ultralytics diffusers
```

SAM 2.1 requires Python≥3.10 and torch≥2.5.1. ProPainter requires manual clone and runs under NTU S-Lab non-commercial license. Cloud deployment options include Replicate (xmem-propainter-inpainting model), Modal (SAM 2 examples), and RunPod for GPU instances.

---

## Conclusion: Building a practical 360° removal pipeline

The optimal stack for automated masking and removal in 360° video combines proven tools through cubemap projection:

**Detection/Segmentation**: SAM 2.1 for prompted masks or YOLO11-seg for automatic person detection. Use Grounded-SAM 2 when text prompts simplify interaction. For fisheye sources, apply YOLOv9-FishEye or ensure cubemap conversion first.

**Shadow handling**: Run SDDNet in parallel with person detection, associate masks via SSISv2 or morphological proximity, apply HomoFormer for removal when needed.

**Video propagation**: SAM 2.1's native video predictor handles most cases; switch to Cutie for scenes with visually similar distractors or SAM2Long for extended sequences.

**Inpainting**: ProPainter for production deployment (8-12GB GPU), DiffuEraser when large masks defeat flow-based approaches.

**Edge refinement**: ViTMatte with auto-generated trimaps from coarse masks, or MatAnyone for video-consistent alpha extraction.

**360° workflow**: Cubemap projection → per-face processing → sphere-aware seam blending. No shortcuts exist for native equirectangular processing—the projection approach is mature and reliable.

All core tools are open-source with active development. The primary licensing constraint is ProPainter's non-commercial S-Lab license; commercial deployments may need to evaluate DiffuEraser (Apache 2.0) or custom alternatives despite higher compute requirements.