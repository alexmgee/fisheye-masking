---
description: Test masking workflow on a new OSV dataset
---

# Test Masking Workflow

This workflow processes a new OSV dataset through the complete masking pipeline for testing/validation.

## Prerequisites
- OSV file from Insta360 camera
- Project folder created in `projects/<dataset_name>/`

## Step 1: Split OSV into Front/Rear Videos

```bash
python scripts/split_osv.py <path_to_osv_file> <output_directory>
```

This creates:
- `<basename>_front.mp4`
- `<basename>_rear.mp4`

## Step 2: Extract Frames from Rear Video

// turbo
```bash
python scripts/extract_frames.py <rear_video_path> <frames_output_dir> --max-frames 150
```

This extracts up to 150 frames from the rear video.

## Step 3: Run Masking on Subsample (Test Run)

// turbo
```bash
python scripts/masking_v2.py <frames_dir> <output_dir> \
    --model yolo-sam3-shadow \
    --geometry fisheye \
    --save-review \
    --pattern "*[02468]0.jpg"
```

The pattern `*[02468]0.jpg` samples ~25 frames (every 10th frame ending in 0, 20, 40, 60, 80).

## Step 4: Review Results

Check the output:
- `output/masks/` - Generated masks
- `output/review/` - Review images with overlay
- `output/flagged/` - Images needing manual review (symlinks)

## Step 5: Manual Review (if needed)

// turbo
```bash
python scripts/review_gallery.py <output_dir>/flagged/
```

## Step 6: Full Run (after validation)

// turbo
```bash
python scripts/masking_v2.py <frames_dir> <output_dir> \
    --model yolo-sam3-shadow \
    --geometry fisheye \
    --save-review
```

---

## Example: ogallala_test

```bash
# Step 1: Split OSV
python scripts/split_osv.py projects/ogallala_test/CAM_20251112155400_0001_D.OSV projects/ogallala_test/

# Step 2: Extract frames
python scripts/extract_frames.py projects/ogallala_test/CAM_20251112155400_0001_D_rear.mp4 projects/ogallala_test/frames_rear --max-frames 150

# Step 3: Test run (25 frames)
python scripts/masking_v2.py projects/ogallala_test/frames_rear projects/ogallala_test/output_test --model yolo-sam3-shadow --geometry fisheye --save-review --pattern "*[02468]0.jpg"
```
