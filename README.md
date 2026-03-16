# Omni-Scan: In-Hand Gaussian Splatting

This is the official code release for **Omni-Scan**, presented at IROS 2025.

> **Omni-Scan: Visually-Accurate Digital Twin Object Modeling using a Bimanual Robot with Handoff and Gaussian Splat Merging**
> Tianshuang Qiu\*, Zehan Ma\*, Karim El-Refai\*, Hiya Shah, Justin Kerr, Chung Min Kim, Ken Goldberg
> UC Berkeley
> [[Project Page]](https://berkeleyautomation.github.io/omni-scan/) [[Paper]](https://berkeleyautomation.github.io/omni-scan/)

---

## Overview

Omni-Scan creates visually-accurate 3D digital twin models of objects by scanning them with a bimanual robot. One arm captures views from one side; the object is then handed off to the other arm to capture views of previously occluded surfaces. The two scans are merged into a unified 3D Gaussian Splatting model for downstream tasks such as visual and geometric inspection.

This repository contains the **nerfstudio plugin** (`inhand`) that implements the IHGS (In-Hand Gaussian Splatting) training method, along with the full pipeline scripts for training, merging, and evaluating models.

---

## Installation

1. Install [nerfstudio](https://docs.nerf.studio/quickstart/installation.html).

2. In `nerfstudio/data/dataparsers/nerfstudio_dataparser.py`, set:
   ```python
   MAX_AUTO_RESOLUTION = 3840
   ```

3. Install this package:
   ```bash
   pip install -e .
   ```

---

## Data Format

Each dataset should be organised as:

```
<data_path>/
  left/
    images/
    masks/
  right/
    images/
    masks/
```

Images are the raw RGB frames; masks are single-channel grayscale JPGs indicating the object region (see `scripts/convert_masks.py` to convert PNG masks).

Before training, generate `transforms.json` for each view using the robot pose data from your capture system.

---

## Training

Run the full pipeline with:

```bash
python scripts/train.py <data_path>
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--gpu-0` | `0` | GPU for the left-view model |
| `--gpu-1` | `1` | GPU for the right-view model |
| `--skip-clean` | `False` | Skip removing previous outputs |

**Pipeline stages:**

1. Train left and right Gaussian Splatting models in parallel (per-camera transforms are optimised during training)
2. Align the two point clouds with ICP and write a combined dataset (`<data_path>/combined/`)
3. Re-train left and right models — a second pass produces tighter cross-view alignment
4. Run ICP alignment a second time on the improved point clouds
5. Train a single merged model (`ihgs-full-merged`) on the combined dataset

---

## Defect Detection

After training, extract Gaussians, align point clouds, and compare renders between a reference and a test object:

```bash
# Extract and align point clouds, render per-frame comparisons
python scripts/load_pipeline.py <pipeline_name_1> <pipeline_name_2> <folder_name>

# Compute per-pixel and SSIM differences between two render sets
python scripts/compare_renders.py --dir1 <renders_A> --dir2 <renders_B>

# Align two point clouds directly with ICP
python scripts/evaluate_merged.py <pcd1.ply> <pcd2.ply>
```

---

## Evaluation

```bash
# Compute PSNR / SSIM / LPIPS for trained models
python scripts/get_psnr_ssim.py

# Aggregate and display results as a table
python scripts/analyze_data.py

# Evaluate left/right split PSNR
python scripts/get_psnr_lr.py <name>
python scripts/analyze_lr.py
```

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `scripts/convert_masks.py` | Convert PNG segmentation masks to grayscale JPG |
| `scripts/filter_frames.py` | Remove outlier frames from `transforms.json` based on mask size changes |

---

## Citation

```bibtex
@inproceedings{qiu2025omniscan,
  title     = {Omni-Scan: Visually-Accurate Digital Twin Object Modeling using a Bimanual Robot with Handoff and Gaussian Splat Merging},
  author    = {Qiu, Tianshuang and Ma, Zehan and El-Refai, Karim and Shah, Hiya and Kerr, Justin and Kim, Chung Min and Goldberg, Ken},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2025}
}
```
