# SplatShot: 3D Face Avatar Generation from a Single Unconstrained Photo

[Hao Liang](https://hliang2.github.io/)¹ &nbsp;·&nbsp; [Zhixuan Ge](TODO)¹ &nbsp;·&nbsp; [Soumendu Majee](TODO)² &nbsp;·&nbsp; [Joanna Li](TODO)¹ &nbsp;·&nbsp; [Ashok Veeraraghavan](TODO)¹ &nbsp;·&nbsp; [Guha Balakrishnan](TODO)¹

¹ Rice University &nbsp;·&nbsp; ² Samsung Research America

[[Paper]](TODO) &nbsp;|&nbsp; [[Project Page]](TODO)

---

![Teaser](assets/teaser.png)

Given a single in-the-wild photo, SplatShot generates a photorealistic 3D Gaussian Splatting (3DGS) face avatar renderable from arbitrary viewpoints — no per-subject training required.

---

## Setup

### 1. Environment

```bash
conda create -n splatshot python=3.10 -y
conda activate splatshot

# PyTorch — adjust the cu118 tag to match your CUDA driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3DGS rasterizer
pip install gsplat

# Remaining dependencies
pip install -r requirements.txt
```

### 2. External model weights

**Face parsing** (BiSeNet, for ControlNet segmentation):

```bash
git clone https://github.com/zllrunning/face-parsing.PyTorch
# Download the pretrained weight to: face-parsing.PyTorch/res/cp/79999_iter.pth
```

IP-Adapter and ControlNet weights are downloaded automatically from HuggingFace Hub on first run.

### 3. Base 3DGS library

SplatShot requires ~300 pre-built NeRSemble base face models (~6 GB total).  
Download them from HuggingFace Hub:

```bash
python download_bases.py --output_dir ./nersemble_bases
```

Then update the two path constants at the top of `inference.py`:

```python
NERSEMBLE_DATA = "/path/to/nersemble-data/EXP-1-head-frame0_export"   # NeRSemble images + cameras
NERSEMBLE_PLY  = "./nersemble_bases"                                    # downloaded PLY files
```

**One-time precomputation** (~5–60 min, done once per machine):

```bash
# Build DINO embedding index for fast base matching (~5 min)
python precompute_base_embeddings.py

# Pre-render ControlNet conditioning assets for all bases (~30–60 min, optional)
python precompute_assets.py
```

If you skip the second step, assets are computed on the fly the first time each base is used (~2 min per new base).

---

## Inference

```bash
CUDA_HOME=/usr/local/cuda python inference.py --image ./photo.jpg
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Path to input photo |
| `--output_dir` | `./output` | Output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--seq_name` | auto | Skip DINO matching; use this NeRSemble sequence directly |
| `--num_views` | all | Subsample to N views (fewer = faster, less 3D coverage) |

Results are written to `output/<image_stem>/`:

```
output/<image_stem>/
├── avatar.ply        — final 3DGS (open in SuperSplat, Gaussian Splatting Viewer, etc.)
├── base.ply          — matched base model before refinement
├── input.jpg         — copy of your input photo
├── diffusion/        — per-view diffusion images + intermediate visualizations
└── cameras/          — COLMAP cameras needed to render the PLY
```

---

## Requirements

| | |
|---|---|
| GPU | 24 GB VRAM recommended (tested on A100) |
| Runtime | ~10–15 min per image (25 steps, all 48 views) |
| Python | 3.10+ |

---

## Project structure

```
SplatShot/
├── inference.py                   — single entry point
├── download_bases.py              — download pre-built base 3DGS models
├── precompute_base_embeddings.py  — one-time: DINO index of base models
├── precompute_assets.py           — one-time: ControlNet assets for all bases
├── requirements.txt
├── core/
│   ├── gs_model.py               — GaussianModel, GaussianRenderer, GaussianTrainer
│   ├── sampler.py                — DDIM sampler with chunked VAE encode/decode
│   ├── diffusion_wrapper.py      — SD 1.5 + ControlNet + IP-Adapter
│   └── semantic_transplant.py   — Semantic Delta Injection (SDI)
├── pipelines/
│   └── _shared_3dgs_guidance.py — 3DGS-guided denoising loop
└── utils/
    ├── colmap.py                 — COLMAP dataset parser
    ├── face_utils.py             — face parsing, landmarks, ArcFace ID
    └── gsplat.py                 — gsplat rasterization helpers
```

---

## Acknowledgments

- [gsplat](https://github.com/nerfstudio-project/gsplat) — 3DGS rasterization
- [Diffusers](https://github.com/huggingface/diffusers) — diffusion pipeline infrastructure
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) — identity conditioning
- [ControlNet](https://github.com/lllyasviel/ControlNet) — pose and segmentation conditioning
- [NeRSemble](https://github.com/tobias-kirschstein/nersemble) — base 3DGS face models
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) — face segmentation
