#!/usr/bin/env python3
"""
Evaluation script: CSIM, CV-CSIM, FID, CLIP-IQA
Compares diffusion results and rendering results against FFHQ.

Usage:
    python evaluate_metrics.py \
        --data_dir ./celeba3d/celeba3d_0.4_INPUT48 \
        --ffhq_dir /scratch/shared/ffhq/images1024x1024 \
        --output_dir ./eval_results
"""

import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm


VIEWS       = ["view_005.png", "view_006.png", "view_008.png", "view_011.png"]
VIEW_PAIRS  = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]  # C(4,2) = 6 pairs
IDS         = [f"{i:05d}" for i in range(100) if i != 1]
FFHQ_SUBSET = 5000
SEED        = 42


# ================================================================
# Model loaders
# ================================================================

def load_arcface(device):
    """Load InsightFace ArcFace model."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
    return app


def get_arcface_embedding(app, image_pil):
    """Extract ArcFace embedding from PIL image. Returns None if no face detected."""
    img_np = np.array(image_pil.convert("RGB"))
    import cv2
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    faces = app.get(img_bgr)
    if len(faces) == 0:
        return None
    # Use largest face
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    emb  = torch.tensor(face.embedding, dtype=torch.float32)
    return F.normalize(emb, dim=0)


def load_clip_iqa(device):
    """Load CLIP-IQA via torchmetrics."""
    from torchmetrics.multimodal import CLIPImageQualityAssessment
    metric = CLIPImageQualityAssessment(prompts=("quality",)).to(device)
    return metric


def load_inception(device):
    """Load Inception-v3 for FID."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    return fid


# ================================================================
# Image loading helpers
# ================================================================

def load_image_tensor(path, size=299):
    """Load image as (3, size, size) float tensor in [0,1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_image_pil(path):
    return Image.open(path).convert("RGB")


# ================================================================
# CSIM: source ↔ generated view cosine similarity
# ================================================================

def compute_csim(data_dir, view_paths_fn, arc_app, desc="CSIM"):
    """
    Compute average CSIM between source.jpg and each of 4 views.
    view_paths_fn(id_str) → list of 4 Path objects (or None if missing)
    """
    scores = []
    skipped = 0

    for id_str in tqdm(IDS, desc=desc):
        src_path = data_dir / id_str / "source.jpg"
        if not src_path.exists():
            skipped += 1
            continue

        src_emb = get_arcface_embedding(arc_app, load_image_pil(src_path))
        if src_emb is None:
            skipped += 1
            continue

        view_paths = view_paths_fn(id_str)
        for vp in view_paths:
            if vp is None or not vp.exists():
                continue
            gen_emb = get_arcface_embedding(arc_app, load_image_pil(vp))
            if gen_emb is None:
                continue
            sim = float(torch.dot(src_emb, gen_emb).clamp(-1, 1))
            scores.append(sim)

    print(f"  {desc}: {len(scores)} pairs, {skipped} skipped")
    return float(np.mean(scores)) if scores else 0.0


# ================================================================
# CV-CSIM: cross-view consistency (6 pairs per identity)
# ================================================================

def compute_cv_csim(data_dir, view_paths_fn, arc_app, desc="CV-CSIM"):
    scores = []
    skipped = 0

    for id_str in tqdm(IDS, desc=desc):
        view_paths = view_paths_fn(id_str)

        # Extract embeddings for all 4 views
        embs = []
        for vp in view_paths:
            if vp is None or not vp.exists():
                embs.append(None)
                continue
            emb = get_arcface_embedding(arc_app, load_image_pil(vp))
            embs.append(emb)

        # Compute all 6 pairs
        for i, j in VIEW_PAIRS:
            if embs[i] is None or embs[j] is None:
                continue
            sim = float(torch.dot(embs[i], embs[j]).clamp(-1, 1))
            scores.append(sim)

        if all(e is None for e in embs):
            skipped += 1

    print(f"  {desc}: {len(scores)} pairs, {skipped} skipped")
    return float(np.mean(scores)) if scores else 0.0


# ================================================================
# FID
# ================================================================

def compute_fid(data_dir, view_paths_fn, ffhq_dir, fid_metric, device, desc="FID"):
    fid_metric.reset()

    # Real images: FFHQ subset
    print(f"  Loading FFHQ subset ({FFHQ_SUBSET} images)...")
    ffhq_files = sorted(Path(ffhq_dir).glob("**/*.png")) + \
                 sorted(Path(ffhq_dir).glob("**/*.jpg"))
    random.seed(SEED)
    ffhq_files = random.sample(ffhq_files, min(FFHQ_SUBSET, len(ffhq_files)))

    for fp in tqdm(ffhq_files, desc="  FFHQ"):
        try:
            t = load_image_tensor(fp, 299).unsqueeze(0).to(device)
            fid_metric.update(t, real=True)
        except Exception:
            continue

    # Generated images: pool all 4 views from all IDs
    print(f"  Loading generated images...")
    for id_str in tqdm(IDS, desc=f"  {desc} gen"):
        view_paths = view_paths_fn(id_str)
        for vp in view_paths:
            if vp is None or not vp.exists():
                continue
            try:
                t = load_image_tensor(vp, 299).unsqueeze(0).to(device)
                fid_metric.update(t, real=False)
            except Exception:
                continue

    score = float(fid_metric.compute())
    fid_metric.reset()
    return score


# ================================================================
# CLIP-IQA
# ================================================================

def compute_clip_iqa(data_dir, view_paths_fn, clip_metric, device, desc="CLIP-IQA"):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    scores = []
    for id_str in tqdm(IDS, desc=desc):
        view_paths = view_paths_fn(id_str)
        for vp in view_paths:
            if vp is None or not vp.exists():
                continue
            try:
                img = load_image_pil(vp)
                t   = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    score = clip_metric(t)
                # score is dict or tensor depending on torchmetrics version
                if isinstance(score, dict):
                    val = float(list(score.values())[0])
                else:
                    val = float(score.mean())
                scores.append(val)
            except Exception:
                continue

    print(f"  {desc}: {len(scores)} images")
    return float(np.mean(scores)) if scores else 0.0


# ================================================================
# View path functions
# ================================================================

def render_view_paths(data_dir):
    """Rendering results: renderings/view_0X.png"""
    def fn(id_str):
        rend_dir = data_dir / id_str / "renderings"
        return [rend_dir / v for v in VIEWS]
    return fn


def diffusion_view_paths(data_dir):
    """Diffusion results: reverse_only/final_view_0X.png"""
    # Map view names to final_view indices
    # view_005 → final_view_005, etc.
    def fn(id_str):
        rev_dir = data_dir / id_str / "reverse_only"
        paths = []
        for v in VIEWS:
            # v = "view_005.png" → "final_view_005.png"
            fname = "final_" + v
            paths.append(rev_dir / fname)
        return paths
    return fn


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   type=str, required=True)
    ap.add_argument("--ffhq_dir",   type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./eval_results")
    ap.add_argument("--device",     type=str, default="cuda")
    args = ap.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    results = {}

    # ----------------------------------------------------------------
    # Load models
    # ----------------------------------------------------------------
    print("Loading models...")
    print("  ArcFace...")
    arc_app = load_arcface(device)

    print("  CLIP-IQA...")
    clip_metric = load_clip_iqa(device)

    print("  Inception (FID)...")
    fid_metric = load_inception(device)

    print("✓ Models loaded\n")

    # ----------------------------------------------------------------
    # Rendering results
    # ----------------------------------------------------------------
    print("="*60)
    print("RENDERING RESULTS")
    print("="*60)
    render_fn = render_view_paths(data_dir)

    print("\n[1/4] CSIM (rendering)...")
    results["render_csim"] = compute_csim(data_dir, render_fn, arc_app, "CSIM-render")

    print("\n[2/4] CV-CSIM (rendering)...")
    results["render_cv_csim"] = compute_cv_csim(data_dir, render_fn, arc_app, "CV-CSIM-render")

    print("\n[3/4] FID (rendering)...")
    results["render_fid"] = compute_fid(data_dir, render_fn, args.ffhq_dir,
                                         fid_metric, device, "FID-render")

    print("\n[4/4] CLIP-IQA (rendering)...")
    results["render_clip_iqa"] = compute_clip_iqa(data_dir, render_fn,
                                                    clip_metric, device, "CLIP-IQA-render")

    # ----------------------------------------------------------------
    # Diffusion results
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("DIFFUSION RESULTS")
    print("="*60)
    diff_fn = diffusion_view_paths(data_dir)

    print("\n[1/4] CSIM (diffusion)...")
    results["diff_csim"] = compute_csim(data_dir, diff_fn, arc_app, "CSIM-diff")

    print("\n[2/4] CV-CSIM (diffusion)...")
    results["diff_cv_csim"] = compute_cv_csim(data_dir, diff_fn, arc_app, "CV-CSIM-diff")

    print("\n[3/4] FID (diffusion)...")
    results["diff_fid"] = compute_fid(data_dir, diff_fn, args.ffhq_dir,
                                       fid_metric, device, "FID-diff")

    print("\n[4/4] CLIP-IQA (diffusion)...")
    results["diff_clip_iqa"] = compute_clip_iqa(data_dir, diff_fn,
                                                  clip_metric, device, "CLIP-IQA-diff")

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'Rendering':>12} {'Diffusion':>12}")
    print("-"*46)
    print(f"{'CSIM ↑':<20} {results['render_csim']:>12.4f} {results['diff_csim']:>12.4f}")
    print(f"{'CV-CSIM ↑':<20} {results['render_cv_csim']:>12.4f} {results['diff_cv_csim']:>12.4f}")
    print(f"{'FID ↓':<20} {results['render_fid']:>12.2f} {results['diff_fid']:>12.2f}")
    print(f"{'CLIP-IQA ↑':<20} {results['render_clip_iqa']:>12.4f} {results['diff_clip_iqa']:>12.4f}")

    # Save
    out_json = output_dir / "metrics.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_json}")


if __name__ == "__main__":
    main()