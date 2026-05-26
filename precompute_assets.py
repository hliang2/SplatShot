#!/usr/bin/env python3
"""
Precompute face assets for all base sequences referenced in matches.txt.
Renders 48 views per base, runs face detection + parsing, stores final
uint8 control images directly — no recomputation needed at generation time.

Usage:
    python precompute_assets.py

Outputs per view:
    ./precomputed_assets/{seq_name}/view_{i:03d}.npz
        valid           -- bool
        padded_image    -- (H, W, 3) uint8, padded rendered image
        heatmap_control -- (H, W, 3) uint8, ready for ControlNet pose
        parsing_control -- (H, W, 3) uint8, ready for ControlNet seg
        viewmat         -- (4, 4) float32
        K               -- (3, 3) float32
    ./precomputed_assets/{seq_name}/meta.json
        width, height   -- padded dimensions (multiple of 16)
        num_views, valid_views, interp_seed
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.colmap import Parser
from utils.face_utils import (
    FaceParser,
    extract_landmarks_mediapipe,
    landmarks_to_heatmap,
    prepare_heatmap_image,
    prepare_parsing_image,
    pad_to_multiple,
    crop_to_original,
)
from core.gs_model import SimplifiedGaussianModel, GaussianRenderer


# ============================================================
# PATHS
# ============================================================
MATCHES_TXT        = "./matching_results_weighted_unique/matches.txt"
NERSEMBLE_DATA     = "/scratch/shared/nersemble-data/EXP-1-head-frame0_export"
NERSEMBLE_PLY      = "/scratch/shared/nersemble_3dgs_no_normalize"
OUTPUT_DIR         = "./precomputed_assets_48"
TARGET_TOTAL_VIEWS = 48
INTERP_SEED        = 42
# ============================================================


def load_required_bases(txt_path: str) -> list:
    seq_names = set()
    with open(txt_path) as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq_names.add(line.split("\t")[1])
    return sorted(seq_names)


def render_views(gs_model, gs_renderer, parser, device, target_total_views, seed):
    """Render target_total_views from PLY. Returns list of (pil, viewmat_np, K_np)."""
    K = torch.from_numpy(list(parser.Ks_dict.values())[0]).float().to(device)
    width, height = list(parser.imsize_dict.values())[0]

    num_original    = min(len(parser.camtoworlds), target_total_views)
    num_interpolated = target_total_views - num_original

    with torch.no_grad():
        gaussian_params = gs_model.get_gaussian_params()
        gaussian_params = {
            k: v.to(device).contiguous() if torch.is_tensor(v) else v
            for k, v in gaussian_params.items()
        }

    results = []
    camtoworlds = parser.camtoworlds[:num_original]

    for pose in camtoworlds:
        pose_t  = torch.from_numpy(pose).float().to(device).contiguous()
        viewmat = torch.inverse(pose_t).contiguous()
        with torch.no_grad():
            rendered, _, _ = gs_renderer.render(
                gaussian_params, viewmat, K.contiguous(), width, height, sh_degree=3
            )
        rendered_np = (rendered.cpu().numpy() * 255).astype(np.uint8)
        results.append((Image.fromarray(rendered_np), viewmat.cpu().numpy(), K.cpu().numpy()))

    if num_interpolated > 0:
        rng = np.random.RandomState(seed)
        for _ in range(num_interpolated):
            idx1, idx2 = rng.choice(len(camtoworlds), 2, replace=False)
            alpha = rng.uniform(0.3, 0.7)
            pose_interp = alpha * camtoworlds[idx1] + (1 - alpha) * camtoworlds[idx2]
            R = pose_interp[:3, :3]
            U, _, Vt = np.linalg.svd(R)
            pose_interp[:3, :3] = U @ Vt
            pose_t  = torch.from_numpy(pose_interp).float().to(device).contiguous()
            viewmat = torch.inverse(pose_t).contiguous()
            with torch.no_grad():
                rendered, _, _ = gs_renderer.render(
                    gaussian_params, viewmat, K.contiguous(), width, height, sh_degree=3
                )
            rendered_np = (rendered.cpu().numpy() * 255).astype(np.uint8)
            results.append((Image.fromarray(rendered_np), viewmat.cpu().numpy(), K.cpu().numpy()))

    return results


def process_sequence(seq_name, face_parser, gs_renderer, device, out_base):
    seq_out_dir = Path(out_base) / seq_name
    meta_path   = seq_out_dir / "meta.json"

    if meta_path.exists():
        print(f"  [SKIP] {seq_name} already done")
        return True

    target_dir   = Path(NERSEMBLE_DATA) / seq_name
    original_ply = Path(NERSEMBLE_PLY) / seq_name / "ply" / "point_cloud_6999.ply"

    if not target_dir.exists():
        print(f"  [WARN] target dir not found: {target_dir}")
        return False
    if not original_ply.exists():
        print(f"  [WARN] PLY not found: {original_ply}")
        return False

    seq_out_dir.mkdir(parents=True, exist_ok=True)

    parser = Parser(data_dir=str(target_dir), factor=1, normalize=False, test_every=10000000)

    gs_model = SimplifiedGaussianModel(sh_degree=3, device=device)
    gs_model.from_ply_dict(str(original_ply))

    views = render_views(gs_model, gs_renderer, parser, device, TARGET_TOTAL_VIEWS, INTERP_SEED)

    # Track padded dimensions (set on first valid view, consistent across all views)
    padded_width  = None
    padded_height = None
    valid_count   = 0

    for i, (img_pil, viewmat_np, K_np) in enumerate(views):
        img_np = np.array(img_pil)
        padded, padding = pad_to_multiple(img_np, 16)  # (H_pad, W_pad, 3)

        # Face detection
        try:
            landmarks = extract_landmarks_mediapipe(padded)
        except Exception as e:
            print(f"    [SKIP view {i}] Face detection failed: {e}")
            np.savez_compressed(
                seq_out_dir / f"view_{i:03d}.npz",
                valid=np.array(False),
                viewmat=viewmat_np,
                K=K_np
            )
            continue

        # Heatmap -> RGB control image, cropped back to original size
        heatmap     = landmarks_to_heatmap(landmarks, padded.shape[:2], sigma=3.0)
        heatmap_pil = prepare_heatmap_image(heatmap)               # PIL, padded size
        heatmap_pil = crop_to_original(heatmap_pil, padding)       # PIL, original size
        heatmap_ctrl = np.array(heatmap_pil)                       # (H, W, 3) uint8

        # Parsing -> RGB control image, cropped back to original size
        parsing_map  = face_parser.parse(padded)
        parsing_oh   = face_parser.to_one_hot(parsing_map)
        parsing_pil  = prepare_parsing_image(parsing_oh)           # PIL, padded size
        parsing_pil  = crop_to_original(parsing_pil, padding)      # PIL, original size
        parsing_ctrl = np.array(parsing_pil)                       # (H, W, 3) uint8

        # Store padded image (this is what gets encoded as latent)
        # padded shape is (H_pad, W_pad, 3) — multiple of 16
        if padded_width is None:
            padded_height, padded_width = padded.shape[:2]

        np.savez_compressed(
            seq_out_dir / f"view_{i:03d}.npz",
            valid=np.array(True),
            padded_image=padded,          # (H_pad, W_pad, 3) uint8
            heatmap_control=heatmap_ctrl, # (H_orig, W_orig, 3) uint8
            parsing_control=parsing_ctrl, # (H_orig, W_orig, 3) uint8
            viewmat=viewmat_np,
            K=K_np
        )
        valid_count += 1

    with open(meta_path, "w") as f:
        json.dump({
            "seq_name":     seq_name,
            "num_views":    len(views),
            "valid_views":  valid_count,
            "padded_width":  padded_width,
            "padded_height": padded_height,
            "interp_seed":  INTERP_SEED,
        }, f, indent=2)

    print(f"  {seq_name}: {valid_count}/{len(views)} views valid, padded size={padded_width}x{padded_height}")
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nScanning {MATCHES_TXT}...")
    required = load_required_bases(MATCHES_TXT)
    print(f"  {len(required)} unique bases required:")
    for s in required:
        print(f"    {s}")

    print("\nLoading face parser...")
    face_parser = FaceParser()
    gs_renderer = GaussianRenderer(device=device)

    print(f"\nPrecomputing -> {OUTPUT_DIR}")
    success, failed = 0, 0

    for seq_name in tqdm(required, desc="Base sequences"):
        print(f"\n[{seq_name}]")
        try:
            ok = process_sequence(seq_name, face_parser, gs_renderer, device, OUTPUT_DIR)
            success += 1 if ok else 0
            failed  += 0 if ok else 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done. Success={success}  Failed={failed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()