#!/usr/bin/env python3
"""
SplatShot inference: 3D face avatar from a single unconstrained photo.

Usage:
    python inference.py --image ./photo.jpg --output_dir ./output

Outputs (under output/<image_stem>/):
    avatar.ply          — 3DGS model, open in any splat viewer
    diffusion/          — per-view diffusion images used to build the avatar
    cameras/            — COLMAP camera data needed to render the PLY
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths to data (edit these to match your setup, or pass via --config)
# ---------------------------------------------------------------------------
NERSEMBLE_DATA  = "/scratch/shared/nersemble-data/EXP-1-head-frame0_export"
NERSEMBLE_PLY   = "/scratch/shared/nersemble_3dgs_no_normalize"
BASE_EMBEDDINGS = "./base_embeddings.npz"
PRECOMPUTED_DIR = "./precomputed_assets"

# ---------------------------------------------------------------------------
# Fixed generation hyperparameters (paper defaults)
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    num_steps           = 25,
    strength            = 0.5,
    guidance_scale      = 4.0,
    guidance_weight     = 50000.0,
    guidance_interval   = 1,
    first_step_fit_iters= 1000,
    later_step_fit_iters= 500,
    controlnet_scales   = [0.2, 0.6],
    ip_adapter_scale    = 0.8,
    prompt              = "high quality portrait, detailed face, photorealistic",
    negative_prompt     = "blurry, low quality, distorted, deformed",
    seed                = 42,
    num_views           = None,  # None = use all available (matches MutualSwap default)
)


# ---------------------------------------------------------------------------
# Base selection
# ---------------------------------------------------------------------------

def find_best_base(image_path: str) -> str:
    """Match input photo to the closest base 3DGS via DINO + skin-tone features."""
    assert Path(BASE_EMBEDDINGS).exists(), (
        f"Base embeddings not found: {BASE_EMBEDDINGS}\n"
        "Run:  python precompute_base_embeddings.py"
    )

    W_HAIR, W_SKIN, W_SHAPE = 0.4, 0.3, 0.3

    script = f"""
import sys, numpy as np, torch, cv2
import torchvision.transforms as T
from PIL import Image

dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=True)
dino.eval()
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def embed(img):
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        f = dino(x).squeeze(0).numpy()
    return f / (np.linalg.norm(f) + 1e-8)

img = Image.open(r"{image_path}").convert("RGB")
w, h = img.size
q_hair  = embed(img.crop((0,0,w,int(h*0.4))).resize((224,224)))
q_shape = embed(img.crop((int(w*0.2),int(h*0.15),int(w*0.8),int(h*0.85))).resize((224,224)))
skin    = np.array(img.crop((int(w*0.2),int(h*0.3),int(w*0.8),int(h*0.7)))).astype(np.uint8)
hsv     = cv2.cvtColor(skin, cv2.COLOR_RGB2HSV).astype(np.float32)
q_skin  = np.concatenate([hsv.mean(axis=(0,1)), hsv.std(axis=(0,1))])
q_skin  = q_skin / (np.linalg.norm(q_skin) + 1e-8)

data = np.load(r"{BASE_EMBEDDINGS}", allow_pickle=True)
sims = {W_HAIR}*(data['hair']@q_hair) + {W_SKIN}*(data['skin']@q_skin) + {W_SHAPE}*(data['shape']@q_shape)
best = int(np.argmax(sims))
print(str(data['names'][best]))
print(f"score={{float(sims[best]):.4f}}", file=sys.stderr)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True, check=True
        )
        seq_name = result.stdout.strip().split("\n")[-1].strip()
        score    = result.stderr.strip().split("\n")[-1].strip()
        print(f"  Best match: {seq_name}  ({score})")
        return seq_name
    except subprocess.CalledProcessError as e:
        print(f"  Matcher stderr: {e.stderr[-500:]}")
        raise RuntimeError("DINO matching failed")
    finally:
        Path(tmp).unlink()


def _resolve_seq_dir(base_path: str, name: str) -> Path:
    """Find a sequence directory by name or numeric prefix."""
    base = Path(base_path)
    if (base / name).exists():
        return base / name
    num = name.split("_")[0]
    matches = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(num + "_")]
    if matches:
        return matches[0]
    matches = [d for d in base.iterdir() if d.is_dir() and name in d.name]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No directory matching '{name}' in {base_path}")


# ---------------------------------------------------------------------------
# Asset precomputation
# ---------------------------------------------------------------------------

def _compute_assets(seq_name: str, target_dir: Path, original_ply: Path, device: str):
    """Compute ControlNet controls + camera data for a base sequence."""
    from core.gs_model import SimplifiedGaussianModel, GaussianRenderer
    from utils.colmap import Parser
    from utils.face_utils import (
        FaceParser, extract_landmarks_mediapipe,
        landmarks_to_heatmap, prepare_heatmap_image,
        prepare_parsing_image, pad_to_multiple,
    )

    out_dir = Path(PRECOMPUTED_DIR) / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = Parser(data_dir=str(target_dir), factor=1, normalize=False, test_every=10000000)
    K      = torch.from_numpy(list(parser.Ks_dict.values())[0]).float().to(device)
    width, height = list(parser.imsize_dict.values())[0]

    gs_model    = SimplifiedGaussianModel(sh_degree=3, device=device)
    gs_model.from_ply_dict(str(original_ply))
    gs_renderer = GaussianRenderer(device=device)
    face_parser = FaceParser()

    num_orig = min(len(parser.camtoworlds), 24)
    indices  = np.linspace(0, len(parser.camtoworlds) - 1, num_orig, dtype=int)

    num_saved = 0
    padded_shape = None
    with torch.no_grad():
        gp = gs_model.get_gaussian_params()
        gp = {k: v.to(device).contiguous() if torch.is_tensor(v) else v for k, v in gp.items()}

        for view_idx, cam_idx in enumerate(indices):
            pose_t  = torch.from_numpy(parser.camtoworlds[cam_idx]).float().to(device)
            viewmat = torch.inverse(pose_t).contiguous()

            rendered, _, _ = gs_renderer.render(gp, viewmat, K.contiguous(), width, height, sh_degree=3)
            rendered_np    = (rendered.cpu().numpy() * 255).astype(np.uint8)

            padded, _ = pad_to_multiple(rendered_np, 16)
            padded_shape = padded.shape

            try:
                landmarks = extract_landmarks_mediapipe(padded)
                valid = True
            except Exception:
                valid = False
                landmarks = None

            if valid:
                heatmap  = landmarks_to_heatmap(landmarks, padded.shape[:2], sigma=3.0)
                hmap_img = np.array(prepare_heatmap_image(heatmap)).astype(np.uint8)
                parsing  = face_parser.parse(padded)
                one_hot  = face_parser.to_one_hot(parsing)
                pars_img = np.array(prepare_parsing_image(one_hot)).astype(np.uint8)
            else:
                hmap_img = np.zeros((*padded.shape[:2], 3), dtype=np.uint8)
                pars_img = np.zeros((*padded.shape[:2], 3), dtype=np.uint8)

            np.savez(
                str(out_dir / f"view_{view_idx:03d}.npz"),
                valid           = np.array(valid),
                padded_image    = padded,
                heatmap_control = hmap_img,
                parsing_control = pars_img,
                viewmat         = viewmat.cpu().numpy(),
                K               = K.cpu().numpy(),
            )
            if valid:
                num_saved += 1

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"padded_width": padded_shape[1], "padded_height": padded_shape[0]}, f)

    print(f"  Saved {num_saved}/{len(indices)} valid views → {out_dir}")


def _load_assets(seq_name: str, device: str):
    """Load precomputed ControlNet controls + cameras for a base sequence."""
    seq_dir   = Path(PRECOMPUTED_DIR) / seq_name
    meta_path = seq_dir / "meta.json"

    if not meta_path.exists():
        raise RuntimeError(
            f"No precomputed assets for {seq_name}\n"
            "Run:  python precompute_assets.py"
        )

    with open(meta_path) as f:
        meta = json.load(f)
    width, height = meta["padded_width"], meta["padded_height"]

    target_images, controls, viewmats, Ks = [], [], [], []

    for npz_path in sorted(seq_dir.glob("view_*.npz")):
        data = np.load(npz_path, allow_pickle=False)
        if not bool(data["valid"]):
            continue
        target_images.append(Image.fromarray(data["padded_image"]))
        controls.append([
            _img_to_tensor(data["heatmap_control"], width, height, device),
            _img_to_tensor(data["parsing_control"],  width, height, device),
        ])
        viewmats.append(torch.from_numpy(data["viewmat"]).float())
        Ks.append(torch.from_numpy(data["K"]).float())

    if not target_images:
        raise RuntimeError(f"All views failed face detection for {seq_name}")

    viewmats = torch.stack(viewmats).to(device).contiguous()
    Ks       = torch.stack(Ks).to(device).contiguous()
    return target_images, controls, viewmats, Ks, width, height


def _img_to_tensor(img_np, width, height, device):
    pil = Image.fromarray(img_np)
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SplatShot: 3D face avatar from a single photo"
    )
    ap.add_argument("--image",       required=True,  help="Input face photo (.jpg/.png)")
    ap.add_argument("--output_dir",  default="./output", help="Root output directory")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seq_name",    default=None,
                    help="Skip auto base-selection, use this NeRSemble sequence name directly")
    ap.add_argument("--num_views",   type=int, default=None,
                    help="Subsample to this many views (default: use all available)")
    args = ap.parse_args()

    device     = args.device
    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    # ----------------------------------------------------------------
    # 1. Find best base
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("SplatShot — 3D Avatar Generation")
    print("="*60)

    if args.seq_name:
        seq_name = args.seq_name
        print(f"Base sequence (provided): {seq_name}")
    else:
        print("\nStep 1/5  Finding best base sequence...")
        seq_name = find_best_base(str(image_path))

    target_dir   = _resolve_seq_dir(NERSEMBLE_DATA, seq_name)
    ply_dir      = _resolve_seq_dir(NERSEMBLE_PLY,  seq_name)
    seq_name     = target_dir.name
    original_ply = ply_dir / "ply" / "point_cloud_6999.ply"

    if not original_ply.exists():
        sys.exit(f"Base PLY not found: {original_ply}")

    # ----------------------------------------------------------------
    # 2. Set up output directory
    # ----------------------------------------------------------------
    run_name = image_path.stem
    out_dir  = Path(args.output_dir) / run_name
    diffusion_dir = out_dir / "diffusion"
    cameras_dir   = out_dir / "cameras"
    diffusion_dir.mkdir(parents=True, exist_ok=True)

    # Copy input for reference
    shutil.copy(image_path, out_dir / f"input{image_path.suffix}")
    shutil.copy(original_ply, out_dir / "base.ply")

    # Camera folder: symlink to the COLMAP dataset
    if not cameras_dir.exists():
        cameras_dir.symlink_to(target_dir.resolve())

    print(f"Output directory: {out_dir}")
    print(f"Matched base:     {seq_name}")

    # ----------------------------------------------------------------
    # 3. Load precomputed assets (compute on the fly if missing)
    # ----------------------------------------------------------------
    print("\nStep 2/5  Loading base-model assets...")
    seq_asset_dir = Path(PRECOMPUTED_DIR) / seq_name
    if not (seq_asset_dir / "meta.json").exists():
        print("  Assets not precomputed — computing now (takes ~2 min)...")
        _compute_assets(seq_name, target_dir, original_ply, device)

    target_images, controls, viewmats, Ks, width, height = _load_assets(seq_name, device)
    print(f"  {len(target_images)} views at {width}×{height}")

    # Subsample views if requested.
    # precomputed_assets_48: views 0-15 = 16 real cameras (cam 08 = frontal),
    # views 16-47 = 32 interpolated. Take the first N to preserve the frontal.
    n_views = args.num_views
    if n_views is not None and n_views < len(target_images):
        target_images = target_images[:n_views]
        controls      = controls[:n_views]
        viewmats      = viewmats[:n_views]
        Ks            = Ks[:n_views]
        print(f"  Subsampled to {len(target_images)} views")

    print(f"  Using resolution {width}×{height}")

    # ----------------------------------------------------------------
    # 4. Load models
    # ----------------------------------------------------------------
    print("\nStep 3/5  Loading models...")
    from core.diffusion_wrapper import DiffusionWrapper
    from core.gs_model import SimplifiedGaussianModel, GaussianRenderer
    from core.sampler import GuidedDDIMSampler
    from pipelines._shared_3dgs_guidance import run_3dgs_guided_denoising

    p = _DEFAULTS

    diffusion = DiffusionWrapper(mode="face_swap", device=device)
    diffusion.pipe.set_ip_adapter_scale(p["ip_adapter_scale"])

    source_image = Image.open(image_path).convert("RGB")

    prompt_embeds = diffusion.pipe._encode_prompt(
        p["prompt"], device, 1, True, p["negative_prompt"]
    )
    ip_embeds = diffusion.pipe.prepare_ip_adapter_image_embeds(
        [source_image], None, device, 1, True
    )

    gs_model = SimplifiedGaussianModel(sh_degree=3, device=device)
    gs_model.from_ply_dict(str(original_ply))
    gs_renderer = GaussianRenderer(device=device)
    print(f"  Loaded {gs_model.num_gaussians:,} Gaussians")

    sampler = GuidedDDIMSampler(
        pipe=diffusion.pipe,
        gs_model=gs_model,
        gs_renderer=gs_renderer,
        device=device,
    )

    # ----------------------------------------------------------------
    # 5. Run 3DGS-guided denoising
    # ----------------------------------------------------------------
    print("\nStep 4/5  Running 3DGS-guided denoising...")
    images_final = run_3dgs_guided_denoising(
        sampler=sampler,
        target_images=target_images,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        prompt_embeds=prompt_embeds,
        control_tensors_list=controls,
        ip_embeds_list=ip_embeds,
        controlnet_scales=p["controlnet_scales"],
        num_inference_steps=p["num_steps"],
        strength=p["strength"],
        guidance_scale=p["guidance_scale"],
        guidance_weight=p["guidance_weight"],
        guidance_interval=p["guidance_interval"],
        first_step_fit_iters=p["first_step_fit_iters"],
        later_step_fit_iters=p["later_step_fit_iters"],
        output_dir=diffusion_dir,
        device=device,
        seed=p["seed"],
    )

    # ----------------------------------------------------------------
    # 6. Save outputs
    # ----------------------------------------------------------------
    print("\nStep 5/5  Saving outputs...")

    # Diffusion output images
    for v, img_t in enumerate(images_final):
        img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(diffusion_dir / f"view_{v:03d}.png")

    # PLY
    ply_path = out_dir / "avatar.ply"
    gs_model.save_ply(str(ply_path))

    print("\n" + "="*60)
    print("Done.")
    print(f"  PLY:      {ply_path}")
    print(f"  Diffusion:{diffusion_dir}/")
    print(f"  Cameras:  {cameras_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
