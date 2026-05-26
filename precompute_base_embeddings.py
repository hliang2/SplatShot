#!/usr/bin/env python3
"""
One-time script: precompute DINO embeddings for all base sequences.
Saves embeddings to ./base_embeddings.npz for fast lookup.

Usage:
    python precompute_base_embeddings.py
"""

import sys
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ============================================================
NERSEMBLE_DATA   = "/scratch/shared/nersemble-data/EXP-1-head-frame0_export"
FRONTAL_SUFFIX   = "00000_08.png"
OUTPUT_FILE      = "./base_embeddings.npz"
W_HAIR           = 0.4
W_SKIN           = 0.3
W_SHAPE          = 0.3
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load DINO
print("Loading DINO ViT-B/16...")
dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=True)
dino.eval().to(device)

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def dino_embed(img: Image.Image) -> np.ndarray:
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = dino(x).squeeze(0).cpu().numpy()
    return feat / (np.linalg.norm(feat) + 1e-8)

def hair_embedding(img):
    w, h = img.size
    return dino_embed(img.crop((0, 0, w, int(h * 0.4))).resize((224, 224)))

def skin_embedding(img):
    w, h = img.size
    face = img.crop((int(w*0.2), int(h*0.3), int(w*0.8), int(h*0.7)))
    hsv  = cv2.cvtColor(np.array(face).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    desc = np.concatenate([hsv.mean(axis=(0,1)), hsv.std(axis=(0,1))])
    return desc / (np.linalg.norm(desc) + 1e-8)

def shape_embedding(img):
    w, h = img.size
    return dino_embed(img.crop((int(w*0.2), int(h*0.15), int(w*0.8), int(h*0.85))).resize((224, 224)))

# Scan all base sequences
print(f"\nScanning {NERSEMBLE_DATA}...")
base_dirs = sorted([d for d in Path(NERSEMBLE_DATA).iterdir() if d.is_dir()])
print(f"  Found {len(base_dirs)} sequences")

base_names  = []
base_hair   = []
base_skin   = []
base_shape  = []
skipped     = 0

for d in tqdm(base_dirs, desc="Embedding bases"):
    frontal = d / "images" / FRONTAL_SUFFIX
    if not frontal.exists():
        skipped += 1
        continue
    img = Image.open(frontal).convert("RGB")
    base_names.append(d.name)
    base_hair.append(hair_embedding(img))
    base_skin.append(skin_embedding(img))
    base_shape.append(shape_embedding(img))

print(f"\n  Embedded {len(base_names)} bases ({skipped} skipped — no frontal)")

np.savez(
    OUTPUT_FILE,
    names  = np.array(base_names),
    hair   = np.stack(base_hair),
    skin   = np.stack(base_skin),
    shape  = np.stack(base_shape),
)
print(f"✓ Saved to {OUTPUT_FILE}")
print(f"  names:  {len(base_names)}")
print(f"  hair:   {np.stack(base_hair).shape}")
print(f"  skin:   {np.stack(base_skin).shape}")
print(f"  shape:  {np.stack(base_shape).shape}")