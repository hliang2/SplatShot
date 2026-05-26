"""
Face Processing Utilities
Functions for face parsing, landmark extraction, and face ID extraction
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
from insightface.app import FaceAnalysis


# Add face-parsing.PyTorch to path
_HERE = os.path.dirname(os.path.abspath(__file__))
FACE_PARSING_DIR = os.path.join(_HERE, '..', 'face-parsing.PyTorch')
if os.path.exists(FACE_PARSING_DIR):
    sys.path.insert(0, FACE_PARSING_DIR)


class FaceParser:
    """Face parsing using BiSeNet"""
    
    def __init__(self, model_path=None):
        from model import BiSeNet

        if model_path is None:
            model_path = os.path.join(FACE_PARSING_DIR, 'res', 'cp', '79999_iter.pth')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        
    def parse(self, image):
        """Parse face image and return segmentation map"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        orig_size = image.size
        image_resized = image.resize((512, 512), Image.BILINEAR)
        
        img_tensor = torch.from_numpy(np.array(image_resized)).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        
        parsing_resized = cv2.resize(
            parsing.astype(np.uint8), 
            orig_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        return parsing_resized
    
    def to_one_hot(self, parsing_map, num_classes=19):
        """Convert parsing map to one-hot encoding"""
        h, w = parsing_map.shape
        one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
        for i in range(num_classes):
            one_hot[i] = (parsing_map == i).astype(np.float32)
        return one_hot


def extract_landmarks_mediapipe(image):
    """Extract 68 landmarks using MediaPipe"""
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in target image")
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        landmarks_468 = np.array([
            [lm.x * w, lm.y * h] for lm in landmarks.landmark
        ])
        
        # MediaPipe 468 to standard 68-point mapping
        indices_68 = [
            *range(0, 17),
            70, 63, 105, 66, 107,
            336, 296, 334, 293, 300,
            168, 6, 197, 195, 5, 4, 1, 19, 94,
            33, 160, 158, 133, 153, 144,
            362, 385, 387, 263, 373, 380,
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308,
            78, 191, 80, 81, 82, 13, 312, 311
        ]
        
        landmarks_68 = landmarks_468[indices_68]
        
        return landmarks_68


def landmarks_to_heatmap(landmarks, image_shape, sigma=2.0):
    """Convert landmarks to heatmap"""
    h, w = image_shape
    num_landmarks = len(landmarks)
    heatmap = np.zeros((h, w, num_landmarks), dtype=np.float32)
    
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    
    for i, (lx, ly) in enumerate(landmarks):
        gaussian = np.exp(-((xx - lx)**2 + (yy - ly)**2) / (2 * sigma**2))
        heatmap[:, :, i] = gaussian
    
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


def extract_face_id_arcface(image):
    """Extract face ID embedding using InsightFace (ArcFace)"""
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = app.get(image_bgr)
    
    if len(faces) == 0:
        raise ValueError("No face detected in source image")
    
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    embedding = face.embedding
    
    return embedding


def prepare_heatmap_image(heatmap):
    """Convert heatmap (H, W, 68) to RGB image for ControlNet"""
    heatmap_sum = heatmap.sum(axis=2)
    heatmap_sum = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-8)
    heatmap_rgb = np.stack([heatmap_sum] * 3, axis=-1)
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
    return Image.fromarray(heatmap_rgb)


def prepare_parsing_image(parsing_one_hot):
    """Convert one-hot parsing (19, H, W) to RGB image for ControlNet"""
    parsing_map = np.argmax(parsing_one_hot, axis=0)
    
    colors = np.array([
        [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
        [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
        [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
        [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
        [0, 51, 0], [255, 153, 51], [0, 204, 0]
    ], dtype=np.uint8)
    
    parsing_rgb = colors[parsing_map]
    return Image.fromarray(parsing_rgb)


def process_target_image(image_path, parser):
    """Extract landmarks and parsing from target image"""
    from .image_utils import pad_to_multiple
    
    print(f"Processing target image: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]
    
    # Pad to multiple of 16
    padded_image, padding = pad_to_multiple(image_np, multiple=16)
    print(f"  Original: {orig_w}x{orig_h} → Padded: {padded_image.shape[1]}x{padded_image.shape[0]}")
    
    # Extract landmarks
    print("  Extracting landmarks...")
    landmarks = extract_landmarks_mediapipe(padded_image)
    heatmap = landmarks_to_heatmap(landmarks, padded_image.shape[:2], sigma=3.0)
    
    # Extract face parsing
    print("  Parsing face...")
    parsing_map = parser.parse(padded_image)
    parsing_one_hot = parser.to_one_hot(parsing_map)
    
    return {
        'padded_image': padded_image,
        'heatmap': heatmap,
        'parsing': parsing_one_hot,
        'padding': padding
    }


def process_source_image(image_path):
    """Extract face ID embedding from source image"""
    print(f"Processing source image: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    print("  Extracting face ID embedding...")
    embedding = extract_face_id_arcface(image_np)
    
    return embedding, image

def pad_to_multiple(image, multiple=16):
    """Pad image to nearest multiple of given value"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        cv2.BORDER_REFLECT
    )
    
    return padded, (top, bottom, left, right)


def crop_to_original(image, padding):
    """Crop padded image back to original size"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    top, bottom, left, right = padding
    h, w = image.shape[:2]
    
    crop_top = top
    crop_bottom = h - bottom
    crop_left = left
    crop_right = w - right
    
    cropped = image[crop_top:crop_bottom, crop_left:crop_right]
    return Image.fromarray(cropped)

def extract_face_mask_from_parsing(
    parsing_one_hot: np.ndarray,  # (19, H, W)
    include_hair: bool = True,
    blur_kernel_size: int = 21,
    blur_sigma: float = 7.0
) -> np.ndarray:
    """
    Extract face mask from parsing map
    
    Parsing classes:
    1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=glasses,
    7=l_ear, 8=r_ear, 9=earring, 10=nose, 11=mouth, 12=u_lip, 13=l_lip,
    14=neck, 15=necklace, 16=cloth, 17=hair, 18=hat
    
    Returns: (H, W) mask in [0, 1] with soft boundaries
    """
    # Face regions (exclude neck=14, cloth=16, background=0)
    face_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # skin + facial features, 14 is neck
    
    if include_hair:
        face_indices.extend([17, 18])  # hair + hat
    
    # Combine regions
    mask = np.zeros(parsing_one_hot.shape[1:], dtype=np.float32)
    for idx in face_indices:
        mask = np.maximum(mask, parsing_one_hot[idx])
    
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Blur for soft boundaries
    if blur_kernel_size > 0:
        mask_blurred = cv2.GaussianBlur(
            mask_uint8, 
            (blur_kernel_size, blur_kernel_size), 
            blur_sigma
        )
        mask = mask_blurred.astype(np.float32) / 255.0
    
    return mask