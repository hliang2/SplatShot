"""
Utility modules for SplatShot
"""

try:
    from .colmap import Parser
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False

# Face utilities (only import if needed for face swap mode)
try:
    from .face_utils import (
        FaceParser,
        extract_landmarks_mediapipe,
        landmarks_to_heatmap,
        extract_face_id_arcface,
        prepare_heatmap_image,
        prepare_parsing_image,
        process_target_image,
        process_source_image,
        pad_to_multiple, 
        crop_to_original
    )
    FACE_UTILS_AVAILABLE = True
except ImportError:
    FACE_UTILS_AVAILABLE = False

__all__ = [
    'Parser',
    'pad_to_multiple',
    'crop_to_original',
]

if FACE_UTILS_AVAILABLE:
    __all__.extend([
        'FaceParser',
        'extract_landmarks_mediapipe',
        'landmarks_to_heatmap',
        'extract_face_id_arcface',
        'prepare_heatmap_image',
        'prepare_parsing_image',
        'process_target_image',
        'process_source_image',
        'pad_to_multiple', 
        'crop_to_original'
    ])