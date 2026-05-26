"""
Diffusion Model Wrapper
Handles loading and setup of diffusion pipeline with ControlNet and IP-Adapter
"""

import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection


class DiffusionWrapper:
    """
    Wrapper for diffusion models
    Supports both face_swap mode (ControlNet + IP-Adapter) and text_edit mode (simple img2img)
    """
    
    def __init__(
        self,
        mode: str = "face_swap",  # "face_swap" or "text_edit"
        base_model: str = "SG161222/Realistic_Vision_V4.0_noVAE",
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda"
    ):
        self.mode = mode
        self.device = device
        
        print(f"Loading diffusion pipeline (mode: {mode})...")
        
        if mode == "face_swap":
            self._load_face_swap_pipeline(base_model, vae_model)
        elif mode == "text_edit":
            self._load_text_edit_pipeline(base_model, vae_model)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'face_swap' or 'text_edit'")
        
        print("✓ Diffusion pipeline loaded")
    
    def _load_face_swap_pipeline(self, base_model: str, vae_model: str):
        """Load pipeline with ControlNet + IP-Adapter for face swapping"""
        
        # ControlNets for pose and segmentation
        print("  Loading ControlNets...")
        controlnet_pose = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        )
        controlnet_seg = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_seg",
            torch_dtype=torch.float16
        )
        
        # Image encoder for IP-Adapter
        print("  Loading CLIP image encoder...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16
        )
        
        # VAE
        print("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch.float16)
        vae.requires_grad_(True)
        
        # Scheduler
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        # Pipeline
        print("  Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model,
            controlnet=[controlnet_pose, controlnet_seg],
            image_encoder=image_encoder,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to(self.device)
        
        # Load IP-Adapter weights
        print("  Loading IP-Adapter weights...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus-face_sd15.bin"
            # weight_name="ip-adapter-plus_sd15.bin"
        )
    
    def _load_text_edit_pipeline(self, base_model: str, vae_model: str):
        """Load simple img2img pipeline for text-guided editing"""
        
        # VAE
        print("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch.float16)
        vae.requires_grad_(True)
        
        # Scheduler
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        # Simple img2img pipeline
        print("  Loading Stable Diffusion img2img pipeline...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to(self.device)
