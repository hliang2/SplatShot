"""
3DGS-Guided Denoising Sampler: 3D Reconstructability-Guided Diffusion
Guides diffusion denoising using 3DGS fitting loss gradients
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from PIL import Image
from tqdm import tqdm

from .gs_model import SimplifiedGaussianModel, GaussianRenderer
from fused_ssim import fused_ssim

class GuidedDDIMScheduler:
    """DDIM scheduler for 3DGS-guided denoising"""
    
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        
    def set_timesteps(self, num_inference_steps, device):
        """Set timesteps for inference"""
        self.num_inference_steps = num_inference_steps
        
        # Evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = torch.flip(self.timesteps, [0]).to(device)
        
    def get_alpha_sigma(self, timestep):
        """Get alpha_t and sigma_t for timestep"""
        t = timestep.item()
        alpha_t = self.alphas_cumprod[t].sqrt()
        sigma_t = (1 - self.alphas_cumprod[t]).sqrt()
        return alpha_t, sigma_t
    
    def step(self, noise_pred, timestep, latent, eta=0.0):
        """DDIM step: x_t -> x_{t-1}"""
        # Get current alpha, sigma
        alpha_t, sigma_t = self.get_alpha_sigma(timestep)
        
        # Predict x0
        x0_pred = (latent - sigma_t * noise_pred) / alpha_t
        
        # Get previous timestep
        prev_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        if prev_idx == len(self.timesteps) - 1:
            # Last step
            return x0_pred
        
        prev_timestep = self.timesteps[prev_idx + 1]
        alpha_prev, sigma_prev = self.get_alpha_sigma(prev_timestep)
        
        # DDIM formula
        x_prev = alpha_prev * x0_pred + sigma_prev * noise_pred
        
        return x_prev


class GuidedDDIMSampler:
    """
    3DGS-Guided Diffusion Sampler
    
    Jointly denoises multiple views with 3DGS fitting loss guidance
    """
    
    def __init__(
        self,
        pipe,  # Diffusion pipeline
        gs_model: SimplifiedGaussianModel,
        gs_renderer: GaussianRenderer,
        device: str = "cuda"
    ):
        self.pipe = pipe
        self.gs_model = gs_model
        self.gs_renderer = gs_renderer
        self.device = device
        
        # Setup scheduler
        self.scheduler = GuidedDDIMScheduler()
        
        # 3DGS optimizer (for quick fitting)
        self.setup_gs_optimizer()
        
    def setup_gs_optimizer(self, lr_scale=1.0):
        """Setup optimizer for quick 3DGS fitting"""
        self.gs_optimizers = {
            'means': torch.optim.Adam([self.gs_model.means], lr=1.6e-4 * lr_scale, eps=1e-15),
            'scales': torch.optim.Adam([self.gs_model.scales], lr=5e-3 * lr_scale, eps=1e-15),
            'quats': torch.optim.Adam([self.gs_model.quats], lr=1e-3 * lr_scale, eps=1e-15),
            'opacities': torch.optim.Adam([self.gs_model.opacities], lr=5e-2 * lr_scale, eps=1e-15),
            'sh0': torch.optim.Adam([self.gs_model.sh0], lr=2.5e-3 * lr_scale, eps=1e-15),
        }
        if self.gs_model.shN is not None:
            self.gs_optimizers['shN'] = torch.optim.Adam(
                [self.gs_model.shN], lr=2.5e-3 / 20 * lr_scale, eps=1e-15
            )
    
    def encode_images(self, images: List[Image.Image], chunk_size: int = 16) -> torch.Tensor:
        """Encode PIL images to latents in chunks to avoid OOM."""
        images_np = [np.array(img) for img in images]
        images_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in images_np
        ])
        images_tensor = ((images_tensor - 0.5) / 0.5).to(self.device, dtype=torch.float16)

        chunks = []
        with torch.no_grad():
            for i in range(0, len(images_tensor), chunk_size):
                chunk = images_tensor[i:i + chunk_size]
                lat = self.pipe.vae.encode(chunk).latent_dist.sample()
                lat = lat * self.pipe.vae.config.scaling_factor
                chunks.append(lat.cpu())
        return torch.cat(chunks, dim=0).to(self.device)

    def decode_latents(self, latents: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
        """Decode latents to images in chunks to avoid OOM. Returns (V,3,H,W) in [0,1]."""
        chunks = []
        with torch.no_grad():
            for i in range(0, len(latents), chunk_size):
                chunk = latents[i:i + chunk_size] / self.pipe.vae.config.scaling_factor
                imgs = self.pipe.vae.decode(chunk).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                chunks.append(imgs.cpu())
        return torch.cat(chunks, dim=0).to(self.device)

    def decode_latents_with_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents WITH gradients (for guidance computation)"""
        latents = latents / self.pipe.vae.config.scaling_factor
        
        # NO torch.no_grad() here!
        images = self.pipe.vae.decode(latents).sample
        
        # Denormalize to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return images  # (V, 3, H, W) in [0, 1]
    
    def fit_3dgs_to_images(
        self,
        images: torch.Tensor,  # (V, 3, H, W) in [0, 1]
        viewmats: torch.Tensor,  # (V, 4, 4)
        Ks: torch.Tensor,  # (V, 3, 3)
        width: int,
        height: int,
        num_steps: int = 100,
        view_weights: Optional[torch.Tensor] = None  # (V,) float, unnormalized confidence weights
    ) -> float:
        """
        Fit 3DGS to images for num_steps.
        view_weights: per-view confidence scores. Views with low weight (hallucinated)
                      contribute near-zero gradient. If None, all views are equal.
        Returns final loss.
        """
        losses = []

        for param in [self.gs_model.means, self.gs_model.scales, self.gs_model.quats, 
                      self.gs_model.opacities, self.gs_model.sh0]:
            if param is not None:
                param.data = param.data.to(self.device).contiguous()
        if self.gs_model.shN is not None:
            self.gs_model.shN.data = self.gs_model.shN.data.to(self.device).contiguous()

        # Normalize weights to [0, 1] so the scale of the loss is preserved
        if view_weights is not None:
            view_weights = view_weights.to(self.device).float()
            w_min, w_max = view_weights.min(), view_weights.max()
            if w_max > w_min:
                view_weights = (view_weights - w_min) / (w_max - w_min)
            else:
                view_weights = torch.ones_like(view_weights)
            # Use as sampling distribution too: higher-confidence views selected more often
            sample_probs = view_weights / view_weights.sum()
        else:
            sample_probs = None

        # Move viewmats, Ks to CPU first, then back to device (refresh)
        viewmats = viewmats.cpu().to(self.device).contiguous()
        Ks = Ks.cpu().to(self.device).contiguous()
        
        for step in range(num_steps):
            # Sample view: weighted if confidence available, uniform otherwise
            if sample_probs is not None:
                idx = torch.multinomial(sample_probs, 1).item()
            else:
                idx = torch.randint(0, len(images), (1,)).item()

            image = images[idx].permute(1, 2, 0)  # (H, W, 3)
            viewmat = viewmats[idx]
            K = Ks[idx]

            # Render
            gaussian_params = self.gs_model.get_gaussian_params()
            colors, _, _ = self.gs_renderer.render(
                gaussian_params,
                viewmat,
                K,
                width, height,
                sh_degree=3
            )
            
            # Loss
            l1_loss = torch.nn.functional.l1_loss(colors, image)
            colors_bchw = colors.unsqueeze(0).permute(0, 3, 1, 2).float()
            image_bchw = image.unsqueeze(0).permute(0, 3, 1, 2).float()
            ssim_loss = 1.0 - fused_ssim(colors_bchw, image_bchw, padding="valid")

            loss = 0.8 * l1_loss + 0.2 * ssim_loss

            # Scale loss by view confidence weight (hallucinated views → near-zero gradient)
            if view_weights is not None:
                loss = loss * view_weights[idx]
            
            # Backward
            loss.backward()
            
            # Optimize
            for optimizer in self.gs_optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def compute_3dgs_fitting_loss(
        self,
        images: torch.Tensor,  # (V, 3, H, W) in [0, 1]
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int
    ) -> torch.Tensor:
        """
        Compute 3DGS fitting loss on all views
        Returns scalar loss (differentiable w.r.t. images)
        """
        total_loss = 0.0
       
        
        for v in range(len(images)):
            target = images[v].permute(1, 2, 0)  # (H, W, 3)
            
            # Render
            with torch.no_grad():
                gaussian_params = self.gs_model.get_gaussian_params()
                rendered, _, _ = self.gs_renderer.render(
                    gaussian_params,
                    viewmats[v],
                    Ks[v],
                    width, height,
                    sh_degree=3
                )
            
            # L1 loss
            total_loss += F.l1_loss(rendered, target)
        
        return total_loss / len(images)
    
    def guidance_step(
        self,
        noise_preds: torch.Tensor,  # (V, 4, H/8, W/8) predicted noise
        latents_t: torch.Tensor,  # (V, 4, H/8, W/8) current latents
        timestep: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        guidance_weight: float,
        num_fit_steps: int = 100
    ) -> torch.Tensor:
        """
        Apply 3DGS guidance to noise predictions
        
        Returns: adjusted noise predictions
        """
        # Get alpha_t, sigma_t
        alpha_t, sigma_t = self.scheduler.get_alpha_sigma(timestep)
        
        # Predict x0 from current latents and noise
        x0_pred = (latents_t - sigma_t * noise_preds) / alpha_t
        
        # Decode to images (need gradients!)
        x0_pred.requires_grad_(True)
        images_pred = self.decode_latents(x0_pred)  # (V, 3, H, W)
        
        # Fit 3DGS for a few steps (detached, no gradients to 3DGS params yet)
        with torch.no_grad():
            fit_loss = self.fit_3dgs_to_images(
                images_pred.detach(),
                viewmats, Ks, width, height,
                num_steps=num_fit_steps
            )
            print(f"    3DGS fit loss: {fit_loss:.6f}")
        
        # Compute 3DGS fitting loss (with gradients to images)
        loss_3d = self.compute_3dgs_fitting_loss(
            images_pred,
            viewmats, Ks, width, height
        )
        
        # Compute gradient: Ã¢Ë†â€¡_ÃŽÂµ L_fit = -(ÃÆ’_t/ÃŽÂ±_t) * Ã¢Ë†â€šL_fit/Ã¢Ë†â€šx0
        grad_x0 = torch.autograd.grad(loss_3d, x0_pred, retain_graph=False)[0]
        grad_noise = -(sigma_t / alpha_t) * grad_x0
        
        # Adjust noise prediction
        noise_adjusted = noise_preds - guidance_weight * grad_noise
        
        return noise_adjusted.detach()
    
    def sample(
        self,
        target_images: List[Image.Image],  # V target images (for init)
        source_image: Image.Image,  # Source identity
        control_images_list: List[List[Image.Image]],  # V x 2 (pose, parsing)
        viewmats: torch.Tensor,  # (V, 4, 4)
        Ks: torch.Tensor,  # (V, 3, 3)
        width: int,
        height: int,
        # Sampling params
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        strength: float = 0.6,
        # 3DGS guidance params
        guidance_weight: float = 100.0,
        guidance_interval: int = 5,
        num_fit_steps: int = 100,
        # Other
        prompt: str = "high quality portrait, detailed face, photorealistic",
        negative_prompt: str = "blurry, low quality, distorted, ugly, deformed",
        controlnet_scales: List[float] = [0.2, 0.6],
        ip_adapter_scale: float = 0.8,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Sample with 3D guidance
        
        Returns: V generated images
        """
        V = len(target_images)
        
        print(f"\n3DGS-Guided Denoising:")
        print(f"  Views: {V}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance weight: {guidance_weight}")
        print(f"  Guidance interval: {guidance_interval}")
        print(f"  Fit steps per guidance: {num_fit_steps}")
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        
        # Initialize latents
        latents = self.encode_images(target_images)  # (V, 4, H/8, W/8)
        
        # Add noise according to strength
        start_step = int(num_inference_steps * strength)
        timesteps = self.scheduler.timesteps[start_step:]
        
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn_like(latents)
        latents = self.scheduler.alphas_cumprod[timesteps[0]].sqrt() * latents + \
                  (1 - self.scheduler.alphas_cumprod[timesteps[0]]).sqrt() * noise
        
        # Prepare conditioning (batch)
        # For simplicity, we'll process views one at a time in UNet
        # (batching ControlNet + IP-Adapter is complex)
        
        # Encode prompt
        prompt_embeds = self.pipe._encode_prompt(
            prompt, self.device, 1, True, negative_prompt
        )
        
        # IP-Adapter embeds
        ip_embeds = self.pipe.prepare_ip_adapter_image_embeds(
            [source_image], None, self.device, 1, True
        )[0]
        
        # Prepare control images (V x 2 x 1 x 3 x H x W)
        control_tensors = []
        for control_images in control_images_list:
            control_batch = []
            for img in control_images:
                tensor = self.pipe.prepare_image(
                    img, width, height, 1, 1, self.device, torch.float16
                )
                control_batch.append(tensor)
            control_tensors.append(control_batch)
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Predict noise for all views
            noise_preds = []
            
            for v in range(V):
                latent_v = latents[v:v+1]  # (1, 4, H/8, W/8)
                
                # CFG: duplicate for unconditional
                latent_input = torch.cat([latent_v] * 2)
                
                # ControlNet
                down_samples, mid_sample = self.pipe.controlnet(
                    latent_input, t, prompt_embeds,
                    controlnet_cond=[c[v] for c in [control_tensors[v][0], control_tensors[v][1]]],
                    return_dict=False
                )
                
                # Scale
                down_samples = [s * scale for s, scale in zip(down_samples, controlnet_scales)]
                mid_sample = mid_sample * controlnet_scales[0]
                
                # UNet
                noise_pred = self.pipe.unet(
                    latent_input, t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    added_cond_kwargs={"image_embeds": ip_embeds}
                ).sample
                
                # CFG
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                
                noise_preds.append(noise_pred)
            
            noise_preds = torch.cat(noise_preds, dim=0)  # (V, 4, H/8, W/8)
            
            # Apply 3DGS guidance at selected steps
            if i % guidance_interval == 0 and i > 0:
                print(f"\n  Step {i}/{len(timesteps)}: Applying 3DGS guidance...")
                noise_preds = self.guidance_step(
                    noise_preds, latents, t,
                    viewmats, Ks, width, height,
                    guidance_weight, num_fit_steps
                )
            
            # DDIM step
            for v in range(V):
                latents[v:v+1] = self.scheduler.step(
                    noise_preds[v:v+1], t, latents[v:v+1]
                )
        
        # Decode final latents
        images_tensor = self.decode_latents(latents)  # (V, 3, H, W)
        
        # Convert to PIL
        images_pil = []
        for v in range(V):
            img_np = (images_tensor[v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images_pil.append(Image.fromarray(img_np))
        
        return images_pil