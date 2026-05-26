#!/usr/bin/env python3
"""
Quick IP-Adapter Token Comparison (Standalone)

Simpler version that doesn't require modifying the full pipeline.
Just loads two images, compares tokens, and runs a basic denoising loop to capture attention.

Usage:
    python quick_token_comparison.py \
        --image1 person_A.jpg \
        --image2 person_B.jpg \
        --output_dir ./token_comparison
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection


def inspect_unet_for_ip_adapter(pipe):
    """
    Debug function to inspect UNet structure and find IP-Adapter components
    """
    print("\n" + "="*70)
    print("INSPECTING UNET FOR IP-ADAPTER LAYERS")
    print("="*70)
    
    # Check top-level
    if hasattr(pipe.unet, 'encoder_hid_proj'):
        print("✓ Found: unet.encoder_hid_proj")
        proj = pipe.unet.encoder_hid_proj
        print(f"  Type: {type(proj).__name__}")
        
        # Try to get shape info
        if hasattr(proj, 'weight'):
            print(f"  Weight shape: {proj.weight.shape}")
        elif hasattr(proj, 'linear_1'):
            print(f"  Has linear_1 (multi-layer projection)")
        
        # Test with dummy input to see output
        try:
            dummy_input = torch.randn(1, 257, 1280, device=pipe.device, dtype=torch.float16)
            with torch.no_grad():
                dummy_output = proj(dummy_input)
            if isinstance(dummy_output, (list, tuple)):
                print(f"  Output: list of {len(dummy_output)} tensors")
                print(f"  First tensor shape: {dummy_output[0].shape}")
            else:
                print(f"  Output shape: {dummy_output.shape}")
        except Exception as e:
            print(f"  Could not test projection: {e}")
    else:
        print("✗ Not found: unet.encoder_hid_proj")
    
    # Check down_blocks
    if hasattr(pipe.unet, 'down_blocks'):
        for block_idx, block in enumerate(pipe.unet.down_blocks):
            if hasattr(block, 'attentions'):
                for attn_idx, attn in enumerate(block.attentions):
                    if hasattr(attn, 'transformer_blocks'):
                        for tb_idx, tb in enumerate(attn.transformer_blocks):
                            if hasattr(tb, 'attn2') and hasattr(tb.attn2, 'processor'):
                                proc = tb.attn2.processor
                                proc_type = type(proc).__name__
                                print(f"\ndown_blocks[{block_idx}].attentions[{attn_idx}].transformer_blocks[{tb_idx}].attn2.processor:")
                                print(f"  Type: {proc_type}")
                                
                                # Check for IP-Adapter attributes
                                if hasattr(proc, 'to_k_ip'):
                                    print(f"  ✓ Has to_k_ip: {proc.to_k_ip.weight.shape if hasattr(proc.to_k_ip, 'weight') else 'N/A'}")
                                if hasattr(proc, 'to_v_ip'):
                                    print(f"  ✓ Has to_v_ip: {proc.to_v_ip.weight.shape if hasattr(proc.to_v_ip, 'weight') else 'N/A'}")
                                if hasattr(proc, 'image_proj'):
                                    print(f"  ✓ Has image_proj")
                                    proj = proc.image_proj
                                    if hasattr(proj, 'weight'):
                                        print(f"    Input: {proj.weight.shape[1]}, Output: {proj.weight.shape[0]}")
    
    # Check mid_block
    if hasattr(pipe.unet, 'mid_block') and hasattr(pipe.unet.mid_block, 'attentions'):
        for attn_idx, attn in enumerate(pipe.unet.mid_block.attentions):
            if hasattr(attn, 'transformer_blocks'):
                for tb_idx, tb in enumerate(attn.transformer_blocks):
                    if hasattr(tb, 'attn2') and hasattr(tb.attn2, 'processor'):
                        proc = tb.attn2.processor
                        proc_type = type(proc).__name__
                        print(f"\nmid_block.attentions[{attn_idx}].transformer_blocks[{tb_idx}].attn2.processor:")
                        print(f"  Type: {proc_type}")
                        
                        if hasattr(proc, 'to_k_ip'):
                            print(f"  ✓ Has to_k_ip")
                        if hasattr(proc, 'image_proj'):
                            print(f"  ✓ Has image_proj")
    
    print("="*70 + "\n")


def load_pipeline(device='cuda'):
    """Load diffusion pipeline with IP-Adapter"""
    print("Loading diffusion pipeline...")
    
    # ControlNets (for face swap, but we'll use minimal conditioning)
    controlnet_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    )
    controlnet_seg = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_seg",
        torch_dtype=torch.float16
    )
    
    # Image encoder for IP-Adapter
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(dtype=torch.float16)
    
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
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V4.0_noVAE",
        controlnet=[controlnet_pose, controlnet_seg],
        image_encoder=image_encoder,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to(device)
    
    # Load IP-Adapter
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin"
    )
    pipe.set_ip_adapter_scale(1.5)
    
    print("✓ Pipeline loaded")
    
    # Inspect UNet structure to find IP-Adapter layers
    inspect_unet_for_ip_adapter(pipe)
    
    return pipe


def extract_ip_adapter_tokens(pipe, image: Image.Image) -> torch.Tensor:
    """
    Extract 16×768 IP-Adapter token embeddings
    
    Flow:
    1. Image → CLIP Vision Encoder → (257, 1280) features
    2. Apply IP-Adapter projection → (16, 768) tokens
    
    The projection layer is stored in the UNet's IP-Adapter modules.
    """
    # Step 1: Get raw CLIP features
    ip_embeds_output = pipe.prepare_ip_adapter_image_embeds(
        [image], None, pipe.device, 1, False
    )
    
    # This returns (1, 1, 257, 1280) or similar
    print(f"    [DEBUG] Raw CLIP features shape: {ip_embeds_output[0].shape}")
    
    # Extract and squeeze: (1, 1, 257, 1280) → (257, 1280)
    clip_features = ip_embeds_output[0]
    while clip_features.dim() > 2:
        clip_features = clip_features.squeeze(0)
    
    print(f"    [DEBUG] Squeezed CLIP features: {clip_features.shape}")
    
    # Step 2: Find and apply IP-Adapter projection layer
    # The projection is stored in the UNet's encoder_hid_proj or in adapter modules
    
    # Try to find projection layer
    projection = None
    
    # Option 1: Check for encoder_hid_proj (common in some IP-Adapter versions)
    if hasattr(pipe.unet, 'encoder_hid_proj'):
        projection = pipe.unet.encoder_hid_proj
        print(f"    [DEBUG] Found projection in unet.encoder_hid_proj")
    
    # Option 2: Check in down_blocks (where adapters are often attached)
    elif hasattr(pipe.unet, 'down_blocks'):
        # Look for projection in first attention block with IP-Adapter
        for block_idx, block in enumerate(pipe.unet.down_blocks):
            if hasattr(block, 'attentions') and len(block.attentions) > 0:
                attn = block.attentions[0]
                if hasattr(attn, 'transformer_blocks') and len(attn.transformer_blocks) > 0:
                    tb = attn.transformer_blocks[0]
                    if hasattr(tb, 'attn2') and hasattr(tb.attn2, 'processor'):
                        processor = tb.attn2.processor
                        # Check if processor has IP-Adapter projection
                        if hasattr(processor, 'to_k_ip') or hasattr(processor, 'image_proj'):
                            if hasattr(processor, 'image_proj'):
                                projection = processor.image_proj
                                print(f"    [DEBUG] Found projection in down_blocks[{block_idx}].processor.image_proj")
                                break
    
    # Option 3: Check mid_block
    if projection is None and hasattr(pipe.unet, 'mid_block'):
        if hasattr(pipe.unet.mid_block, 'attentions') and len(pipe.unet.mid_block.attentions) > 0:
            attn = pipe.unet.mid_block.attentions[0]
            if hasattr(attn, 'transformer_blocks') and len(attn.transformer_blocks) > 0:
                tb = attn.transformer_blocks[0]
                if hasattr(tb, 'attn2') and hasattr(tb.attn2, 'processor'):
                    processor = tb.attn2.processor
                    if hasattr(processor, 'image_proj'):
                        projection = processor.image_proj
                        print(f"    [DEBUG] Found projection in mid_block.processor.image_proj")
    
    # If we found a projection, apply it
    if projection is not None:
        with torch.no_grad():
            # Add batch dimension if needed
            if clip_features.dim() == 2:
                clip_features_batch = clip_features.unsqueeze(0)  # (1, 257, 1280)
            else:
                clip_features_batch = clip_features
            
            # Apply projection
            tokens = projection(clip_features_batch)  # May return list or tensor
            
            # Handle list output (multi-adapter case)
            if isinstance(tokens, (list, tuple)):
                print(f"    [DEBUG] Projection returned list of {len(tokens)} elements")
                tokens = tokens[0]  # Take first adapter
            
            print(f"    [DEBUG] After projection (pre-squeeze): {tokens.shape}")
            
            # Remove batch dimension
            while tokens.dim() > 2:
                tokens = tokens.squeeze(0)  # (16, 768)
            
            print(f"    [DEBUG] Final tokens: {tokens.shape}")
            return tokens
    else:
        # Fallback: If no projection found, we'll have to work with CLIP features
        print(f"    [WARNING] No IP-Adapter projection layer found!")
        print(f"    [WARNING] Falling back to raw CLIP features (257, 1280)")
        print(f"    [WARNING] Results may not be meaningful for token comparison")
        
        # Return CLIP features as-is for now
        return clip_features  # (257, 1280)


def compare_tokens(tokens1: torch.Tensor, tokens2: torch.Tensor) -> List[Dict]:
    """
    Compare two sets of IP-Adapter tokens
    
    Args:
        tokens1, tokens2: (num_tokens, hidden_dim) tensors
    
    Returns:
        comparisons: List of {token_idx, cos_sim, l2_dist}
    """
    print(f"\n  Comparing tokens:")
    print(f"    tokens1 shape: {tokens1.shape}")
    print(f"    tokens2 shape: {tokens2.shape}")
    
    assert tokens1.shape == tokens2.shape, \
        f"Token shapes must match! Got {tokens1.shape} vs {tokens2.shape}"
    
    num_tokens = tokens1.shape[0]
    comparisons = []
    
    for i in range(num_tokens):
        # Extract single token embeddings
        token1 = tokens1[i]  # (hidden_dim,)
        token2 = tokens2[i]  # (hidden_dim,)
        
        # Cosine similarity (expects 1D tensors)
        cos_sim = F.cosine_similarity(
            token1.unsqueeze(0),  # (1, hidden_dim)
            token2.unsqueeze(0),  # (1, hidden_dim)
            dim=1
        ).item()
        
        # L2 distance
        l2_dist = torch.norm(token1 - token2).item()
        
        comparisons.append({
            'token_idx': i,
            'cos_sim': cos_sim,
            'l2_dist': l2_dist
        })
    
    return comparisons


def create_mixed_tokens(
    tokens_A: torch.Tensor,
    tokens_B: torch.Tensor,
    comparisons: List[Dict],
    threshold: float = 0.5
) -> tuple:
    """
    Create mixed tokens for semantic transplant
    
    Strategy:
    - Low similarity (< threshold): "Drivers" → Keep from A (identity/structure)
    - High similarity (>= threshold): "Anchors" → Take from B (style/context)
    
    Args:
        tokens_A: (num_tokens, hidden_dim) - Base subject tokens
        tokens_B: (num_tokens, hidden_dim) - Reference style tokens
        comparisons: List of comparison dicts with cos_sim
        threshold: Similarity threshold (default 0.5)
    
    Returns:
        mixed_tokens: (num_tokens, hidden_dim)
        mask: (num_tokens,) bool tensor - True=from A, False=from B
    """
    num_tokens = tokens_A.shape[0]
    mixed_tokens = torch.zeros_like(tokens_A)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    
    print(f"\n  Creating mixed tokens (threshold={threshold}):")
    print(f"    Low similarity (< {threshold}): Keep from A (Drivers)")
    print(f"    High similarity (>= {threshold}): Take from B (Anchors)")
    print()
    
    for comp in comparisons:
        idx = comp['token_idx']
        sim = comp['cos_sim']
        
        if sim < threshold:
            # Low similarity → Driver → Keep A
            mixed_tokens[idx] = tokens_A[idx]
            mask[idx] = True
            source = "A (Driver)"
        else:
            # High similarity → Anchor → Take B
            mixed_tokens[idx] = tokens_B[idx]
            mask[idx] = False
            source = "B (Anchor)"
        
        print(f"    Token {idx:2d}: sim={sim:.3f} → from {source}")
    
    num_from_A = mask.sum().item()
    num_from_B = (~mask).sum().item()
    print(f"\n  ✓ Mixed tokens: {num_from_A} from A, {num_from_B} from B")
    
    return mixed_tokens, mask


def visualize_token_mixing(
    comparisons: List[Dict],
    mask: torch.Tensor,
    threshold: float,
    save_path: Path
):
    """
    Visualize which tokens come from A vs B
    """
    num_tokens = len(comparisons)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Similarity scores with threshold line
    token_indices = [c['token_idx'] for c in comparisons]
    similarities = [c['cos_sim'] for c in comparisons]
    colors = ['green' if mask[i] else 'orange' for i in range(num_tokens)]
    
    ax1.bar(token_indices, similarities, color=colors, alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    ax1.set_xlabel('Token Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Token Similarity & Source Selection')
    ax1.legend(['Threshold', 'From A (Driver)', 'From B (Anchor)'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(num_tokens))
    
    # Plot 2: Token source visualization
    source_viz = mask.float().cpu().numpy()
    im = ax2.imshow(source_viz.reshape(1, -1), cmap='RdYlGn', aspect='auto', 
                     vmin=0, vmax=1)
    ax2.set_yticks([])
    ax2.set_xticks(range(num_tokens))
    ax2.set_xlabel('Token Index')
    ax2.set_title('Token Source (Green=A/Identity, Red=B/Style)')
    
    # Add text labels
    for i in range(num_tokens):
        label = 'A' if mask[i] else 'B'
        color = 'white' if mask[i] else 'black'
        ax2.text(i, 0, label, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax2, label='From A (1) vs From B (0)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved token mixing visualization: {save_path}")


def print_comparison(comparisons: List[Dict], output_path: Path):
    """Print and save comparison results"""
    
    # Keep original order (0-15), don't sort!
    
    print("\n" + "="*70)
    print("IP-ADAPTER TOKEN COMPARISON")
    print("="*70)
    print(f"{'Token':<8} {'Cosine Sim':<15} {'L2 Distance':<15} {'Type'}")
    print("-"*70)
    
    lines = ["IP-ADAPTER TOKEN COMPARISON", "="*70]
    lines.append(f"{'Token':<8} {'Cosine Sim':<15} {'L2 Distance':<15} {'Type'}")
    lines.append("-"*70)
    
    for item in comparisons:  # Use original order
        idx = item['token_idx']
        cos = item['cos_sim']
        l2 = item['l2_dist']
        
        if cos < 0.85:
            typ = "🔴 IDENTITY"
        elif cos < 0.95:
            typ = "🟡 MIXED"
        else:
            typ = "🟢 BACKGROUND"
        
        line = f"Token {idx:<3} {cos:<15.4f} {l2:<15.2f} {typ}"
        print(line)
        lines.append(line)
    
    print("="*70)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Saved to: {output_path}")


class AttentionCapture:
    """Capture attention maps by replacing the processor"""
    
    def __init__(self):
        self.attention_maps = []
        self.current_timestep = None
        self.original_processors = {}
    
    def register(self, pipe):
        """Replace ALL attn2 processors (like the working script does)"""
        for name, module in pipe.unet.named_modules():
            # Hook into all Cross-Attention (attn2) layers
            if name.endswith("attn2") and hasattr(module, "processor"):
                self.original_processors[name] = module.processor
                # Direct assignment like the working script
                module.processor = AttentionCapturingProcessor(
                    module.processor,
                    name,
                    self
                )
        
        print(f"✓ Replaced {len(self.original_processors)} attn2 processors")
    
    def set_timestep(self, t):
        self.current_timestep = t
    
    def clear(self, pipe):
        # Restore original processors
        for name in self.original_processors:
            # Navigate to the module
            parts = name.split('.')
            current = pipe.unet
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            # Restore processor
            current.processor = self.original_processors[name]


class AttentionCapturingProcessor(torch.nn.Module):
    """Processor wrapper that captures IP-Adapter attention (matches working script)"""
    
    def __init__(self, original_processor, layer_name, capture_obj):
        super().__init__()
        self.original_processor = original_processor
        self.layer_name = layer_name
        self.capture = capture_obj
    
    def _calculate_attention_map(self, attn, hidden_states, key_states, attention_mask=None):
        """Calculate attention map between query and keys"""
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Project Q (always standard)
        query = attn.to_q(hidden_states)
        
        # Project K - check if it's IP-Adapter (short sequence)
        is_likely_image = key_states.shape[1] < 60  # Heuristic
        
        if is_likely_image and hasattr(self.original_processor, "to_k_ip"):
            # Use IP-Adapter projection
            ip_layers = self.original_processor.to_k_ip
            if isinstance(ip_layers, torch.nn.ModuleList):
                key = ip_layers[0](key_states)
            else:
                key = ip_layers(key_states)
        else:
            # Use standard projection
            key = attn.to_k(key_states)
        
        dim = query.shape[-1]
        heads = attn.heads
        dim_head = dim // heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, heads, dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, heads, dim_head).permute(0, 2, 1, 3)
        
        # Compute attention: Q @ K^T / sqrt(d)
        scale = dim_head ** -0.5
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        # Softmax
        attention_probs = attention_scores.softmax(dim=-1)
        
        return attention_probs
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_embeds=None, **kwargs):
        """Process attention and capture IP-Adapter attention"""
        
        # --- 1. UNPACK IP CANDIDATES ---
        ip_candidates = []
        
        if isinstance(encoder_hidden_states, tuple):
            for item in encoder_hidden_states:
                if isinstance(item, torch.Tensor):
                    if item.shape[1] == 16:
                        ip_candidates.append(item)
                
                elif isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, torch.Tensor):
                            if sub_item.ndim == 4 and sub_item.shape[1] == 1:
                                sub_item = sub_item.squeeze(1)
                            if sub_item.shape[1] == 16:
                                ip_candidates.append(sub_item)
        
        elif isinstance(encoder_hidden_states, torch.Tensor):
            if encoder_hidden_states.shape[1] == 16:
                ip_candidates.append(encoder_hidden_states)
        
        # --- 2. STORE IP-ADAPTER ATTENTION ---
        # Capture from up_blocks.2 for 32×32 resolution
        if "up_blocks.2.attentions.0.transformer_blocks.0.attn2" in self.layer_name:
            for idx, embed in enumerate(ip_candidates):
                try:
                    ip_probs = self._calculate_attention_map(attn, hidden_states, embed, attention_mask)
                    if ip_probs is not None:
                        # ip_probs is (B, heads, spatial, 16)
                        ip_probs_avg = ip_probs.mean(dim=1)  # (B, spatial, 16)
                        attention_map = ip_probs_avg[0]  # (spatial, 16)
                        
                        # Auto-detect 2D shape from spatial dimension
                        spatial = attention_map.shape[0]
                        
                        # Common resolutions
                        if spatial == 256:  # 16×16
                            h, w = 16, 16
                        elif spatial == 1024:  # 32×32
                            h, w = 32, 32
                        elif spatial == 1785:  # 51×35
                            h, w = 51, 35
                        elif spatial == 4096:  # 64×64
                            h, w = 64, 64
                        else:
                            sqrt_spatial = int(spatial ** 0.5)
                            if sqrt_spatial * sqrt_spatial == spatial:
                                h, w = sqrt_spatial, sqrt_spatial
                            else:
                                continue
                        
                        attention_2d = attention_map.reshape(h, w, 16)
                        
                        self.capture.attention_maps.append({
                            'timestep': self.capture.current_timestep,
                            'attention': attention_2d.detach().cpu(),
                            'num_tokens': 16
                        })
                
                except Exception as e:
                    pass
        
        # --- 3. Delegate to Original Processor ---
        return self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )




def run_simple_denoising(
    pipe,
    image: Image.Image,
    ip_embeds: torch.Tensor,
    num_steps: int = 4,
    device: str = 'cuda',
    start_from_noise: bool = False,
    strength: float = 0.5
) -> AttentionCapture:
    """
    Run simple denoising loop to capture attention at each timestep
    
    Args:
        pipe: Diffusion pipeline
        image: Input image (for IP-Adapter tokens, and optionally as init image)
        ip_embeds: IP-Adapter embeddings (1, 16, 768)
        num_steps: Number of denoising steps (4 for PeRFlow)
        start_from_noise: If True, generate from pure noise. If False, do img2img from image.
        strength: How much noise to add (0.0 = no change, 1.0 = pure noise)
    
    Returns:
        capture: AttentionCapture with filled attention_maps
    """
    print(f"\nRunning denoising with {num_steps} steps...")
    
    # Setup capture
    capture = AttentionCapture()
    capture.register(pipe)
    
    # Setup scheduler
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    if start_from_noise:
        # Generate from pure noise
        print("  Starting from pure noise (generation mode)")
        latents = torch.randn(
            1, 4, 512 // 8, 512 // 8,
            device=device, dtype=torch.float16
        )
    else:
        # Img2img: encode image and add noise
        print(f"  Starting from image with noise (img2img mode, strength={strength})")
        image_tensor = pipe.image_processor.preprocess(image).to(device, dtype=torch.float16)
        with torch.no_grad():
            latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
        
        # Add noise to latents
        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]
        
        noise = torch.randn_like(latents)
        latents = pipe.scheduler.add_noise(latents, noise, timesteps[0:1])
        
        print(f"  Starting from timestep {timesteps[0].item()}")
    
    # Prepare text embeddings (empty prompt)
    # Use encode_prompt instead of deprecated _encode_prompt
    try:
        # Try new API (returns tuple)
        prompt_embeds_tuple = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=""
        )
        # New API returns (prompt_embeds, negative_prompt_embeds)
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    except AttributeError:
        # Fallback to manual encoding if encode_prompt doesn't exist
        text_inputs = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            text_embeddings = pipe.text_encoder(text_input_ids)[0]
        
        # Duplicate for CFG (unconditional + conditional, both empty)
        prompt_embeds = torch.cat([text_embeddings] * 2)
    
    # Simple denoising loop
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        capture.set_timestep(t.item())
        
        with torch.no_grad():
            # Duplicate for CFG
            latent_input = torch.cat([latents] * 2)
            
            # UNet forward (attention hook fires here!)
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"image_embeds": ip_embeds}
            ).sample
            
            # CFG (no guidance, just take conditional)
            noise_pred = noise_pred.chunk(2)[1]
            
            # DDIM step
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    capture.clear(pipe)
    
    # Decode final latents to image
    print("\nDecoding final image...")
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        image_decoded = pipe.vae.decode(latents).sample
        image_decoded = (image_decoded / 2 + 0.5).clamp(0, 1)
        image_np = image_decoded[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        final_image = Image.fromarray(image_np)
    
    # Store in capture object
    capture.final_image = final_image
    
    if len(capture.attention_maps) == 0:
        print("  ⚠️  WARNING: No attention captured!")
    else:
        print(f"✓ Captured {len(capture.attention_maps)} attention maps")
    
    return capture


def visualize_attention(
    attention: torch.Tensor,
    timestep: int,
    comparisons: List[Dict],
    save_path: Path
):
    """Visualize all tokens at one timestep (in order 0-15, not sorted)"""
    
    # Keep original order 0-15
    num_tokens = attention.shape[2]
    
    # Create grid: 4 columns, enough rows
    num_cols = 4
    num_rows = (num_tokens + num_cols - 1) // num_cols  # Ceiling division
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4.5*num_rows))
    fig.suptitle(f'Timestep {timestep}: IP-Adapter Token Attention ({num_tokens} tokens)', 
                 fontsize=16, y=0.995)
    
    # Flatten axes for easier indexing
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, item in enumerate(comparisons):  # Use original order
        token_idx = item['token_idx']
        cos_sim = item['cos_sim']
        l2_dist = item['l2_dist']
        
        ax = axes_flat[idx]
        
        # Get and normalize attention
        attn = attention[:, :, token_idx].numpy()
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # Plot
        im = ax.imshow(attn_norm, cmap='viridis', aspect='auto')
        
        # Color-coded title
        if cos_sim < 0.85:
            color, label = 'red', 'IDENTITY'
        elif cos_sim < 0.95:
            color, label = 'orange', 'MIXED'
        else:
            color, label = 'green', 'BACKGROUND'
        
        ax.set_title(
            f'Token {token_idx} ({label})\ncos={cos_sim:.3f}, L2={l2_dist:.1f}',
            color=color, fontsize=9, fontweight='bold'
        )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(num_tokens, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path.name}")


def run_semantic_transplant(
    pipe,
    image_base: Image.Image,
    mixed_tokens: torch.Tensor,
    num_steps: int = 4,
    device: str = 'cuda',
    strength: float = 0.8
) -> tuple:
    """
    Run denoising with mixed IP-Adapter tokens for semantic transplant
    
    Args:
        pipe: Diffusion pipeline
        image_base: Base image (for img2img initialization)
        mixed_tokens: (16, 768) mixed token embeddings
        num_steps: Denoising steps
        device: Device
        strength: Noise strength for img2img
    
    Returns:
        (final_image, attention_capture)
    """
    print(f"\nRunning semantic transplant with {num_steps} steps...")
    
    # Setup capture
    capture = AttentionCapture()
    capture.register(pipe)
    
    # Setup scheduler
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Encode base image and add noise (img2img mode)
    print(f"  Starting from base image (img2img mode, strength={strength})")
    image_tensor = pipe.image_processor.preprocess(image_base).to(device, dtype=torch.float16)
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    
    # Add noise
    init_timestep = min(int(num_steps * strength), num_steps)
    t_start = max(num_steps - init_timestep, 0)
    timesteps = timesteps[t_start:]
    
    noise = torch.randn_like(latents)
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[0:1])
    
    print(f"  Starting from timestep {timesteps[0].item()}")
    
    # Prepare text embeddings (NULL TEXT for pure image guidance)
    print("  Using NULL text prompt (pure image guidance)")
    try:
        prompt_embeds_tuple = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=""
        )
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    except AttributeError:
        text_inputs = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            text_embeddings = pipe.text_encoder(text_input_ids)[0]
        
        prompt_embeds = torch.cat([text_embeddings] * 2)
    
    # Prepare mixed IP tokens for CFG: (2, 16, 768)
    # Uncond (negative) = zeros, Cond (positive) = mixed tokens
    print(f"  Preparing mixed IP tokens: {mixed_tokens.shape}")
    mixed_ip_for_cfg = torch.cat([
        torch.zeros(1, 16, 768, device=device, dtype=torch.float16),  # Uncond
        mixed_tokens.unsqueeze(0).to(device, dtype=torch.float16)     # Cond
    ], dim=0)
    
    print(f"  Mixed IP tokens for CFG: {mixed_ip_for_cfg.shape}")
    
    # Combine as tuple: (text_embeds, [ip_embeds])
    # This bypasses prepare_ip_adapter_image_embeds and injects tokens directly
    encoder_hidden_states_combined = (prompt_embeds, [mixed_ip_for_cfg])
    
    # Denoising loop with mixed tokens injected directly
    for i, t in enumerate(tqdm(timesteps, desc="Transplanting")):
        capture.set_timestep(t.item())
        
        with torch.no_grad():
            # Duplicate for CFG
            latent_input = torch.cat([latents] * 2)
            
            # --- THE FIX STARTS HERE ---
            # 1. Save the original config
            orig_encoder_hid_dim_type = pipe.unet.config.encoder_hid_dim_type
            
            # 2. Temporarily set to None to bypass validation & internal projection
            pipe.unet.config.encoder_hid_dim_type = None
            
            try:
                # UNet forward with MIXED tokens via encoder_hidden_states tuple
                noise_pred = pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states_combined,
                    added_cond_kwargs={}  # Valid now because we disabled the check
                ).sample
            finally:
                # 3. Restore config immediately (safety first!)
                pipe.unet.config.encoder_hid_dim_type = orig_encoder_hid_dim_type
            # --- THE FIX ENDS HERE ---

            # CFG (no guidance, just take conditional)
            noise_pred = noise_pred.chunk(2)[1]
            
            # DDIM step
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    capture.clear(pipe)
    
    # Decode final image
    print("\nDecoding transplanted image...")
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        image_decoded = pipe.vae.decode(latents).sample
        image_decoded = (image_decoded / 2 + 0.5).clamp(0, 1)
        image_np = image_decoded[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        final_image = Image.fromarray(image_np)
    
    capture.final_image = final_image
    
    if len(capture.attention_maps) == 0:
        print("  ⚠️  WARNING: No attention captured!")
    else:
        print(f"✓ Captured {len(capture.attention_maps)} attention maps")
    
    return final_image, capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', type=str, required=True, help='Image A (base subject)')
    parser.add_argument('--image2', type=str, required=True, help='Image B (reference style)')
    parser.add_argument('--output_dir', type=str, default='./token_comparison')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_steps', type=int, default=4, help='Denoising steps (default: 4 for PeRFlow)')
    parser.add_argument('--from_noise', action='store_true', help='Generate from pure noise instead of img2img')
    parser.add_argument('--strength', type=float, default=0.8, help='Noise strength for img2img (default: 0.8)')
    
    # NEW: Semantic transplant arguments
    parser.add_argument('--transplant', action='store_true', 
                       help='Run semantic transplant experiment (mix tokens based on similarity)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for token mixing (default: 0.5)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    if args.transplant:
        print("SEMANTIC TRANSPLANT EXPERIMENT")
        print("="*70)
        print(f"  Strategy: Mix tokens based on similarity (threshold={args.threshold})")
        print(f"  Low similarity (< {args.threshold}): Keep from A (Drivers/Identity)")
        print(f"  High similarity (>= {args.threshold}): Take from B (Anchors/Style)")
    else:
        print("IP-ADAPTER TOKEN COMPARISON")
    print("="*70)
    
    # Load images
    print("\n1. Loading images...")
    image1 = Image.open(args.image1).convert('RGB').resize((512, 512))
    image2 = Image.open(args.image2).convert('RGB').resize((512, 512))
    print(f"  ✓ Image A (Base): {args.image1}")
    print(f"  ✓ Image B (Reference): {args.image2}")
    
    # Save input images for reference
    image1.save(output_dir / "input_A_base.png")
    image2.save(output_dir / "input_B_reference.png")
    
    # Load pipeline
    print("\n2. Loading pipeline...")
    pipe = load_pipeline(args.device)
    
    # Extract tokens
    print("\n3. Extracting IP-Adapter tokens...")
    tokens1 = extract_ip_adapter_tokens(pipe, image1)
    tokens2 = extract_ip_adapter_tokens(pipe, image2)
    print(f"  ✓ Tokens: {tokens1.shape}")
    
    # Compare
    print("\n4. Comparing tokens...")
    comparisons = compare_tokens(tokens1, tokens2)
    print_comparison(comparisons, output_dir / "token_comparison.txt")
    
    if args.transplant:
        # ===============================================================
        # SEMANTIC TRANSPLANT EXPERIMENT
        # ===============================================================
        print("\n" + "="*70)
        print("RUNNING SEMANTIC TRANSPLANT")
        print("="*70)
        
        # Create mixed tokens
        print("\n5. Creating mixed tokens...")
        mixed_tokens, mask = create_mixed_tokens(
            tokens1, tokens2, comparisons, threshold=args.threshold
        )
        
        # Visualize mixing strategy
        print("\n6. Visualizing token mixing strategy...")
        visualize_token_mixing(
            comparisons, mask, args.threshold,
            output_dir / "token_mixing_strategy.png"
        )
        
        # Run transplant
        print("\n7. Running semantic transplant...")
        transplant_image, transplant_capture = run_semantic_transplant(
            pipe, image1, mixed_tokens, args.num_steps, args.device, args.strength
        )
        
        # Save transplant result
        transplant_path = output_dir / "transplanted_result.png"
        transplant_image.save(transplant_path)
        print(f"\n  ✓ Saved transplanted image: {transplant_path}")
        
        # Create comparison grid
        print("\n8. Creating comparison grid...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image1)
        axes[0].set_title('Image A (Base Subject)\nDrivers kept', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(transplant_image)
        axes[1].set_title(f'Transplanted Result\n(threshold={args.threshold})', 
                         fontsize=14, fontweight='bold', color='green')
        axes[1].axis('off')
        
        axes[2].imshow(image2)
        axes[2].set_title('Image B (Reference Style)\nAnchors transplanted', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = output_dir / "transplant_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved comparison: {comparison_path}")
        
        # Visualize transplant attention maps
        if len(transplant_capture.attention_maps) > 0:
            print("\n9. Visualizing attention maps from transplant...")
            for item in transplant_capture.attention_maps:
                timestep = item['timestep']
                attention = item['attention']  # (32, 32, 16)
                
                save_path = output_dir / f"transplant_timestep_{timestep:04d}_attention.png"
                visualize_attention(attention, timestep, comparisons, save_path)
        
    else:
        # ===============================================================
        # STANDARD TOKEN COMPARISON (original behavior)
        # ===============================================================
        # Run denoising with image1 to capture attention
        print("\n5. Running denoising to capture attention...")
        ip_embeds = pipe.prepare_ip_adapter_image_embeds(
            [image1], None, pipe.device, 1, True
        )
        
        capture = run_simple_denoising(
            pipe, image1, ip_embeds, args.num_steps, args.device,
            start_from_noise=args.from_noise,
            strength=args.strength
        )
        
        # Visualize
        print("\n6. Creating visualizations...")
        
        # Save final denoised image
        if hasattr(capture, 'final_image'):
            final_path = output_dir / "final_denoised.png"
            capture.final_image.save(final_path)
            print(f"  ✓ Saved final image: {final_path}")
        
        if len(capture.attention_maps) == 0:
            print("  ⚠️  No attention maps to visualize!")
        else:
            print(f"  Captured {len(capture.attention_maps)} attention maps at 32×32 resolution")
            
            for item in capture.attention_maps:
                timestep = item['timestep']
                attention = item['attention']  # (32, 32, 16)
                
                save_path = output_dir / f"timestep_{timestep:04d}_attention.png"
                visualize_attention(attention, timestep, comparisons, save_path)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print(f"📁 Results: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()