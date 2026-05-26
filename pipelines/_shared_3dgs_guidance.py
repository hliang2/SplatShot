"""
3DGS-guided denoising loop.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from core.sampler import GuidedDDIMSampler
from fused_ssim import fused_ssim


def _move_embeds(embeds, device):
    """Move IP-Adapter embed (tensor or list of tensors) to device."""
    if isinstance(embeds, (list, tuple)):
        return [e.to(device) for e in embeds]
    return embeds.to(device)


def run_3dgs_guided_denoising(
    sampler: GuidedDDIMSampler,
    target_images: List[Image.Image],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    # Conditioning (mode-specific)
    prompt_embeds: torch.Tensor,
    control_tensors_list: Optional[List] = None,  # For face_swap mode
    ip_embeds_list: Optional[torch.Tensor] = None,  # For face_swap mode
    controlnet_scales: Optional[List[float]] = None,
    # Sampling params
    num_inference_steps: int = 50,
    strength: float = 0.5,
    guidance_scale: float = 4.0,
    guidance_interval: int = 5,
    guidance_weight: float = 100.0,
    first_step_fit_iters: int = 1000,
    later_step_fit_iters: int = 200,
    self_recurrence_steps: int = 1,
    output_dir: Path = None,
    device: str = "cuda",
    seed: Optional[int] = None,
    # Semantic Delta Injection params (face_swap only)
    semantic_delta_omega: float = 0.5,
) -> torch.Tensor:
    """
    3DGS-guided denoising across multiple views
    
    Works for both face_swap (with ControlNet+IP-Adapter) and text_edit (simple img2img)
    
    Args:
        sampler: GuidedDDIMSampler instance
        target_images: List of target images
        viewmats: (V, 4, 4) camera matrices
        Ks: (V, 3, 3) intrinsics
        width, height: Image dimensions
        prompt_embeds: Text conditioning
        control_tensors_list: Optional ControlNet conditioning (face_swap mode)
        ip_embeds_list: Optional IP-Adapter conditioning (face_swap mode)
        controlnet_scales: ControlNet conditioning scales
        ... (other params)
    
    Returns:
        images_final: (V, 3, H, W) final generated images
    """
    V = len(target_images)
    
    # Setup output dir
    if output_dir is None:
        output_dir = Path("./3dgs_guidance_output")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "x0_per_step").mkdir(exist_ok=True)
    (output_dir / "renders_per_step").mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("3DGS-GUIDED DENOISING")
    print("="*60)
    print(f"  Views: {V}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Strength: {strength}")
    print(f"  Guidance weight: {guidance_weight}")
    print(f"  Guidance interval: {guidance_interval}")
    print(f"  Self-recurrence: {self_recurrence_steps}")
    print("="*60)
    
    # ===================================================================
    # 1. ENCODE IMAGES & ADD NOISE
    # ===================================================================
    print("\n1. Encoding images and adding noise...")
    
    latents = sampler.encode_images(target_images)

    # Store clean latents and GT noise for Semantic Delta Injection (SDI)
    latents_x_orig = latents.clone()   # x_orig: clean encoded PLY renders (V, 4, H/8, W/8)

    sampler.scheduler.set_timesteps(num_inference_steps, device)
    start_step = int(num_inference_steps * strength)
    timesteps = sampler.scheduler.timesteps[start_step:]
    
    print(f"  Starting from step {start_step}/{num_inference_steps}")
    print(f"  Will denoise for {len(timesteps)} steps")
    print(f"  First timestep: {timesteps[0].item()}")
    
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(latents)
    latents_noise_gt = noise.clone()   # ε_GT: the exact noise added to x_orig

    alpha_t, sigma_t = sampler.scheduler.get_alpha_sigma(timesteps[0])
    latents = alpha_t * latents + sigma_t * noise

    # -----------------------------------------------------------------------
    # Pre-compute per-view IP-Adapter embeddings for the original target faces
    # (used as ε_src conditioning in Semantic Delta Injection).
    # Only needed in face_swap mode (control_tensors_list is not None).
    # -----------------------------------------------------------------------
    ip_embeds_src = None  # Single shared embedding for original target identity (CPU)
    if control_tensors_list is not None:
        print("\n  Pre-computing source IP-Adapter embedding for SDI (single representative view)...")
        # All target views are renders of the same person, so one embedding suffices.
        # Use the first view as the representative.
        with torch.no_grad():
            src_embed = sampler.pipe.prepare_ip_adapter_image_embeds(
                [target_images[0]], None, device, 1, True
            )
        if isinstance(src_embed, (list, tuple)):
            ip_embeds_src = [e.cpu() for e in src_embed]
        else:
            ip_embeds_src = src_embed.cpu()
        torch.cuda.empty_cache()
        print("  Done.")

    # Output dirs for SDI visualizations
    (output_dir / "sdi_x0_per_step").mkdir(exist_ok=True)
    
    # ===================================================================
    # 2. DENOISING LOOP WITH 3DGS GUIDANCE
    # ===================================================================
    print("\n2. Starting denoising with 3DGS guidance...")
    
    for step_idx, t in enumerate(tqdm(timesteps, desc="3DGS-Guided Denoising")):
        
        # Determine if we apply 3DGS guidance this step
        apply_guidance = step_idx % guidance_interval == 0
        
        # Number of self-recurrence iterations
        num_recurrence = self_recurrence_steps if apply_guidance else 1
        
        # Self-recurrence loop
        for recurrence_iter in range(num_recurrence):
            
            if recurrence_iter > 0:
                print(f"    Self-recurrence iteration {recurrence_iter+1}/{num_recurrence}")
            
            # ---------------------------------------------------------------
            # 2a + 2b. PER-VIEW: PREDICT NOISE + SDI GUIDANCE IN ONE PASS
            # ---------------------------------------------------------------
            # Process one view at a time to minimise peak VRAM:
            # for each view — run UNet x2, compute eps_smart, immediately
            # do the 3DGS gradient backprop, accumulate grad, then free.
            # Only eps_tgt.detach() is kept across views (for DDIM step).

            alpha_t_val, sigma_t_val = sampler.scheduler.get_alpha_sigma(t)

            noise_preds       = []          # detached eps_tgt, cpu
            grad_eps_tgt_list = []          # accumulated guidance gradients
            x0_smart_imgs_all = []          # for 3DGS fitting (detached)
            LOSS_SCALE        = 1000.0

            # --- Pass 1: collect eps_tgt, x0_smart_latents, x0_smart_imgs (all detached, no grad) ---
            # Cache x0_smart_v on CPU so Pass 2 needs zero UNet re-runs.
            if apply_guidance and control_tensors_list is not None:
                x0_smart_latents_cpu = []  # (V,) list of (1,4,h,w) cpu tensors

                for v in range(V):
                    torch.cuda.empty_cache()
                    latent_v     = latents[v:v+1]
                    latent_input = torch.cat([latent_v] * 2)

                    with torch.no_grad():
                        control_v = control_tensors_list[v]
                        down_samples, mid_sample = sampler.pipe.controlnet(
                            latent_input, t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_cond=control_v,
                            conditioning_scale=controlnet_scales,
                            return_dict=False
                        )
                        noise_pred_tgt = sampler.pipe.unet(
                            latent_input, t,
                            encoder_hidden_states=prompt_embeds,
                            down_block_additional_residuals=down_samples,
                            mid_block_additional_residual=mid_sample,
                            added_cond_kwargs={"image_embeds": ip_embeds_list}
                        ).sample
                        nu, nc = noise_pred_tgt.chunk(2)
                        eps_tgt = nu + guidance_scale * (nc - nu)
                        noise_preds.append(eps_tgt.cpu())

                        noise_pred_src = sampler.pipe.unet(
                            latent_input, t,
                            encoder_hidden_states=prompt_embeds,
                            down_block_additional_residuals=down_samples,
                            mid_block_additional_residual=mid_sample,
                            added_cond_kwargs={"image_embeds": _move_embeds(ip_embeds_src, device)}
                        ).sample
                        nu_s, nc_s = noise_pred_src.chunk(2)
                        eps_src = nu_s + guidance_scale * (nc_s - nu_s)

                        eps_gt_v  = latents_noise_gt[v:v+1].to(dtype=eps_tgt.dtype)
                        eps_smart = eps_gt_v + semantic_delta_omega * (eps_tgt - eps_src)
                        x0_smart_v = (latent_v.to(dtype=eps_smart.dtype) - sigma_t_val * eps_smart) / alpha_t_val

                        # Cache x0_smart latent on CPU for Pass 2 (avoids re-running UNet)
                        x0_smart_latents_cpu.append(x0_smart_v.cpu())

                        x0_img_v = sampler.decode_latents(x0_smart_v)
                        if x0_img_v.shape[2] != height or x0_img_v.shape[3] != width:
                            x0_img_v = F.interpolate(x0_img_v, size=(height, width), mode='bilinear', align_corners=False)
                        x0_smart_imgs_all.append(x0_img_v.cpu())

                        del down_samples, mid_sample, noise_pred_tgt, nu, nc
                        del noise_pred_src, nu_s, nc_s, eps_src, eps_gt_v, eps_smart, x0_smart_v, x0_img_v
                    del latent_input

                torch.cuda.empty_cache()
                noise_preds_tensor = torch.cat(noise_preds, dim=0).to(device)
                x0_smart_imgs_all  = torch.cat(x0_smart_imgs_all, dim=0).to(device)  # (V, 3, H, W)

                # --- Compute per-view confidence weights from render vs x0_smart L1 ---
                # Lower error = higher confidence. Convert to weights via negation + softmax.
                viewmats_d = viewmats.to(device).contiguous()
                Ks_d = Ks.to(device).contiguous()
                with torch.no_grad():
                    # Ensure all Gaussian params are on device before rendering
                    for param in [sampler.gs_model.means, sampler.gs_model.scales,
                                  sampler.gs_model.quats, sampler.gs_model.opacities,
                                  sampler.gs_model.sh0]:
                        if param is not None:
                            param.data = param.data.to(device).contiguous()
                    if sampler.gs_model.shN is not None:
                        sampler.gs_model.shN.data = sampler.gs_model.shN.data.to(device).contiguous()

                    gaussian_params_pre = sampler.gs_model.get_gaussian_params()
                    per_view_errors = []
                    for v_c in range(V):
                        render_c, _, _ = sampler.gs_renderer.render(
                            gaussian_params_pre, viewmats_d[v_c], Ks_d[v_c], width, height, sh_degree=3
                        )
                        render_c_chw = render_c.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
                        err = F.l1_loss(render_c_chw, x0_smart_imgs_all[v_c:v_c+1], reduction='mean')
                        per_view_errors.append(err.item())
                    per_view_errors = torch.tensor(per_view_errors, device=device)
                    # Higher error → lower weight. Use negative-error softmax for smooth weights.
                    view_weights = torch.softmax(-per_view_errors * 10.0, dim=0)  # (V,)

                # --- Fit 3DGS to x0_smart images (once per guidance step) ---
                if recurrence_iter == 0:
                    fit_iters = first_step_fit_iters if step_idx == 0 else later_step_fit_iters
                    fit_loss  = sampler.fit_3dgs_to_images(
                        x0_smart_imgs_all, viewmats_d, Ks_d, width, height,
                        num_steps=fit_iters, view_weights=view_weights
                    )
                    print(f"\n  [Step {step_idx}, t={t.item()}] fit_loss={fit_loss:.6f}")

                    # Save viz
                    for v_idx in range(min(V, 5)):
                        img_np = (x0_smart_imgs_all[v_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(
                            output_dir / "sdi_x0_per_step" / f"step{step_idx:03d}_view{v_idx:03d}.png"
                        )
                    with torch.no_grad():
                        for v_idx in range(min(V, 5)):
                            gaussian_params = sampler.gs_model.get_gaussian_params()
                            render, _, _ = sampler.gs_renderer.render(
                                gaussian_params, viewmats[v_idx], Ks[v_idx], width, height, sh_degree=3
                            )
                            render_np = (render.cpu().numpy() * 255).astype(np.uint8)
                            Image.fromarray(render_np).save(
                                output_dir / "renders_per_step" / f"step{step_idx:03d}_view{v_idx:03d}.png"
                            )

                del x0_smart_imgs_all
                torch.cuda.empty_cache()

                # --- Pass 2: per-view gradient — VAE decode only, NO UNet re-run ---
                # Load cached x0_smart latent, attach grad, decode, compute loss,
                # backprop to x0_smart_v_leaf, then manually chain to eps_tgt grad.
                from fused_ssim import fused_ssim as _fused_ssim
                for v in range(V):
                    torch.cuda.empty_cache()

                    x0_smart_v_leaf = x0_smart_latents_cpu[v].to(device).requires_grad_(True)
                    image_x0_v = sampler.decode_latents_with_grad(x0_smart_v_leaf)

                    with torch.no_grad():
                        gaussian_params = sampler.gs_model.get_gaussian_params()
                        rendered_v, _, _ = sampler.gs_renderer.render(
                            gaussian_params, viewmats[v], Ks[v], width, height, sh_degree=3
                        )
                        rendered_v = rendered_v.float()

                    target_v      = image_x0_v[0].permute(1, 2, 0).float()
                    rendered_bchw = rendered_v.unsqueeze(0).permute(0, 3, 1, 2)
                    target_bchw   = target_v.unsqueeze(0).permute(0, 3, 1, 2)

                    l1     = F.l1_loss(target_v, rendered_v)
                    ssim   = 1.0 - _fused_ssim(rendered_bchw, target_bchw, padding="valid")
                    loss_v = (0.8 * l1 + 0.2 * ssim) * LOSS_SCALE

                    grad_x0_v = torch.autograd.grad(loss_v, x0_smart_v_leaf, retain_graph=False)[0]
                    grad_x0_v = (grad_x0_v / LOSS_SCALE).detach()

                    # Chain rule: d(loss)/d(eps_tgt) = -(σ/α) * ω * grad_x0
                    grad_eps_v = (-sigma_t_val / alpha_t_val) * semantic_delta_omega * grad_x0_v
                    grad_eps_tgt_list.append(grad_eps_v.cpu())

                    del x0_smart_v_leaf, image_x0_v, target_v, rendered_v
                    del rendered_bchw, target_bchw, l1, ssim, loss_v, grad_x0_v, grad_eps_v

                del x0_smart_latents_cpu
                torch.cuda.empty_cache()

                # Apply guidance to noise_preds
                grad_eps_tgt = torch.cat(grad_eps_tgt_list, dim=0).to(device)
                noise_preds_tensor = noise_preds_tensor - guidance_weight * grad_eps_tgt
                noise_preds = noise_preds_tensor
                del grad_eps_tgt, noise_preds_tensor
                print(f"    SDI guidance applied")

            else:
                # Non-guidance steps or text_edit mode: standard single UNet pass
                noise_preds = []
                for v in range(V):
                    torch.cuda.empty_cache()
                    latent_input = torch.cat([latents[v:v+1]] * 2)
                    with torch.no_grad():
                        if control_tensors_list is not None:
                            control_v = control_tensors_list[v]
                            down_samples, mid_sample = sampler.pipe.controlnet(
                                latent_input, t,
                                encoder_hidden_states=prompt_embeds,
                                controlnet_cond=control_v,
                                conditioning_scale=controlnet_scales,
                                return_dict=False
                            )
                            noise_pred = sampler.pipe.unet(
                                latent_input, t,
                                encoder_hidden_states=prompt_embeds,
                                down_block_additional_residuals=down_samples,
                                mid_block_additional_residual=mid_sample,
                                added_cond_kwargs={"image_embeds": ip_embeds_list}
                            ).sample
                            del down_samples, mid_sample
                        else:
                            noise_pred = sampler.pipe.unet(
                                latent_input, t,
                                encoder_hidden_states=prompt_embeds,
                                return_dict=False
                            )[0]
                        nu, nc = noise_pred.chunk(2)
                        noise_pred = nu + guidance_scale * (nc - nu)
                        del nu, nc
                    noise_preds.append(noise_pred.cpu())
                    del latent_input
                torch.cuda.empty_cache()
                noise_preds_tensor = torch.cat(noise_preds, dim=0).to(device)
                noise_preds = noise_preds_tensor
            
            # ---------------------------------------------------------------
            # 2c. DDIM STEP
            # ---------------------------------------------------------------
            latents_new = torch.zeros_like(latents)
            for v in range(V):
                latents_new[v:v+1] = sampler.scheduler.step(
                    noise_preds[v:v+1], t, latents[v:v+1]
                )
            
            # Self-recurrence: if not last iteration, add noise back
            if recurrence_iter < num_recurrence - 1:
                with torch.no_grad():
                    alpha_t, sigma_t = sampler.scheduler.get_alpha_sigma(t)
                    x0_from_new = (latents_new - sigma_t * noise_preds) / alpha_t
                    noise_for_recurrence = torch.randn_like(latents)
                    latents = alpha_t * x0_from_new + sigma_t * noise_for_recurrence
            else:
                latents = latents_new
    
    # ===================================================================
    # 3. DECODE FINAL RESULTS
    # ===================================================================
    print("\n3. Decoding final results...")
    
    images_final = sampler.decode_latents(latents)

    # Compute final per-view confidence weights from 3DGS renders vs decoded images
    print("  Computing final view confidence weights...")
    viewmats_d = viewmats.to(device).contiguous()
    Ks_d = Ks.to(device).contiguous()
    with torch.no_grad():
        for param in [sampler.gs_model.means, sampler.gs_model.scales,
                      sampler.gs_model.quats, sampler.gs_model.opacities,
                      sampler.gs_model.sh0]:
            if param is not None:
                param.data = param.data.to(device).contiguous()
        if sampler.gs_model.shN is not None:
            sampler.gs_model.shN.data = sampler.gs_model.shN.data.to(device).contiguous()

        gaussian_params_final = sampler.gs_model.get_gaussian_params()
        final_errors = []
        for v_f in range(V):
            render_f, _, _ = sampler.gs_renderer.render(
                gaussian_params_final, viewmats_d[v_f], Ks_d[v_f], width, height, sh_degree=3
            )
            render_f_chw = render_f.permute(2, 0, 1).unsqueeze(0)
            img_f = images_final[v_f:v_f+1]
            if img_f.shape[2] != height or img_f.shape[3] != width:
                img_f = F.interpolate(img_f, size=(height, width), mode='bilinear', align_corners=False)
            err = F.l1_loss(render_f_chw, img_f, reduction='mean')
            final_errors.append(err.item())
        final_errors_t = torch.tensor(final_errors, device=device)
        final_view_weights = torch.softmax(-final_errors_t * 10.0, dim=0)  # (V,)
        print(f"  Weight range: min={final_view_weights.min():.4f}, max={final_view_weights.max():.4f}")

    # 4. POST-DENOISING DENSIFICATION PASS
    # ===================================================================
    print("\n" + "="*60)
    print("4. POST-DENOISING DENSIFICATION PASS")
    print("="*60)
    print("  Fitting 3DGS to final decoded images WITH densification enabled")

    # from core.gs_model import GaussianTrainerWithDensification

    # DENSIFY_ITERS = 2000

    # # Prepare final images as (V, H, W, 3) float32 on device
    # with torch.no_grad():
    #     final_images_hwc = images_final.permute(0, 2, 3, 1).float()  # (V, H, W, 3)
    #     if final_images_hwc.shape[1] != height or final_images_hwc.shape[2] != width:
    #         final_images_hwc = F.interpolate(
    #             final_images_hwc.permute(0, 3, 1, 2),
    #             size=(height, width),
    #             mode='bilinear',
    #             align_corners=False
    #         ).permute(0, 2, 3, 1)
    #     final_images_hwc = final_images_hwc.to(device).contiguous()

    # # viewmats_d / Ks_d already on device from confidence weight computation above

    # densify_trainer = GaussianTrainerWithDensification(
    #     model=sampler.gs_model,
    #     learning_rate=1.6e-4,
    #     device=device
    # )

    # print(f"  Starting Gaussians: {sampler.gs_model.num_gaussians}")
    # print(f"  Training for {DENSIFY_ITERS} steps")
    # print(f"  Densification schedule: start={GaussianTrainerWithDensification.REFINE_START}, "
    #       f"every={GaussianTrainerWithDensification.REFINE_EVERY}, "
    #       f"stop={GaussianTrainerWithDensification.REFINE_STOP}")

    # loss_log = []
    # densify_sample_probs = final_view_weights / final_view_weights.sum()
    # for step in tqdm(range(DENSIFY_ITERS), desc="Densification"):
    #     v_idx = torch.multinomial(densify_sample_probs, 1).item()
    #     result = densify_trainer.train_step(
    #         final_images_hwc[v_idx], viewmats_d[v_idx], Ks_d[v_idx], step
    #     )
    #     loss_log.append(result['loss'])
    #     if step % 200 == 0:
    #         print(f"    Step {step:4d}: loss={result['loss']:.5f}, gaussians={result['num_gaussians']}")

    # print(f"\n  ✓ Densification complete!")
    # print(f"  Final Gaussians: {sampler.gs_model.num_gaussians}")
    # print(f"  Final loss (avg last 100): {np.mean(loss_log[-100:]):.5f}")

    # # Sync densified splats back into gs_model
    # sampler.gs_model.means     = densify_trainer.splats['means'].detach().requires_grad_(True)
    # sampler.gs_model.scales    = densify_trainer.splats['scales'].detach().requires_grad_(True)
    # sampler.gs_model.quats     = densify_trainer.splats['quats'].detach().requires_grad_(True)
    # sampler.gs_model.opacities = densify_trainer.splats['opacities'].detach().requires_grad_(True)
    # sampler.gs_model.sh0       = densify_trainer.splats['sh0'].detach().requires_grad_(True)
    # if 'shN' in densify_trainer.splats:
    #     sampler.gs_model.shN   = densify_trainer.splats['shN'].detach().requires_grad_(True)
    
    return images_final