"""
Simplified 3D Gaussian Splatting Model
No densification - fixed number of Gaussians throughout training
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from gsplat.rendering import rasterization

# Import utilities
from scipy.spatial import cKDTree

# Import SSIM from utils
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from fused_ssim import fused_ssim
from utils.gsplat import load_ply

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to spherical harmonics (0th degree)"""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert SH to RGB"""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors distance for scale initialization"""
    from scipy.spatial import cKDTree
    
    points_np = points.cpu().numpy()
    tree = cKDTree(points_np)
    distances, _ = tree.query(points_np, k=k)
    return torch.from_numpy(distances).float()


@dataclass
class GaussianModelParams:
    """Gaussian model parameters"""
    means: torch.Tensor          # (N, 3)
    scales: torch.Tensor         # (N, 3) log scales
    quats: torch.Tensor          # (N, 4) quaternions
    opacities: torch.Tensor      # (N, 1) logit opacities
    sh0: torch.Tensor            # (N, 1, 3) SH degree 0
    shN: Optional[torch.Tensor]  # (N, K, 3) SH degree 1+ (optional)


class SimplifiedGaussianModel:
    """
    Simplified 3DGS model
    Focuses on training and rendering, no densification
    """
    
    def __init__(
        self,
        sh_degree: int = 3,
        device: str = "cuda"
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.device = device
        
        # Will be initialized from point cloud
        self.means = None
        self.scales = None
        self.quats = None
        self.opacities = None
        self.sh0 = None  # SH degree 0 (DC)
        self.shN = None  # SH degree 1+ (rest)
        
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # Use full degree from start
        
    def initialize_from_colmap(self, points: np.ndarray, colors: np.ndarray):
        """
        Initialize from COLMAP point cloud
        
        Args:
            points: (N, 3) xyz
            colors: (N, 3) rgb in [0, 1]
        """
        # Safety check
        if colors.max() > 1.0:
            print(f"WARNING: colors max={colors.max()}, expected [0,1]. Normalizing...")
            colors = colors / 255.0
        
        N = len(points)
        
        # Means
        self.means = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Scales (log space, initialized using KNN)
        print(f"Point cloud bounds: {points.min(axis=0)} to {points.max(axis=0)}")
        print(f"Point cloud extent: {points.max(axis=0) - points.min(axis=0)}")
        
        dist2_avg = (knn(torch.tensor(points), 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        init_scale = 1.0
        self.scales = (
            torch.log(dist_avg * init_scale)
            .unsqueeze(-1)
            .repeat(1, 3)
            .to(self.device)
            .to(torch.float32)
        )
        
        print(f"Scale stats: mean={self.scales.mean():.3f}, std={self.scales.std():.3f}")
        print(f"After exp: mean={torch.exp(self.scales).mean():.3f}")

        # Rotations (identity quaternion [w, x, y, z])
        self.quats = torch.zeros(N, 4, device=self.device)
        self.quats[:, 0] = 1.0
        
        # Opacities (logit space, initialized to ~0.1 after sigmoid)
        self.opacities = torch.logit(torch.full((N, 1), 0.1, device=self.device))
        
        # SH features
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device=self.device)
        self.sh0 = rgb_to_sh(colors_tensor).unsqueeze(1)  # (N, 1, 3)
        
        # Higher SH degrees (initialize to zero)
        if self.sh_degree > 0:
            K = (self.sh_degree + 1) ** 2 - 1
            self.shN = torch.zeros(N, K, 3, device=self.device)
        
        # Make all parameters require gradients
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.opacities.requires_grad = True
        self.sh0.requires_grad = True
        if self.shN is not None:
            self.shN.requires_grad = True
        
        print(f"Initialized {N} Gaussians")
    
    def get_params_list(self):
        """Get list of parameters for optimizer"""
        params = [self.means, self.scales, self.quats, self.opacities, self.sh0]
        if self.shN is not None:
            params.append(self.shN)
        return params
    
    def get_gaussian_params(self):
        """Get processed Gaussian parameters for rendering"""
        means = self.means  # [N, 3] - raw
        quats = self.quats  # [N, 4] - rasterization does normalization internally
        scales = torch.exp(self.scales)  # [N, 3] - exp transform
        opacities = torch.sigmoid(self.opacities).squeeze(-1)  # [N] - sigmoid transform
        
        # Concatenate ALL SH features
        if self.shN is not None:
            colors = torch.cat([self.sh0, self.shN], dim=1)  # [N, 1+K, 3]
        else:
            colors = self.sh0
        
        return {
            'means': means,
            'scales': scales,
            'quats': quats,
            'opacities': opacities,
            'colors': colors,
        }
    
    def save(self, path: Path):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'means': self.means.detach().cpu(),
            'scales': self.scales.detach().cpu(),
            'quats': self.quats.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
            'sh0': self.sh0.detach().cpu(),
            'shN': self.shN.detach().cpu() if self.shN is not None else None,
            'sh_degree': self.sh_degree,
            'active_sh_degree': self.active_sh_degree,
        }, path)
        print(f"Saved model to {path}")

    def from_ply_dict(self, path: Path):
        """Load model from PLY file"""
        
        path = Path(path)
        print(f"Loading PLY from {path}...")
        
        # Load using gsplat utility
        splat_dict = load_ply(str(path))
        
        # Convert dict to model parameters
        self.means = splat_dict['means'].detach().clone().requires_grad_(True)
        self.scales = splat_dict['scales'].detach().clone().requires_grad_(True)
        self.quats = splat_dict['quats'].detach().clone().requires_grad_(True)
        self.opacities = splat_dict['opacities'].detach().clone().requires_grad_(True)
        self.sh0 = splat_dict['sh0'].detach().clone().requires_grad_(True)
        
        if 'shN' in splat_dict and splat_dict['shN'].numel() > 0:
            self.shN = splat_dict['shN'].detach().clone().requires_grad_(True)
        else:
            self.shN = None
        
        # Update sh_degree based on loaded data
        if self.shN is not None:
            num_rest = self.shN.shape[1]
            self.sh_degree = int(np.sqrt(num_rest + 1)) - 1
            self.active_sh_degree = self.sh_degree
        
        print(f"Loaded {self.num_gaussians} Gaussians from PLY")
        print(f"  SH degree: {self.sh_degree}")
    
    def load(self, path: Path):
        """Load model checkpoint"""
        ckpt = torch.load(path, map_location=self.device)
        
        self.means = ckpt['means'].to(self.device).requires_grad_(True)
        self.scales = ckpt['scales'].to(self.device).requires_grad_(True)
        self.quats = ckpt['quats'].to(self.device).requires_grad_(True)
        self.opacities = ckpt['opacities'].to(self.device).requires_grad_(True)
        self.sh0 = ckpt['sh0'].to(self.device).requires_grad_(True)
        self.shN = ckpt['shN'].to(self.device).requires_grad_(True) if ckpt['shN'] is not None else None
        
        self.sh_degree = ckpt['sh_degree']
        self.active_sh_degree = ckpt['active_sh_degree']
        
        print(f"Loaded model from {path}")

    def save_ply(self, path):
        """Save model as a standard 3DGS PLY file."""
        import struct

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        means    = self.means.detach().cpu().float().numpy()     # (N, 3)
        scales   = self.scales.detach().cpu().float().numpy()    # (N, 3)  log-space
        quats    = self.quats.detach().cpu().float().numpy()     # (N, 4)  w,x,y,z
        opacities= self.opacities.detach().cpu().float().numpy().reshape(-1)  # (N,)
        sh0      = self.sh0.detach().cpu().float().numpy()       # (N, 1, 3) or (N, 3)
        shN      = self.shN.detach().cpu().float().numpy() if self.shN is not None else None

        N = means.shape[0]
        sh0_flat = sh0.reshape(N, -1)                              # (N, 3) — same either way
        # Standard 3DGS format is channel-first: all-R, then all-G, then all-B.
        # shN is (N, K, 3); transpose to (N, 3, K) then flatten → (N, 3K).
        shN_flat = shN.transpose(0, 2, 1).reshape(N, -1) if shN is not None else None
        num_sh_rest = shN_flat.shape[1] if shN_flat is not None else 0

        # Build property list (matches the standard gaussian-splatting viewer format)
        props = (
            ['x', 'y', 'z', 'nx', 'ny', 'nz']
            + [f'f_dc_{i}' for i in range(sh0_flat.shape[1])]
            + [f'f_rest_{i}' for i in range(num_sh_rest)]
            + ['opacity']
            + [f'scale_{i}' for i in range(scales.shape[1])]
            + [f'rot_{i}' for i in range(quats.shape[1])]
        )

        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            + "".join(f"property float {p}\n" for p in props)
            + "end_header\n"
        )

        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            zeros3 = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                row = np.concatenate([
                    means[i], zeros3[i],
                    sh0_flat[i],
                    shN_flat[i] if shN_flat is not None else [],
                    [opacities[i]],
                    scales[i],
                    quats[i],
                ]).astype(np.float32)
                f.write(row.tobytes())

        print(f"Saved PLY to {path}")
    
    @property
    def num_gaussians(self):
        return self.means.shape[0]


class GaussianRenderer:
    """Render Gaussians using gsplat rasterization"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def render(
        self,
        gaussian_params: dict,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        sh_degree: Optional[int] = None,
        background: Optional[torch.Tensor] = None,
        background_gaussians: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Render Gaussians from a viewpoint
        """
        if background is None:
            background = torch.ones(3, device=self.device)
        
        # NEW: Handle empty foreground (for bg-only rendering)
        if gaussian_params['means'].shape[0] == 0 and background_gaussians is not None:
            # Only rendering background
            params_to_render = background_gaussians
        elif background_gaussians is not None:
            # Combine bg + fg
            combined_params = {
                'means': torch.cat([
                    background_gaussians['means'].detach(),
                    gaussian_params['means']
                ], dim=0),
                'scales': torch.cat([
                    background_gaussians['scales'].detach(),
                    gaussian_params['scales']
                ], dim=0),
                'quats': torch.cat([
                    background_gaussians['quats'].detach(),
                    gaussian_params['quats']
                ], dim=0),
                'opacities': torch.cat([
                    background_gaussians['opacities'].detach(),
                    gaussian_params['opacities']
                ], dim=0),
                'colors': torch.cat([
                    background_gaussians['colors'].detach(),
                    gaussian_params['colors']
                ], dim=0),
            }
            params_to_render = combined_params
        else:
            params_to_render = gaussian_params
            
        # Rasterize
        render_colors, render_alphas, info = rasterization(
            means=params_to_render['means'],
            quats=params_to_render['quats'],
            scales=params_to_render['scales'],
            opacities=params_to_render['opacities'],
            colors=params_to_render['colors'],
            viewmats=viewmat.unsqueeze(0),  # (1, 4, 4)
            Ks=K.unsqueeze(0),               # (1, 3, 3)
            width=width,
            height=height,
            packed=False,
            absgrad=True,
            sh_degree=sh_degree,
            render_mode="RGB",
            # backgrounds=background.unsqueeze(0),  # (1, 3)
        )

        # Composite with white background
        # white_background = torch.ones_like(render_colors)
        # bg = background.view(1, 1, 1, 3)
        # render_colors = render_colors * render_alphas + white_background * (1 - render_alphas)
        
        # Output shapes: (1, H, W, 3), (1, H, W, 1)
        colors = render_colors[0]
        alphas = render_alphas[0]
        
        return colors, alphas, info


class GaussianTrainer:
    """
    Simplified 3DGS trainer
    No densification - fixed number of Gaussians
    """
    
    def __init__(
        self,
        model: SimplifiedGaussianModel,
        learning_rate: float = 1.6e-4,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.renderer = GaussianRenderer(device=device)
        
        # Create separate optimizers for each parameter
        self.optimizers = {
            'means': torch.optim.Adam([model.means], lr=learning_rate, eps=1e-15),
            'scales': torch.optim.Adam([model.scales], lr=learning_rate * 5e-3/1.6e-4, eps=1e-15),
            'quats': torch.optim.Adam([model.quats], lr=learning_rate * 1e-3/1.6e-4, eps=1e-15),
            'opacities': torch.optim.Adam([model.opacities], lr=learning_rate * 5e-2/1.6e-4, eps=1e-15),
            'sh0': torch.optim.Adam([model.sh0], lr=learning_rate * 2.5e-3/1.6e-4, eps=1e-15),
        }
        
        if model.shN is not None:
            self.optimizers['shN'] = torch.optim.Adam(
                [model.shN], 
                lr=learning_rate * 0.0025 / 20, 
                eps=1e-15
            )
    
    def train_step(
        self,
        image: torch.Tensor,       # (H, W, 3) target image
        viewmat: torch.Tensor,     # (4, 4) camera pose
        K: torch.Tensor,           # (3, 3) intrinsics
        step: int
    ) -> dict:
        """Single training step"""
        # Get current Gaussian parameters
        gaussian_params = self.model.get_gaussian_params()
        H, W = image.shape[:2]
        
        # Render
        colors, alphas, info = self.renderer.render(
            gaussian_params,
            viewmat,
            K,
            W, H,
            sh_degree=3
        )
        
        # Loss
        from fused_ssim import fused_ssim
        
        l1_loss = torch.nn.functional.l1_loss(colors, image)
        colors_bchw = colors.unsqueeze(0).permute(0, 3, 1, 2).float()
        image_bchw = image.unsqueeze(0).permute(0, 3, 1, 2).float()
        ssim_loss = 1.0 - fused_ssim(colors_bchw, image_bchw, padding="valid")

        loss = 0.8 * l1_loss + 0.2 * ssim_loss
            
        # Backward
        loss.backward()
        
        # Optimizer step
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'num_gaussians': self.model.num_gaussians,
        }
    
    def render_view(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int
    ) -> torch.Tensor:
        """Render from a viewpoint (no gradients)"""
        with torch.no_grad():
            gaussian_params = self.model.get_gaussian_params()
            colors, _, _ = self.renderer.render(
                gaussian_params,
                viewmat,
                K,
                width, height,
                sh_degree=3
            )
        return colors



class GaussianTrainerWithDensification:
    """
    3DGS trainer with densification via gsplat DefaultStrategy.

    Key insight from simple_trainer.py: splats must be a torch.nn.ParameterDict,
    not a plain dict. DefaultStrategy.step_post_backward writes resized tensors
    back into the ParameterDict in-place, so self.splats["means"] always returns
    the live tensor. A plain dict returned fresh each call cannot receive these
    in-place updates.
    """

    REFINE_START = 100
    REFINE_EVERY = 100
    REFINE_STOP  = 1500
    RESET_EVERY  = 500

    def __init__(
        self,
        model: 'SimplifiedGaussianModel',
        learning_rate: float = 1.6e-4,
        device: str = "cuda"
    ):
        from gsplat.strategy import DefaultStrategy

        self.model = model
        self.device = device
        self.renderer = GaussianRenderer(device=device)
        self.learning_rate = learning_rate

        # Build a ParameterDict exactly as simple_trainer does.
        # DefaultStrategy writes resized tensors back into this dict in-place.
        params = {
            'means':     torch.nn.Parameter(model.means.detach().to(device)),
            'scales':    torch.nn.Parameter(model.scales.detach().to(device)),
            'quats':     torch.nn.Parameter(model.quats.detach().to(device)),
            'opacities': torch.nn.Parameter(model.opacities.detach().to(device)),
            'sh0':       torch.nn.Parameter(model.sh0.detach().to(device)),
        }
        if model.shN is not None:
            params['shN'] = torch.nn.Parameter(model.shN.detach().to(device))

        self.splats = torch.nn.ParameterDict(params)

        self._build_optimizers()

        self.strategy = DefaultStrategy(
            prune_opa=0.005,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            prune_scale3d=0.1,
            prune_scale2d=0.15,
            refine_start_iter=self.REFINE_START,
            refine_stop_iter=self.REFINE_STOP,
            reset_every=self.RESET_EVERY,
            refine_every=self.REFINE_EVERY,
            absgrad=True,
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def _build_optimizers(self):
        lr = self.learning_rate
        self.optimizers = {
            'means':     torch.optim.Adam([self.splats['means']],     lr=lr,               eps=1e-15),
            'scales':    torch.optim.Adam([self.splats['scales']],    lr=lr * 5e-3/1.6e-4, eps=1e-15),
            'quats':     torch.optim.Adam([self.splats['quats']],     lr=lr * 1e-3/1.6e-4, eps=1e-15),
            'opacities': torch.optim.Adam([self.splats['opacities']], lr=lr * 5e-2/1.6e-4, eps=1e-15),
            'sh0':       torch.optim.Adam([self.splats['sh0']],       lr=lr * 2.5e-3/1.6e-4, eps=1e-15),
        }
        if 'shN' in self.splats:
            self.optimizers['shN'] = torch.optim.Adam(
                [self.splats['shN']], lr=lr * 0.0025 / 20, eps=1e-15
            )

    def _get_gaussian_params_from_splats(self) -> dict:
        """Get processed Gaussian parameters for rendering from self.splats."""
        means     = self.splats['means']
        quats     = self.splats['quats']
        scales    = torch.exp(self.splats['scales'])
        opacities = torch.sigmoid(self.splats['opacities']).squeeze(-1)
        if 'shN' in self.splats:
            colors = torch.cat([self.splats['sh0'], self.splats['shN']], dim=1)
        else:
            colors = self.splats['sh0']
        return {
            'means': means, 'scales': scales, 'quats': quats,
            'opacities': opacities, 'colors': colors,
        }

    def train_step(
        self,
        image: torch.Tensor,    # (H, W, 3) in [0, 1]
        viewmat: torch.Tensor,  # (4, 4)
        K: torch.Tensor,        # (3, 3)
        step: int
    ) -> dict:
        gaussian_params = self._get_gaussian_params_from_splats()
        H, W = image.shape[:2]

        colors, alphas, info = self.renderer.render(
            gaussian_params, viewmat, K, W, H, sh_degree=3
        )

        if 'means2d' in info and info['means2d'].requires_grad:
            info['means2d'].retain_grad()

        # 1. PRE-BACKWARD
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # 2. LOSS + BACKWARD
        l1_loss   = torch.nn.functional.l1_loss(colors, image)
        c_bchw    = colors.unsqueeze(0).permute(0, 3, 1, 2).float()
        i_bchw    = image.unsqueeze(0).permute(0, 3, 1, 2).float()
        ssim_loss = 1.0 - fused_ssim(c_bchw, i_bchw, padding="valid")
        loss      = 0.8 * l1_loss + 0.2 * ssim_loss
        loss.backward()

        # 3. OPTIMIZER STEP
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 4. POST-BACKWARD (densify / prune / reset)
        n_before = len(self.splats['means'])
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
            packed=False,
        )
        n_after = len(self.splats['means'])

        if n_after != n_before:
            print(f"    Densification: {n_before} -> {n_after} Gaussians")

        return {
            'loss': loss.item(),
            'num_gaussians': n_after,
        }