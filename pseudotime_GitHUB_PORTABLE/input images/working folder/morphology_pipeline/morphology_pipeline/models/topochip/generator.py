"""TopoChip flow-based generator interface."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...config import ModelConfig
from .architecture import UNetModel_imcond_wrapper
from . import topochip_utils as tutils


class TopochipFlowGenerator:
    """Thin wrapper that loads a flow-matching UNet and produces samples."""
    def __init__(self, config: ModelConfig) -> None:
        if not config.weights_path:
            raise ValueError("Model weights path must be provided when model integration is enabled.")
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = config.image_size
        self.num_steps = config.num_integration_steps
        self.use_cfg = config.use_cfg
        self.cfg_strength = config.cfg_strength
        self.num_blocks = len(config.channel_mult)
        self.samples_per_env = config.samples_per_env

        channel_mult = tuple(config.channel_mult)
        attention = tuple(config.attention_downsample)

        self.model = UNetModel_imcond_wrapper(
            dim=(3, self.image_size, self.image_size),
            num_res_blocks=config.num_res_blocks,
            num_channels=config.base_channels,
            channel_mult=channel_mult,
            num_heads=config.num_heads,
            num_head_channels=config.num_head_channels,
            attention_ds=attention,
            dropout=config.dropout,
            num_cond_im_channels=config.num_cond_channels,
        )

        # Load network weights (EMA preferred) once at construction time.
        checkpoint = torch.load(Path(config.weights_path), map_location=self.device)
        key = "ema_model" if config.use_ema and "ema_model" in checkpoint else "net_model"
        state = checkpoint[key]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, env_id: str, backgrounds: np.ndarray | torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            cond = self._prepare_conditioning(backgrounds)
            cond_uncond = None
            if self.use_cfg:
                # Unconditional branch used when classifier-free guidance is enabled.
                cond_uncond = torch.zeros_like(cond)
            latent = tutils.sample_prior(
                (cond.shape[0], 3, self.image_size, self.image_size), self.device
            )
            samples = self._integrate(latent, cond, cond_uncond)
            samples = samples.clamp(-1.0, 1.0)
        return samples.cpu().numpy()

    def _prepare_conditioning(self, backgrounds: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(backgrounds, np.ndarray):
            try:
                tensor = torch.from_numpy(backgrounds)
            except Exception:
                tensor = torch.tensor(backgrounds.tolist(), dtype=torch.float32)
        else:
            tensor = backgrounds.clone().detach()
        tensor = tensor.float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.shape[0] in (1, 3):
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 4:
            if tensor.shape[1] not in (1, self.config.num_cond_channels):
                tensor = tensor.unsqueeze(1)
        else:
            raise ValueError("Background tensor must have 2, 3, or 4 dimensions")

        # Resize conditioning images to the model resolution.
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.config.normalize_background == "per-image":
            tensor = tutils.normalize_background_per_image(tensor)
        elif self.config.normalize_background == "none":
            pass
        else:
            raise ValueError(f"Unknown background normalization mode {self.config.normalize_background}")

        if self.use_cfg:
            tensor = tutils.prepare_cfg_conditioning(tensor, prob_no_guidance=0.0)

        return tensor.to(self.device)

    def _integrate(
        self,
        latent: torch.Tensor,
        cond: torch.Tensor,
        cond_uncond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        steps = max(self.num_steps, 1)
        dt = 1.0 / steps
        x = latent
        cond_uncond = cond_uncond.to(self.device) if cond_uncond is not None else None

        for step in range(steps):
            t_val = torch.full((x.shape[0],), step / steps, device=self.device)
            if cond_uncond is None:
                velocity = self.model(t_val, x, im_cond=cond)
            else:
                velocity_guided = self.model(t_val, x, im_cond=cond)
                velocity_uncond = self.model(t_val, x, im_cond=cond_uncond)
                velocity = self.cfg_strength * velocity_guided + (1.0 - self.cfg_strength) * velocity_uncond
            x = x + dt * velocity
        return x

    @staticmethod
    def to_uint8(sample: np.ndarray) -> np.ndarray:
        array = np.clip((sample + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return array
