''

import torch

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape


class VideoConditionByReferenceLatent(ConditioningItem):
    ''

    def __init__(
        self,
        latent: torch.Tensor,
        downscale_factor: int = 1,
        strength: float = 1.0,
    ):
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        ''
        tokens = latent_tools.patchifier.patchify(self.latent)

                                                                       
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.latent.shape),
            device=self.latent.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        )
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

                                                                  
        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor               
            positions[:, 2, ...] *= self.downscale_factor              

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
        )
