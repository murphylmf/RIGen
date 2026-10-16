''

from dataclasses import replace

import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class ConditioningItemAttentionStrengthWrapper(ConditioningItem):
    ''

    def __init__(
        self,
        conditioning: ConditioningItem,
        attention_mask: float | torch.Tensor,
    ):
        self.conditioning = conditioning
        self.attention_mask = attention_mask

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: LatentTools,
    ) -> LatentState:
        ''
                                                       
        original_state = latent_state

                                                                           
        new_state = self.conditioning.apply_to(latent_state, latent_tools)

        num_new_tokens = new_state.latent.shape[1] - original_state.latent.shape[1]
        if num_new_tokens == 0:
            return new_state

                                                                              
                                                            
        new_attention_mask = update_attention_mask(
            latent_state=original_state,
            attention_mask=self.attention_mask,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=num_new_tokens,
            batch_size=new_state.latent.shape[0],
            device=new_state.latent.device,
            dtype=new_state.latent.dtype,
        )

        return replace(new_state, attention_mask=new_attention_mask)
