''

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ltx_core.types import LatentState


def resolve_cross_mask(
    attention_mask: float | int | torch.Tensor,
    num_new_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ''
    if isinstance(attention_mask, (int, float)):
        return torch.full(
            (batch_size, num_new_tokens),
            fill_value=float(attention_mask),
            device=device,
            dtype=dtype,
        )
    mask = attention_mask.to(device=device, dtype=dtype)

                                                      
    if mask.dim() == 0:
        return torch.full(
            (batch_size, num_new_tokens),
            fill_value=float(mask.item()),
            device=device,
            dtype=dtype,
        )

    if mask.dim() == 1:
        if mask.shape[0] != num_new_tokens:
            raise ValueError(
                f"1-D attention_mask length must equal num_new_tokens ({num_new_tokens}), got shape {tuple(mask.shape)}"
            )
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    elif mask.dim() == 2:
        b, m = mask.shape
        if m != num_new_tokens:
            raise ValueError(
                f"2-D attention_mask second dimension must equal num_new_tokens ({num_new_tokens}), "
                f"got shape {tuple(mask.shape)}"
            )
        if b not in (batch_size, 1):
            raise ValueError(
                f"2-D attention_mask batch dimension must equal batch_size ({batch_size}) or 1, "
                f"got shape {tuple(mask.shape)}"
            )
        if b == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
    else:
        raise ValueError(
            f"attention_mask tensor must be 0-D, 1-D, or 2-D, got {mask.dim()}-D with shape {tuple(mask.shape)}"
        )
    return mask


def update_attention_mask(
    latent_state: LatentState,
    attention_mask: float | torch.Tensor | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    ''
    if attention_mask is None:
        if latent_state.attention_mask is None:
            return None
                                                                            
                                                                            
                          
        cross_mask = torch.ones(batch_size, num_new_tokens, device=device, dtype=dtype)
        return build_attention_mask(
            existing_mask=latent_state.attention_mask,
            num_noisy_tokens=num_noisy_tokens,
            num_new_tokens=num_new_tokens,
            num_existing_tokens=latent_state.latent.shape[1],
            cross_mask=cross_mask,
            device=device,
            dtype=dtype,
        )

    cross_mask = resolve_cross_mask(attention_mask, num_new_tokens, batch_size, device, dtype)
    return build_attention_mask(
        existing_mask=latent_state.attention_mask,
        num_noisy_tokens=num_noisy_tokens,
        num_new_tokens=num_new_tokens,
        num_existing_tokens=latent_state.latent.shape[1],
        cross_mask=cross_mask,
        device=device,
        dtype=dtype,
    )


def build_attention_mask(
    existing_mask: torch.Tensor | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    num_existing_tokens: int,
    cross_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ''
    batch_size = cross_mask.shape[0]
    total = num_existing_tokens + num_new_tokens

                      
    mask = torch.zeros((batch_size, total, total), device=device, dtype=dtype)

                                                                       
    if existing_mask is not None:
        mask[:, :num_existing_tokens, :num_existing_tokens] = existing_mask
    else:
        mask[:, :num_existing_tokens, :num_existing_tokens] = 1.0

                                                                   
    mask[:, num_existing_tokens:, num_existing_tokens:] = 1.0

                                                                   
                                                                                  

                                                                        
                                                       
    mask[:, :num_noisy_tokens, num_existing_tokens:] = cross_mask.unsqueeze(1)

                                                                        
                                                    
    mask[:, num_existing_tokens:, :num_noisy_tokens] = cross_mask.unsqueeze(2)

                                                                                 

    return mask
